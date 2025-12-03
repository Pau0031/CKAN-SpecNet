import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


WAVENUMBER_RANGE = np.linspace(4000, 400, 1801)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_results(save_dir, metrics, predictions, config=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(save_dir / "metrics.csv", index=False)

    if predictions is not None:
        y_true, y_pred, y_probs = predictions
        pred_dict = {
            "true_label": y_true,
            "predicted_label": y_pred,
        }
        for i in range(y_probs.shape[1]):
            pred_dict[f"prob_class_{i}"] = y_probs[:, i]
        pred_dict["correct"] = y_true == y_pred
        pd.DataFrame(pred_dict).to_csv(save_dir / "predictions.csv", index=False)

    if config is not None:
        pd.DataFrame([config]).to_csv(save_dir / "config.csv", index=False)


def plot_spectrum(spectrum, wavenumbers=None, title="IR Spectrum", save_path=None):
    if wavenumbers is None:
        wavenumbers = WAVENUMBER_RANGE

    plt.figure(figsize=(12, 4))
    plt.plot(wavenumbers, spectrum, "b-", linewidth=1)
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Absorbance")
    plt.title(title)
    plt.gca().invert_xaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix", save_path=None):
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    if "train_acc" in history:
        axes[1].plot(history["train_acc"], label="Train")
    if "val_acc" in history:
        axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def extract_kan_importance(model, input_size=1801):
    importance = np.zeros(input_size)
    count = 0

    for name, module in model.named_modules():
        if hasattr(module, "spline_weight"):
            weights = module.spline_weight.detach().cpu().numpy()
            if weights.ndim == 3:
                imp = np.abs(weights).mean(axis=(0, 2))
                if len(imp) == input_size:
                    importance += imp
                    count += 1
            elif weights.ndim == 2:
                imp = np.abs(weights).mean(axis=1)
                if len(imp) == input_size:
                    importance += imp
                    count += 1

    if count > 0:
        importance /= count

    return importance


def plot_importance_spectrum(spectrum, importance, wavenumbers=None, title="", save_path=None):
    if wavenumbers is None:
        wavenumbers = WAVENUMBER_RANGE

    fig, ax = plt.subplots(figsize=(14, 5))

    importance_2d = importance.reshape(1, -1)
    y_min, y_max = spectrum.min(), spectrum.max()
    ax.imshow(
        importance_2d,
        aspect="auto",
        cmap="YlOrRd",
        alpha=0.4,
        extent=[wavenumbers[0], wavenumbers[-1], y_min, y_max],
        origin="lower",
    )

    ax.plot(wavenumbers, spectrum, "b-", linewidth=1.5, label="Spectrum")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Absorbance")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
