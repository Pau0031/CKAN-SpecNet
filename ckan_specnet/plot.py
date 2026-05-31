from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ckan_specnet.core import COMMON_X, Config, TaskCatalog, clear_cuda
from ckan_specnet.data import Preprocessor, make_loader
from ckan_specnet.eval import load_fold_model, load_manifest, predict


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "figure.titlesize": 20,
        }
    )


def load_plot_ensemble(
    run_dir: Path,
    tasks: TaskCatalog,
    fold_files: list[str] | None = None,
) -> dict:
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    fold_files = fold_files or manifest["ensemble"]["fold_files"]
    normalize = manifest["config"]["normalize"]

    models = []
    checkpoints = []

    for fold_file in fold_files:
        model, ckpt = load_fold_model(run_dir / fold_file, tasks)
        models.append(model)
        checkpoints.append(ckpt)

    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "models": models,
        "checkpoints": checkpoints,
        "preprocessors": [Preprocessor(normalize) for _ in models],
        "fold_files": fold_files,
    }


@torch.inference_mode()
def predict_plot_ensemble(
    models: list[nn.Module],
    preprocessors: list[Preprocessor],
    x: np.ndarray,
    tasks: TaskCatalog,
    cfg: Config,
    device: torch.device,
) -> dict:
    fold_probs = []

    for i, (model, preprocessor) in enumerate(
        zip(models, preprocessors, strict=True), start=1
    ):
        print(f"Predicting fold {i}/{len(models)}")

        model = model.to(device)
        model.eval()

        loader = make_loader(
            x=x,
            y=None,
            preprocessor=preprocessor,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            device=device,
        )

        _, _, prob = predict(model, loader, tasks, device)
        fold_probs.append(prob)

        model.to("cpu")
        clear_cuda()

    prob = {
        task: np.mean([fold_prob[task] for fold_prob in fold_probs], axis=0)
        for task in tasks.all
    }

    pred = {task: values.argmax(axis=1) for task, values in prob.items()}

    return {
        "y_pred": pred,
        "prob": prob,
        "fold_probs": fold_probs,
    }


def smooth_1d(x, window=15):
    x = np.asarray(x, dtype=np.float32)

    if window is None or window <= 1:
        return x

    window = int(window)
    window += int(window % 2 == 0)

    return np.convolve(x, np.ones(window, dtype=np.float32) / window, mode="same")


def task_display_name(task):
    return {
        "alkane": "Alkane",
        "alkene": "Alkene",
        "alkyne": "Alkyne",
        "aromatics": "Aromatic ring",
        "esters": "Ester",
        "ketones": "Ketone",
        "ortho": "Ortho",
        "meta": "Meta",
        "para": "Para",
        "alkyl_halides": "Alkyl halide",
        "alkyl_halides_4class": "Alkyl halide",
        "alcohols": "Alcohol",
        "alcohols_4class": "Alcohol",
        "ether": "Ether",
        "ether_4class": "Ether",
        "amines": "Amine",
        "amines_4class": "Amine",
        "carbonyl_oxygen": "Carbonyl oxygen",
        "carbonyl_oxygen_4class": "Carbonyl oxygen",
        "aldehydes": "Aldehyde",
        "aldehydes_3class": "Aldehyde",
        "acyl_halides": "Acyl halide",
        "acyl_halides_3class": "Acyl halide",
        "amides": "Amide",
        "amides_3class": "Amide",
        "nitriles": "Nitrile",
        "nitriles_3class": "Nitrile",
        "nitro": "Nitro",
        "nitro_3class": "Nitro",
        "isocyanate": "Isocyanate",
        "isocyanate_3class": "Isocyanate",
        "isothiocyanate": "Isothiocyanate",
        "isothiocyanate_3class": "Isothiocyanate",
    }.get(task, task)


def class_meaning(task, cls):
    if cls == 0:
        return "absent"

    if task.endswith("_3class"):
        return {1: "one", 2: "two or more"}.get(cls, f"class {cls}")

    if task.endswith("_4class"):
        return {1: "one", 2: "two", 3: "three or more"}.get(cls, f"class {cls}")

    return "present" if cls == 1 else f"class {cls}"


def find_correct_class_indices(y_true, pred_result, task, cls):
    true = np.asarray(y_true[task])
    pred = np.asarray(pred_result["y_pred"][task])
    return np.where((true == cls) & (pred == cls))[0]


def select_correct_sample(y_true, pred_result, task, cls, rank=0):
    idx = find_correct_class_indices(y_true, pred_result, task, cls)

    if len(idx) == 0:
        return None

    confidence = pred_result["prob"][task][idx, cls]
    order = np.argsort(-confidence)
    rank = min(int(rank), len(order) - 1)

    return int(idx[order[rank]])


@torch.inference_mode()
def get_kan_contribution_sample(
    models,
    preprocessors,
    x,
    y,
    pred_result,
    task,
    sample_idx,
    device,
):
    raw = np.asarray(x[sample_idx], dtype=np.float32)
    ens_prob = pred_result["prob"][task][sample_idx]
    pred = int(np.argmax(ens_prob))
    true = int(y[task][sample_idx]) if y is not None else -1

    spectra = []
    contribs = []
    logits = []

    for model, preprocessor in zip(models, preprocessors, strict=True):
        spectrum = preprocessor(raw).astype(np.float32)

        x_tensor = torch.as_tensor(
            spectrum,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        model = model.to(device)
        model.eval()

        outputs, contrib = model(
            x_tensor,
            return_contrib=True,
            logit_mode="full",
        )

        spectra.append(spectrum)
        contribs.append(contrib[task][0].detach().cpu().numpy())
        logits.append(outputs[task][0].detach().cpu().numpy())

        model.to("cpu")
        clear_cuda()

    return {
        "task": task,
        "sample_idx": int(sample_idx),
        "spectrum": np.mean(spectra, axis=0),
        "contrib": np.mean(contribs, axis=0),
        "logits": np.mean(logits, axis=0),
        "true": true,
        "pred": pred,
        "prob": ens_prob,
    }


def get_transition_evidence(
    sample,
    target_class,
    reference_class,
    smooth_window=15,
):
    evidence = sample["contrib"][target_class] - sample["contrib"][reference_class]
    return smooth_1d(np.clip(evidence, 0, None), window=smooth_window)


def plot_transition_evidence(
    sample,
    target_class,
    reference_class,
    smooth_window=15,
    save_path=None,
    save_pdf=True,
):
    spectrum = sample["spectrum"]

    if len(COMMON_X) != len(spectrum):
        raise ValueError(
            f"COMMON_X length={len(COMMON_X)} != spectrum length={len(spectrum)}"
        )

    task = sample["task"]
    n_classes = sample["contrib"].shape[0]

    if target_class >= n_classes or reference_class >= n_classes:
        raise ValueError(f"{task} has {n_classes} classes")

    evidence = get_transition_evidence(
        sample=sample,
        target_class=target_class,
        reference_class=reference_class,
        smooth_window=smooth_window,
    )

    vmax = np.percentile(evidence, 99)
    vmax = vmax if vmax > 1e-12 else float(evidence.max()) + 1e-12

    y_min = float(np.min(spectrum))
    y_max = float(np.max(spectrum))
    y_pad = 0.08 * (y_max - y_min + 1e-12)
    y_min -= y_pad
    y_max += y_pad

    fig, ax = plt.subplots(1, 1, figsize=(14.5, 5.2), constrained_layout=True)

    im = ax.imshow(
        np.clip(evidence / vmax, 0, 1)[np.newaxis, :],
        aspect="auto",
        cmap="YlOrRd",
        extent=[float(COMMON_X[0]), float(COMMON_X[-1]), y_min, y_max],
        alpha=0.62,
        origin="lower",
        vmin=0,
        vmax=1,
    )

    ax.plot(COMMON_X, spectrum, color="black", linewidth=1.6, zorder=3, label="Spectrum")
    ax.set_ylim(y_min, y_max)
    ax.invert_xaxis()
    ax.set_xlabel("Wavenumber / cm$^{-1}$", labelpad=8)
    ax.set_ylabel("Normalized intensity", labelpad=8)
    ax.set_title(
        f"{task_display_name(task)} | sample {sample['sample_idx']} | "
        f"true C{sample['true']} | pred C{sample['pred']} | "
        f"C{reference_class} → C{target_class}",
        pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.026, pad=0.012)
    cbar.set_label("Contribution evidence", labelpad=10)

    ax.legend(loc="upper right", frameon=False)
    ax.grid(False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

        if save_pdf:
            fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")

    return fig
