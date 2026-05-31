from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)

from ckan_specnet.core import Config, ModelConfig, TaskCatalog, clear_cuda
from ckan_specnet.data import Preprocessor, load_eval_sets_from_test_parquet, make_loader
from ckan_specnet.model import build_model


class Poly1Loss(nn.Module):
    def __init__(self, n_classes: int, epsilon: float = 1.0, weight=None):
        super().__init__()
        self.n_classes = n_classes
        self.epsilon = epsilon
        self.weight = weight

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = F.softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        return (ce + self.epsilon * (1 - pt)).mean()


def finite_mean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def finite_std(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.std()) if values.size else np.nan


def task_metrics(
    y_true: dict[str, np.ndarray],
    y_pred: dict[str, np.ndarray],
    tasks: TaskCatalog,
) -> pl.DataFrame:
    rows = []

    for task in tasks.all:
        yt = np.asarray(y_true[task])
        yp = np.asarray(y_pred[task])
        n_classes = tasks.num_classes[task]
        labels = list(range(n_classes))
        observed = np.unique(yt)

        precision, recall, f1, _ = precision_recall_fscore_support(
            yt,
            yp,
            labels=observed,
            average="macro",
            zero_division=0,
        )

        qwk = (
            cohen_kappa_score(yt, yp, labels=labels, weights="quadratic")
            if n_classes > 2 and observed.size > 1
            else np.nan
        )

        rows.append(
            {
                "task": task,
                "n_classes": n_classes,
                "observed_labels": observed.tolist(),
                "accuracy": accuracy_score(yt, yp) * 100,
                "precision": precision * 100,
                "recall": recall * 100,
                "f1": f1 * 100,
                "qwk": qwk,
            }
        )

    return pl.DataFrame(rows)


def summarize_metrics(metrics: pl.DataFrame) -> dict:
    binary = metrics.filter(pl.col("n_classes") == 2)
    multi = metrics.filter(pl.col("n_classes") > 2)

    def mean(df: pl.DataFrame, col: str) -> float:
        return finite_mean(df[col].to_numpy()) if df.height else np.nan

    return {
        "binary_tasks": binary.height,
        "multiclass_tasks": multi.height,
        "overall_tasks": metrics.height,
        "binary_acc": mean(binary, "accuracy"),
        "binary_precision": mean(binary, "precision"),
        "binary_recall": mean(binary, "recall"),
        "binary_f1": mean(binary, "f1"),
        "multiclass_acc": mean(multi, "accuracy"),
        "multiclass_precision": mean(multi, "precision"),
        "multiclass_recall": mean(multi, "recall"),
        "multiclass_f1": mean(multi, "f1"),
        "multiclass_qwk": mean(multi, "qwk"),
        "overall_acc": mean(metrics, "accuracy"),
        "overall_precision": mean(metrics, "precision"),
        "overall_recall": mean(metrics, "recall"),
        "overall_f1": mean(metrics, "f1"),
    }


def summary_std(fold_summaries: list[dict]) -> dict:
    if not fold_summaries:
        return {}

    skip = {"binary_tasks", "multiclass_tasks", "overall_tasks"}
    keys = [key for key in fold_summaries[0] if key not in skip]

    return {
        f"{key}_std": finite_std(
            [summary.get(key, np.nan) for summary in fold_summaries]
        )
        for key in keys
    }


def task_metrics_with_std(
    ensemble_metrics: pl.DataFrame,
    fold_metrics: list[pl.DataFrame],
) -> pl.DataFrame:
    rows = []

    for task in ensemble_metrics["task"].to_list():
        fold_rows = [
            df.filter(pl.col("task") == task).to_dicts()[0] for df in fold_metrics
        ]

        rows.append(
            {
                "task": task,
                **{
                    f"{col}_std": finite_std(
                        [row.get(col, np.nan) for row in fold_rows]
                    )
                    for col in ("accuracy", "precision", "recall", "f1", "qwk")
                },
            }
        )

    return ensemble_metrics.join(pl.DataFrame(rows), on="task", how="left")


def csv_safe_df(df: pl.DataFrame) -> pl.DataFrame:
    out = df.clone()

    if "observed_labels" not in out.columns:
        return out

    labels_json = []

    for value in out["observed_labels"].to_list():
        if isinstance(value, pl.Series):
            value = value.to_list()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, tuple):
            value = list(value)
        elif value is None:
            value = []
        else:
            value = list(value)

        labels_json.append(json.dumps(value, ensure_ascii=False))

    out = out.drop("observed_labels").with_columns(
        pl.Series("observed_labels", labels_json)
    )

    cols = [col for col in out.columns if col != "observed_labels"]
    insert_at = 2 if len(cols) >= 2 else len(cols)

    return out.select(cols[:insert_at] + ["observed_labels"] + cols[insert_at:])


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader,
    tasks: TaskCatalog,
    device: torch.device,
):
    model.eval()

    y_true = {task: [] for task in tasks.all}
    y_pred = {task: [] for task in tasks.all}
    y_prob = {task: [] for task in tasks.all}
    has_labels = False

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x, y = batch
            has_labels = True
        else:
            x, y = batch, None

        outputs = model(x.to(device, non_blocking=True))

        for task in tasks.all:
            prob = F.softmax(outputs[task], dim=1).detach().cpu().numpy()
            y_prob[task].append(prob)
            y_pred[task].append(prob.argmax(axis=1))

            if has_labels:
                y_true[task].append(y[task].numpy())

    y_pred = {task: np.concatenate(values) for task, values in y_pred.items()}
    y_prob = {task: np.vstack(values) for task, values in y_prob.items()}

    if not has_labels:
        return None, y_pred, y_prob

    y_true = {task: np.concatenate(values) for task, values in y_true.items()}

    return y_true, y_pred, y_prob


def load_manifest(run_dir: Path) -> dict:
    path = Path(run_dir) / "manifest.json"

    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(path.read_text())


def tasks_from_manifest(manifest: dict) -> TaskCatalog:
    if "task_config" in manifest:
        return TaskCatalog.from_dict(manifest["task_config"])

    return TaskCatalog.build()


def load_fold_model(fold_path: Path, tasks: TaskCatalog) -> tuple[nn.Module, dict]:
    ckpt = torch.load(fold_path, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_config"])

    model = build_model(
        input_size=int(ckpt["input_size"]),
        cfg=model_cfg,
        tasks=tasks,
    )

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to("cpu")

    return model, ckpt


@torch.inference_mode()
def predict_saved_ensemble_streaming(
    run_dir: Path,
    manifest: dict,
    x: np.ndarray,
    tasks: TaskCatalog,
    cfg: Config,
    device: torch.device,
) -> dict:
    fold_files = manifest["ensemble"]["fold_files"]
    normalize = manifest["config"]["normalize"]

    prob_sum = {task: None for task in tasks.all}
    fold_preds = []

    for i, fold_file in enumerate(fold_files, start=1):
        fold_path = Path(run_dir) / fold_file
        print(f"Predicting fold {i}/{len(fold_files)}: {fold_path.name}")

        model, _ = load_fold_model(fold_path, tasks)
        model = model.to(device)

        loader = make_loader(
            x=x,
            y=None,
            preprocessor=Preprocessor(normalize),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            device=device,
        )

        _, _, prob = predict(model, loader, tasks, device)
        fold_pred = {task: prob[task].argmax(axis=1) for task in tasks.all}
        fold_preds.append(fold_pred)

        for task in tasks.all:
            arr = prob[task].astype(np.float64)
            prob_sum[task] = arr if prob_sum[task] is None else prob_sum[task] + arr

        model.to("cpu")
        del model
        clear_cuda()

    ensemble_prob = {task: prob_sum[task] / len(fold_files) for task in tasks.all}
    ensemble_pred = {task: prob.argmax(axis=1) for task, prob in ensemble_prob.items()}

    return {
        "y_pred": ensemble_pred,
        "prob": ensemble_prob,
        "fold_preds": fold_preds,
    }


def evaluate_saved_ensemble_on_eval_sets(
    run_dir: Path,
    manifest: dict,
    test_df: pl.DataFrame,
    tasks: TaskCatalog,
    cfg: Config,
    device: torch.device,
) -> dict:
    eval_sets = load_eval_sets_from_test_parquet(test_df, tasks)
    results = {}

    for eval_name, (x_eval, y_eval) in eval_sets.items():
        print(f"\nEvaluating {eval_name}")

        pred_result = predict_saved_ensemble_streaming(
            run_dir=run_dir,
            manifest=manifest,
            x=x_eval,
            tasks=tasks,
            cfg=cfg,
            device=device,
        )

        metrics = task_metrics(y_eval, pred_result["y_pred"], tasks)
        fold_metrics = [
            task_metrics(y_eval, fold_pred, tasks)
            for fold_pred in pred_result["fold_preds"]
        ]
        fold_summaries = [summarize_metrics(df) for df in fold_metrics]
        summary = {**summarize_metrics(metrics), **summary_std(fold_summaries)}

        results[eval_name] = {
            "y_true": y_eval,
            "y_pred": pred_result["y_pred"],
            "prob": pred_result["prob"],
            "fold_preds": pred_result["fold_preds"],
            "metrics": metrics,
            "metrics_with_std": task_metrics_with_std(metrics, fold_metrics),
            "summary": summary,
            "fold_metrics": fold_metrics,
            "fold_summaries": fold_summaries,
        }

    return results


def save_evaluation_tables(eval_results: dict, out_dir: Path) -> pl.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for eval_name, item in eval_results.items():
        n = len(next(iter(item["y_true"].values())))

        summary_rows.append(
            {
                "eval_name": eval_name,
                "n_samples": n,
                **item["summary"],
            }
        )

        csv_safe_df(item["metrics"]).write_csv(out_dir / f"{eval_name}_task_metrics.csv")
        csv_safe_df(item["metrics_with_std"]).write_csv(
            out_dir / f"{eval_name}_task_metrics_with_std.csv"
        )

        pl.DataFrame(item["fold_summaries"]).with_columns(
            pl.Series("fold", list(range(1, len(item["fold_summaries"]) + 1)))
        ).select(["fold", *item["fold_summaries"][0].keys()]).write_csv(
            out_dir / f"{eval_name}_fold_summaries.csv"
        )

    summary_df = pl.DataFrame(summary_rows)
    summary_df.write_csv(out_dir / "summary.csv")

    return summary_df
