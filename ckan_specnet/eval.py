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
    balanced_accuracy_score,
    matthews_corrcoef,
)

from ckan_specnet.core import Config, ModelConfig, TaskCatalog, clear_cuda
from ckan_specnet.data import Preprocessor, load_eval_sets_from_test_parquet, make_loader
from ckan_specnet.model import build_model


def classification_loss_terms(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1.0,
    weight: torch.Tensor | None = None,
    extra_ce_weight: float = 0.0,
    extra_ce_use_class_weight: bool = False,
):
    """Per-sample decomposition of Poly1 / cross-entropy (Poly1Loss and PlainCELoss share it).

    Per-sample loss (CE_i = -log p_{i,y_i}, p_i = softmax(z_i)):

        l_i = w_{y_i} * CE_i  +  eps * (1 - p_{i,y_i})  +  lam * w'_{y_i} * CE_i
              \\___________/     \\_________________/      \\________________/
                 main term            Poly1 term          extra plain cross-entropy
                 (weighted CE)

    where w is the class weight of the main term (1 when weight=None), lam =
    extra_ce_weight, and w' = w (extra_ce_use_class_weight=True) or 1 (default,
    i.e. "plain" cross-entropy).

    Gradient with respect to the logits (every gradient statistic in
    ckan_specnet/grad_track.py follows exactly this expression):

        dl_i / dz_i = [ w_{y_i} + lam * w'_{y_i} + eps * p_{i,y_i} ] * (p_i - onehot(y_i))
                    = (scale_ce_i + scale_poly_i) * (p_i - onehot(y_i))

    The CE term and the Poly1 term are therefore collinear in logits space: Poly1
    is equivalent to scaling the CE gradient per sample by
    (scale_ce + scale_poly) / scale_ce (the amplification is limited for
    low-confidence minority samples and approaches 1 + eps / w_c for samples the
    model already predicts confidently).

    Returns
    -------
    (loss, stats): loss is a scalar (mean of the per-sample losses, i.e. the
    optimisation objective); stats is a dict of per-sample tensors used for loss /
    gradient logging, all already detached.
    """
    nll = F.cross_entropy(logits, target, reduction="none")
    prob = F.softmax(logits, dim=1)
    pt = prob.gather(1, target[:, None]).squeeze(1)

    if weight is None:
        w = torch.ones_like(nll)
    else:
        w = weight.to(device=logits.device, dtype=logits.dtype).gather(0, target)

    ce_term = w * nll
    poly_term = float(epsilon) * (1.0 - pt)

    if extra_ce_weight:
        extra_scale = w if extra_ce_use_class_weight else torch.ones_like(nll)
        extra_term = float(extra_ce_weight) * extra_scale * nll
    else:
        extra_scale = torch.zeros_like(nll)
        extra_term = torch.zeros_like(nll)

    per_sample = ce_term + poly_term + extra_term
    loss = per_sample.mean()

    scale_ce = (w + float(extra_ce_weight) * extra_scale).detach()
    scale_poly = (float(epsilon) * pt).detach()

    stats = {
        "loss": loss.detach(),
        "per_sample": per_sample.detach(),
        "ce": ce_term.detach(),
        "poly": poly_term.detach(),
        "extra_ce": extra_term.detach(),
        "nll": nll.detach(),
        "pt": pt.detach(),
        "prob": prob.detach(),
        "class_weight": w.detach(),
        "scale_ce": scale_ce,
        "scale_poly": scale_poly,
        # dl_i/dz_i = scale_i * (p_i - onehot(y_i))
        "scale": scale_ce + scale_poly,
    }

    return loss, stats


class Poly1Loss(nn.Module):
    """Poly1 loss + an optional plain cross-entropy term.

    l = mean_i [ w_{y_i} * CE_i + eps * (1 - p_{i,y_i}) + ce_weight * CE_i ]

    - ce_weight=0 (default) reproduces the original implementation exactly;
    - ce_weight>0 adds one more plain cross-entropy term inside Poly1 (no class
      weights by default; set ce_use_class_weight=True for the weighted version).

    For the "plain cross-entropy" control experiment you can either swap
    Poly1Loss for PlainCELoss in train.py (the constructor signatures are
    compatible) or use --loss ce / --loss poly1_ce.
    """

    def __init__(
        self,
        n_classes: int,
        epsilon: float = 1.0,
        weight=None,
        ce_weight: float = 0.0,
        ce_use_class_weight: bool = False,
        use_class_weight: bool = True,
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.epsilon = float(epsilon)
        self.ce_weight = float(ce_weight)
        self.ce_use_class_weight = bool(ce_use_class_weight)
        self.use_class_weight = bool(use_class_weight)
        # note: weight is a plain attribute (kept identical to the original implementation),
        # nn.Module does not move it across devices automatically
        self.weight = weight if self.use_class_weight else None
        self.loss_name = "poly1" if self.ce_weight == 0.0 else "poly1+ce"

    def forward(self, logits, target, return_stats: bool = False):
        loss, stats = classification_loss_terms(
            logits,
            target,
            epsilon=self.epsilon,
            weight=self.weight,
            extra_ce_weight=self.ce_weight,
            extra_ce_use_class_weight=self.ce_use_class_weight,
        )
        return (loss, stats) if return_stats else loss


class PlainCELoss(nn.Module):
    """Plain cross-entropy (optional class weights); constructor signature compatible
    with Poly1Loss so it can be swapped in directly for the control experiment.

    Use it as "Poly1Loss(epsilon=0)"; epsilon can still be passed explicitly to
    reproduce Poly1.
    """

    def __init__(
        self,
        n_classes: int,
        epsilon: float = 0.0,
        weight=None,
        ce_weight: float = 0.0,
        ce_use_class_weight: bool = False,
        use_class_weight: bool = True,
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.epsilon = float(epsilon)
        self.ce_weight = float(ce_weight)
        self.ce_use_class_weight = bool(ce_use_class_weight)
        self.use_class_weight = bool(use_class_weight)
        self.weight = weight if self.use_class_weight else None
        self.loss_name = "ce" if self.epsilon == 0.0 else "ce+poly1"

    def forward(self, logits, target, return_stats: bool = False):
        loss, stats = classification_loss_terms(
            logits,
            target,
            epsilon=self.epsilon,
            weight=self.weight,
            extra_ce_weight=self.ce_weight,
            extra_ce_use_class_weight=self.ce_use_class_weight,
        )
        return (loss, stats) if return_stats else loss


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
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Returns two DataFrames:
      - per-task summary table (one row per task): accuracy, precision, recall,
        f1, qwk, balanced_accuracy, matthews_corrcoef and other macro metrics
      - per-class detail table (one row per class): precision, recall, f1, support
    """
    task_summary_rows = []
    per_class_rows = []

    for task in tasks.all:
        yt = np.asarray(y_true[task])
        yp = np.asarray(y_pred[task])
        n_classes = tasks.num_classes[task]
        all_labels = list(range(n_classes))
        observed = np.unique(yt)

        # ---------- per-class metrics ----------
        per_cls_prec, per_cls_rec, per_cls_f1, support = precision_recall_fscore_support(
            yt, yp,
            labels=all_labels,
            average=None,          # no averaging, return per-class arrays
            zero_division=0,
        )

        # ---------- macro metrics ----------
        macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
            yt, yp,
            labels=observed,
            average="macro",
            zero_division=0,
        )

        qwk = (
            cohen_kappa_score(yt, yp, labels=all_labels, weights="quadratic")
            if n_classes > 2 and observed.size > 1
            else np.nan
        )
        acc = accuracy_score(yt, yp) * 100
        bacc = balanced_accuracy_score(yt, yp) * 100
        mcc = matthews_corrcoef(yt, yp)          # valid for multiclass and binary alike

        # ---------- per-task summary row ----------
        task_summary_rows.append({
            "task": task,
            "n_classes": n_classes,
            "observed_labels": observed.tolist(),
            "accuracy": acc,
            "precision": macro_prec * 100,
            "recall": macro_rec * 100,
            "f1": macro_f1 * 100,
            "qwk": qwk,
            "balanced_accuracy": bacc,
            "matthews_corrcoef": mcc,
        })

        # ---------- per-class detail rows ----------
        for cls_id, prec, rec, f1, cnt in zip(all_labels, per_cls_prec, per_cls_rec, per_cls_f1, support):
            per_class_rows.append({
                "task": task,
                "class_id": cls_id,
                "support_samples": int(cnt),
                "precision": prec * 100,
                "recall": rec * 100,
                "f1": f1 * 100,
            })

    df_summary = pl.DataFrame(task_summary_rows)
    df_per_class = pl.DataFrame(per_class_rows)
    return df_summary, df_per_class


def summarize_metrics(metrics: pl.DataFrame) -> dict:
    """
    Group the per-task summary table (binary / multiclass / overall) and compute
    the mean of every metric.
    """
    binary = metrics.filter(pl.col("n_classes") == 2)
    multi = metrics.filter(pl.col("n_classes") > 2)

    def mean(df: pl.DataFrame, col: str) -> float:
        return finite_mean(df[col].to_numpy()) if df.height else np.nan

    return {
        "binary_tasks": binary.height,
        "multiclass_tasks": multi.height,
        "overall_tasks": metrics.height,
        # binary
        "binary_acc": mean(binary, "accuracy"),
        "binary_precision": mean(binary, "precision"),
        "binary_recall": mean(binary, "recall"),
        "binary_f1": mean(binary, "f1"),
        "binary_balanced_accuracy": mean(binary, "balanced_accuracy"),
        "binary_matthews_corrcoef": mean(binary, "matthews_corrcoef"),
        # multiclass
        "multiclass_acc": mean(multi, "accuracy"),
        "multiclass_precision": mean(multi, "precision"),
        "multiclass_recall": mean(multi, "recall"),
        "multiclass_f1": mean(multi, "f1"),
        "multiclass_balanced_accuracy": mean(multi, "balanced_accuracy"),
        "multiclass_matthews_corrcoef": mean(multi, "matthews_corrcoef"),
        "multiclass_qwk": mean(multi, "qwk"),
        # overall
        "overall_acc": mean(metrics, "accuracy"),
        "overall_precision": mean(metrics, "precision"),
        "overall_recall": mean(metrics, "recall"),
        "overall_f1": mean(metrics, "f1"),
        "overall_balanced_accuracy": mean(metrics, "balanced_accuracy"),
        "overall_matthews_corrcoef": mean(metrics, "matthews_corrcoef"),
    }


def summary_std(fold_summaries: list[dict]) -> dict:
    """Standard deviation of every summary metric across folds (static fields such as task counts are skipped)"""
    if not fold_summaries:
        return {}

    skip = {"binary_tasks", "multiclass_tasks", "overall_tasks"}
    keys = [key for key in fold_summaries[0] if key not in skip]

    return {
        f"{key}_std": finite_std([s.get(key, np.nan) for s in fold_summaries])
        for key in keys
    }


def task_metrics_with_std(
    ensemble_metrics: pl.DataFrame,
    fold_metrics: list[pl.DataFrame],
) -> pl.DataFrame:
    """
    Add the across-fold standard deviation of every metric in the per-task summary table.
    """
    rows = []
    for task in ensemble_metrics["task"].to_list():
        fold_rows = [
            df.filter(pl.col("task") == task).to_dicts()[0]
            for df in fold_metrics
        ]
        rows.append({
            "task": task,
            **{
                f"{col}_std": finite_std([row.get(col, np.nan) for row in fold_rows])
                for col in ("accuracy", "precision", "recall", "f1", "qwk",
                            "balanced_accuracy", "matthews_corrcoef")
            },
        })
    return ensemble_metrics.join(pl.DataFrame(rows), on="task", how="left")


def per_class_metrics_with_std(
    ensemble_per_class: pl.DataFrame,
    fold_per_class_list: list[pl.DataFrame],
) -> pl.DataFrame:
    """
    Add the across-fold standard deviation of precision/recall/f1 for every class.
    """
    all_folds = pl.concat(fold_per_class_list)
    std_df = all_folds.group_by(["task", "class_id"]).agg(
        pl.col("precision").std(ddof=0).alias("precision_std"),
        pl.col("recall").std(ddof=0).alias("recall_std"),
        pl.col("f1").std(ddof=0).alias("f1_std"),
    )
    return ensemble_per_class.join(std_df, on=["task", "class_id"], how="left")


def csv_safe_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert the observed_labels column into JSON strings so that the CSV export is safe.
    """
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
def predict_and_loss(
    model: nn.Module,
    loader,
    tasks: TaskCatalog,
    device: torch.device,
    losses: dict[str, nn.Module] | None = None,
):
    """One forward pass that returns the predictions and (optionally) the validation loss.

    Note: as in predict(), this uses the logit_mode="base" logits (the branch the
    validation metrics and early stopping rely on), so val_loss is a loss on the
    base logits as well.

    Returns (y_true, y_pred, y_prob, loss_info); without labels y_true and
    loss_info are None.
    loss_info = {"val_loss": float, "task_val_loss": {task: float}}
    """
    model.eval()

    y_true = {task: [] for task in tasks.all}
    y_pred = {task: [] for task in tasks.all}
    y_prob = {task: [] for task in tasks.all}
    has_labels = False

    task_loss_sum = {task: None for task in tasks.all}
    total_loss_sum = None
    loss_steps = 0

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x, y = batch
            has_labels = True
        else:
            x, y = batch, None

        x = x.to(device, non_blocking=True)
        outputs = model(x, logit_mode="base")   # in case the model expects logit_mode

        for task in tasks.all:
            prob = F.softmax(outputs[task], dim=1).detach().cpu().numpy()
            y_prob[task].append(prob)
            y_pred[task].append(prob.argmax(axis=1))
            if has_labels:
                y_true[task].append(y[task].numpy())

        if losses is not None and has_labels:
            step_total = None
            for task in tasks.all:
                target = y[task].to(device, non_blocking=True)
                task_loss = losses[task](outputs[task], target).detach()
                task_loss_sum[task] = (
                    task_loss if task_loss_sum[task] is None else task_loss_sum[task] + task_loss
                )
                step_total = task_loss if step_total is None else step_total + task_loss
            total_loss_sum = step_total if total_loss_sum is None else total_loss_sum + step_total
            loss_steps += 1

    y_pred = {task: np.concatenate(values) for task, values in y_pred.items()}
    y_prob = {task: np.vstack(values) for task, values in y_prob.items()}

    loss_info = None
    if losses is not None and has_labels and loss_steps > 0:
        loss_info = {
            "val_loss": float(total_loss_sum) / loss_steps,
            "task_val_loss": {
                task: float(value) / loss_steps
                for task, value in task_loss_sum.items()
                if value is not None
            },
        }

    if not has_labels:
        return None, y_pred, y_prob, None
    y_true = {task: np.concatenate(values) for task, values in y_true.items()}
    return y_true, y_pred, y_prob, loss_info


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader,
    tasks: TaskCatalog,
    device: torch.device,
):
    y_true, y_pred, y_prob, _ = predict_and_loss(model, loader, tasks, device, losses=None)
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

        # ensemble: per-task summary + per-class detail
        metrics, per_class_metrics = task_metrics(y_eval, pred_result["y_pred"], tasks)

        # per-fold metrics
        fold_metrics_summary = []
        fold_metrics_per_class = []
        for fold_pred in pred_result["fold_preds"]:
            fold_sum, fold_cls = task_metrics(y_eval, fold_pred, tasks)
            fold_metrics_summary.append(fold_sum)
            fold_metrics_per_class.append(fold_cls)

        # per-fold summaries and standard deviations
        fold_summaries = [summarize_metrics(df) for df in fold_metrics_summary]
        summary = {**summarize_metrics(metrics), **summary_std(fold_summaries)}

        # per-class metrics + across-fold standard deviation
        per_class_with_std = per_class_metrics_with_std(per_class_metrics, fold_metrics_per_class)

        results[eval_name] = {
            "y_true": y_eval,
            "y_pred": pred_result["y_pred"],
            "prob": pred_result["prob"],
            "fold_preds": pred_result["fold_preds"],
            # task level
            "metrics": metrics,
            "metrics_with_std": task_metrics_with_std(metrics, fold_metrics_summary),
            "fold_metrics": fold_metrics_summary,
            "fold_summaries": fold_summaries,
            # class level
            "per_class_metrics": per_class_metrics,
            "per_class_metrics_with_std": per_class_with_std,
            "fold_per_class": fold_metrics_per_class,
            # summary
            "summary": summary,
        }

    return results


def save_evaluation_tables(eval_results: dict, out_dir: Path) -> pl.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for eval_name, item in eval_results.items():
        n = len(next(iter(item["y_true"].values())))

        summary_rows.append({
            "eval_name": eval_name,
            "n_samples": n,
            **item["summary"],
        })

        # per-task table
        csv_safe_df(item["metrics"]).write_csv(out_dir / f"{eval_name}_task_metrics.csv")
        csv_safe_df(item["metrics_with_std"]).write_csv(
            out_dir / f"{eval_name}_task_metrics_with_std.csv"
        )
        # per-fold summary table
        pl.DataFrame(item["fold_summaries"]).with_columns(
            pl.Series("fold", list(range(1, len(item["fold_summaries"]) + 1)))
        ).select(["fold", *item["fold_summaries"][0].keys()]).write_csv(
            out_dir / f"{eval_name}_fold_summaries.csv"
        )

        # per-class tables
        csv_safe_df(item["per_class_metrics"]).write_csv(
            out_dir / f"{eval_name}_per_class_metrics.csv"
        )
        csv_safe_df(item["per_class_metrics_with_std"]).write_csv(
            out_dir / f"{eval_name}_per_class_metrics_with_std.csv"
        )

    summary_df = pl.DataFrame(summary_rows)
    summary_df.write_csv(out_dir / "summary.csv")
    return summary_df