#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

LAUNCH_DIR = Path.cwd()

import numpy as np
import polars as pl
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

from ckan_specnet.core import (
    TASKS,
    Config,
    ModelConfig,
    clear_cuda,
    configure_torch_runtime,
    device_of,
    seed_everything,
    unwrap_model,
)
from ckan_specnet.data import (
    Preprocessor,
    load_train_data_excluding_test,
    make_loader,
    prepare_or_load_test_parquet,
    take,
)
from ckan_specnet.eval import (
    PlainCELoss,
    Poly1Loss,
    evaluate_saved_ensemble_on_eval_sets,
    predict_and_loss,
    save_evaluation_tables,
    summarize_metrics,
    task_metrics,
)
from ckan_specnet.grad_track import (
    DEFAULT_MINORITY_THRESHOLD_PCT,
    GradientTracker,
    RunLogger,
    class_counts_from_targets,
    minority_specs,
    specs_payload,
    write_json,
)
from ckan_specnet.model import build_model
from ckan_specnet.paths import TrainPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train five-fold CKAN-SpecNet models.")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", type=str, default="ContributionKAN_b64_a001_avgmax_norm")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--save-fp16", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")

    # ---------------- loss configuration ----------------
    parser.add_argument(
        "--loss",
        choices=("poly1", "poly1_ce", "ce"),
        default="poly1",
        help="poly1=original Poly1 loss; poly1_ce=Poly1+plain cross-entropy; ce=plain weighted cross-entropy (control experiment)",
    )
    parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon of Poly1")
    parser.add_argument(
        "--ce-weight",
        type=float,
        default=0.1,
        help="weight of the extra plain cross-entropy term used by --loss poly1_ce",
    )
    parser.add_argument(
        "--ce-use-class-weight",
        action="store_true",
        help="also apply class weights to the extra cross-entropy (off by default, i.e. plain cross-entropy)",
    )
    parser.add_argument(
        "--class-weight",
        choices=("balanced", "none"),
        default="balanced",
        help="whether the main term uses balanced class weights (none = no weights at all)",
    )

    # ---------------- gradient / loss logging configuration ----------------
    parser.add_argument(
        "--grad-track",
        choices=("off", "logit", "full"),
        default="full",
        help="minority-class gradient logging granularity: off=disabled; logit=logits-space gradients; "
        "full=also add last-classifier-layer parameter-gradient attribution (default)",
    )
    parser.add_argument(
        "--minority-mode",
        choices=("group", "class"),
        default="group",
        help="minority definition: group=by functional-group prevalence table (<threshold); class=by actual training class frequency",
    )
    parser.add_argument(
        "--minority-threshold",
        type=float,
        default=DEFAULT_MINORITY_THRESHOLD_PCT,
        help="prevalence / class-frequency threshold (%%), default 5",
    )
    parser.add_argument(
        "--no-val-loss",
        action="store_true",
        help="do not compute the validation loss (saves one validation forward pass; logging is on by default)",
    )
    return parser.parse_args()


def make_class_weights(
    y: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out = {}

    for task, values in y.items():
        n_classes = TASKS.num_classes[task]
        weights = np.ones(n_classes, dtype=np.float32)
        present = np.unique(values)

        balanced = compute_class_weight(
            class_weight="balanced",
            classes=present,
            y=values,
        ).astype(np.float32)

        for cls, weight in zip(present, balanced, strict=True):
            weights[cls] = weight

        out[task] = torch.as_tensor(weights, dtype=torch.float32, device=device)

    return out


def make_losses(
    y: dict[str, np.ndarray],
    device: torch.device,
    *,
    loss_name: str = "poly1",
    epsilon: float = 1.0,
    ce_weight: float = 0.0,
    ce_use_class_weight: bool = False,
    class_weight_mode: str = "balanced",
) -> dict[str, torch.nn.Module]:
    """Build the loss of every task.

    - loss_name="poly1"    : Poly1 (default, identical to the original implementation)
    - loss_name="poly1_ce" : Poly1 + plain cross-entropy (weight ce_weight)
    - loss_name="ce"       : plain cross-entropy (control experiment)
    """
    weights = make_class_weights(y, device) if class_weight_mode == "balanced" else None

    losses: dict[str, torch.nn.Module] = {}
    for task in TASKS.all:
        weight = None if weights is None else weights[task]
        n_classes = TASKS.num_classes[task]
        use_class_weight = weights is not None

        if loss_name == "poly1":
            losses[task] = Poly1Loss(
                n_classes=n_classes, epsilon=epsilon, weight=weight,
                use_class_weight=use_class_weight,
            )
        elif loss_name == "poly1_ce":
            losses[task] = Poly1Loss(
                n_classes=n_classes,
                epsilon=epsilon,
                weight=weight,
                ce_weight=ce_weight,
                ce_use_class_weight=ce_use_class_weight,
                use_class_weight=use_class_weight,
            )
        elif loss_name == "ce":
            losses[task] = PlainCELoss(
                n_classes=n_classes, weight=weight, use_class_weight=use_class_weight,
            )
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

    return losses


def compute_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    losses: dict[str, torch.nn.Module],
    tracker: GradientTracker | None = None,
):
    """Return (total loss, {task: loss}, {task: stats}); total loss matches the original implementation."""
    task_losses = []
    per_task: dict[str, torch.Tensor] = {}
    per_task_stats: dict[str, dict] = {}

    for task in TASKS.all:
        loss_t, stats = losses[task](outputs[task], targets[task], return_stats=True)
        per_task[task] = loss_t
        per_task_stats[task] = stats
        task_losses.append(loss_t)

        if tracker is not None:
            tracker.observe_task(task, stats, targets[task])

    return torch.stack(task_losses).sum(), per_task, per_task_stats


def total_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    losses: dict[str, torch.nn.Module],
) -> torch.Tensor:
    loss, _, _ = compute_losses(outputs, targets, losses, tracker=None)
    return loss


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    decay, no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lower = name.lower()

        if (
            param.ndim <= 1
            or "bias" in lower
            or "norm" in lower
            or "bn" in lower
            or "alpha" in lower
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    return torch.compile(model) if enabled else model


def model_regularization_loss(
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    base = unwrap_model(model)

    if hasattr(base, "regularization_loss") and callable(base.regularization_loss):
        reg = base.regularization_loss()
        if isinstance(reg, torch.Tensor):
            return reg

    return torch.zeros((), dtype=torch.float32, device=device)


def global_grad_norm(model: torch.nn.Module) -> torch.Tensor:
    base = unwrap_model(model)
    total = None

    for param in base.parameters():
        if param.grad is None:
            continue
        norm = param.grad.detach().norm()
        total = norm * norm if total is None else total + norm * norm

    if total is None:
        return torch.zeros((), dtype=torch.float32)

    return total.sqrt()


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: optim.Optimizer,
    losses: dict[str, torch.nn.Module],
    device: torch.device,
    tracker: GradientTracker | None = None,
) -> dict:
    """Run one epoch and return loss / gradient norms / per-task loss breakdown (all plain Python scalars)."""
    model.train()

    tasks = list(TASKS.all)
    zero = torch.zeros((), device=device)

    task_loss_sum = {task: zero.clone() for task in tasks}
    task_ce_sum = {task: zero.clone() for task in tasks}
    task_poly_sum = {task: zero.clone() for task in tasks}
    task_extra_sum = {task: zero.clone() for task in tasks}

    total = zero.clone()
    reg_total = zero.clone()
    grad_norm_total = zero.clone()

    steps = 0
    samples = 0
    skipped = 0

    for x, y in loader:
        if x.shape[0] < 2:
            skipped += 1
            continue

        x = x.to(device, non_blocking=True)
        y = {task: target.to(device, non_blocking=True) for task, target in y.items()}

        optimizer.zero_grad(set_to_none=True)

        if tracker is not None:
            tracker.begin_step()

        outputs = model(x)
        loss, per_task, per_task_stats = compute_losses(outputs, y, losses, tracker=tracker)
        reg_loss = model_regularization_loss(model, device)
        loss = loss + reg_loss

        if not torch.isfinite(loss):
            if tracker is not None:
                tracker.end_step()
            raise RuntimeError(f"Non-finite training loss: {loss.item()}")

        loss.backward()
        grad_norm = global_grad_norm(model)
        optimizer.step()

        if tracker is not None:
            tracker.end_step()

        for task in tasks:
            stats = per_task_stats[task]
            task_loss_sum[task] = task_loss_sum[task] + per_task[task].detach()
            task_ce_sum[task] = task_ce_sum[task] + stats["ce"].mean().detach()
            task_poly_sum[task] = task_poly_sum[task] + stats["poly"].mean().detach()
            task_extra_sum[task] = task_extra_sum[task] + stats["extra_ce"].mean().detach()

        total = total + loss.detach()
        reg_total = reg_total + reg_loss.detach()
        grad_norm_total = grad_norm_total + grad_norm.detach()

        steps += 1
        samples += int(x.shape[0])

    if skipped:
        print(f"Skipped singleton batches: {skipped}")

    if steps == 0:
        raise RuntimeError("No training batches were processed.")

    per_task_out = {
        task: {
            "loss": float(task_loss_sum[task]) / steps,
            "ce": float(task_ce_sum[task]) / steps,
            "poly": float(task_poly_sum[task]) / steps,
            "extra_ce": float(task_extra_sum[task]) / steps,
        }
        for task in tasks
    }

    return {
        "loss": float(total) / steps,
        "reg_loss": float(reg_total) / steps,
        "grad_norm": float(grad_norm_total) / steps,
        "ce": sum(item["ce"] for item in per_task_out.values()),
        "poly": sum(item["poly"] for item in per_task_out.values()),
        "extra_ce": sum(item["extra_ce"] for item in per_task_out.values()),
        "steps": steps,
        "samples": samples,
        "task": per_task_out,
    }


def evaluate_model(
    model: torch.nn.Module,
    x: np.ndarray,
    y: dict[str, np.ndarray],
    preprocessor: Preprocessor,
    cfg: Config,
    device: torch.device,
    losses: dict[str, torch.nn.Module] | None = None,
) -> dict:
    loader = make_loader(
        x=x,
        y=y,
        preprocessor=preprocessor,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        device=device,
    )

    y_true, y_pred, prob, loss_info = predict_and_loss(model, loader, TASKS, device, losses)
    # note: task_metrics returns (per-task summary table, per-class detail table)
    metrics, per_class_metrics = task_metrics(y_true, y_pred, TASKS)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "prob": prob,
        "metrics": metrics,
        "per_class_metrics": per_class_metrics,
        "summary": summarize_metrics(metrics),
        "loss": loss_info,
    }


def tensor_to_save(t: torch.Tensor, fp16: bool = False) -> torch.Tensor:
    t = t.detach().cpu()
    return t.half() if fp16 and t.is_floating_point() else t


def save_fold_checkpoint(
    model: torch.nn.Module,
    fold: int,
    best_score: float,
    run_dir: Path,
    input_size: int,
    model_cfg: ModelConfig,
    cfg: Config,
) -> Path:
    state_dict = {
        key: tensor_to_save(value, fp16=cfg.save_fp16)
        for key, value in unwrap_model(model).state_dict().items()
    }

    ckpt = {
        "fold": int(fold + 1),
        "best_score": float(best_score),
        "input_size": int(input_size),
        "model_config": asdict(model_cfg),
        "task_config": TASKS.to_dict(),
        "normalize": cfg.normalize,
        "save_fp16": bool(cfg.save_fp16),
        "state_dict": state_dict,
    }

    path = run_dir / f"fold_{fold + 1}.pt"
    torch.save(ckpt, path)

    return path


def fit_fold_and_save(
    fold: int,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    x: np.ndarray,
    y: dict[str, np.ndarray],
    model_cfg: ModelConfig,
    cfg: Config,
    run_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
    specs,
    logger: RunLogger,
) -> dict:
    print(f"\nFold {fold + 1}/{cfg.folds}: train={len(train_idx):,} valid={len(valid_idx):,}")

    train_y = take(y, train_idx)
    valid_y = take(y, valid_idx)
    preprocessor = Preprocessor(cfg.normalize).fit(x[train_idx])

    train_loader = make_loader(
        x=x[train_idx],
        y=train_y,
        preprocessor=preprocessor,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        device=device,
    )

    model = build_model(x.shape[1], model_cfg, TASKS).to(device)
    model = maybe_compile(model, cfg.compile_model)

    optimizer = build_optimizer(unwrap_model(model), cfg.lr, cfg.weight_decay)
    losses = make_losses(
        train_y,
        device,
        loss_name=args.loss,
        epsilon=args.epsilon,
        ce_weight=args.ce_weight,
        ce_use_class_weight=args.ce_use_class_weight,
        class_weight_mode=args.class_weight,
    )

    # ---------------- minority-class gradient tracker ----------------
    tracker = None
    if args.grad_track != "off":
        with_params = args.grad_track == "full"
        if with_params and cfg.compile_model:
            print(
                "[grad-track] torch.compile is enabled: parameter-level (head/param) metrics "
                "are disabled for compatibility, only logits-space gradients are logged"
            )
            with_params = False

        tracker = GradientTracker(
            TASKS, specs, losses, track_params=with_params, fold=fold + 1
        )
        tracker.attach(model)
        if with_params and not tracker.tasks_with_params:
            print("[grad-track] base_heads[task][-1] not found, parameter-level metrics will be missing")
        else:
            print(
                f"[grad-track] mode={args.grad_track} tracking {len(specs)} minority (task, class) entries "
                f"+ {len(tracker.specs_by_task)} per-task reference rows"
            )

    best_score = -np.inf
    best_state = None
    best_epoch = 0
    wait = 0
    epochs_run = 0

    for epoch in range(1, cfg.epochs + 1):
        started = time.perf_counter()
        if tracker is not None:
            tracker.begin_epoch(epoch)

        train_stats = train_epoch(
            model, train_loader, optimizer, losses, device, tracker=tracker
        )

        valid = evaluate_model(
            model=model,
            x=x[valid_idx],
            y=valid_y,
            preprocessor=preprocessor,
            cfg=cfg,
            device=device,
            losses=None if args.no_val_loss else losses,
        )

        val_loss = valid["loss"]["val_loss"] if valid["loss"] else float("nan")
        score = valid["summary"]["overall_acc"]
        is_best = bool(score > best_score)
        early_stop = False

        print(
            f"fold={fold + 1} epoch={epoch:03d} "
            f"loss={train_stats['loss']:.4f} "
            f"(ce={train_stats['ce']:.4f} poly={train_stats['poly']:.4f}) "
            f"val_loss={val_loss:.4f} "
            f"val_acc={valid['summary']['overall_acc']:.2f} "
            f"val_f1={valid['summary']['overall_f1']:.2f}"
        )

        if is_best:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in unwrap_model(model).state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        if wait >= cfg.patience:
            early_stop = True
            print(f"early stop @ epoch {epoch}")

        epochs_run = epoch
        epoch_seconds = time.perf_counter() - started

        # ---------------- loss logging ----------------
        logger.log(
            "train_log",
            {
                "fold": fold + 1,
                "epoch": epoch,
                "steps": train_stats["steps"],
                "samples": train_stats["samples"],
                "train_loss": train_stats["loss"],
                "train_loss_ce": train_stats["ce"],
                "train_loss_poly": train_stats["poly"],
                "train_loss_extra_ce": train_stats["extra_ce"],
                "train_reg_loss": train_stats["reg_loss"],
                "grad_norm": train_stats["grad_norm"],
                "lr": cfg.lr,
                "val_loss": val_loss,
                "val_acc": valid["summary"]["overall_acc"],
                "val_f1": valid["summary"]["overall_f1"],
                "val_balanced_accuracy": valid["summary"]["overall_balanced_accuracy"],
                "best_score": best_score,
                "is_best": int(is_best),
                "early_stop": int(early_stop),
                "epoch_seconds": epoch_seconds,
            },
        )

        task_val_loss = valid["loss"]["task_val_loss"] if valid["loss"] else {}
        for task in TASKS.all:
            logger.log(
                "train_loss_by_task",
                {
                    "fold": fold + 1,
                    "epoch": epoch,
                    "task": task,
                    "train_loss": train_stats["task"][task]["loss"],
                    "train_loss_ce": train_stats["task"][task]["ce"],
                    "train_loss_poly": train_stats["task"][task]["poly"],
                    "train_loss_extra_ce": train_stats["task"][task]["extra_ce"],
                    "val_loss": task_val_loss.get(task, float("nan")),
                },
            )

        # ---------------- minority-class gradient logging ----------------
        if tracker is not None:
            for row in tracker.epoch_rows():
                logger.log("minority_grad_by_epoch", row)

        if early_stop:
            break

    if tracker is not None:
        tracker.close()

    if best_state is not None:
        unwrap_model(model).load_state_dict(best_state)

    fold_path = save_fold_checkpoint(
        model=model,
        fold=fold,
        best_score=best_score,
        run_dir=run_dir,
        input_size=x.shape[1],
        model_cfg=model_cfg,
        cfg=cfg,
    )

    model.to("cpu")
    del model
    clear_cuda()

    print(f"Saved fold {fold + 1}: {fold_path}")
    print(f"Fold {fold + 1} best val_acc={best_score:.2f} (epoch {best_epoch})")

    return {
        "fold": int(fold + 1),
        "path": fold_path.name,
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "epochs_run": int(epochs_run),
        "final_train_loss": float(train_stats["loss"]),
    }


def train_cv_and_save(
    name: str,
    x: np.ndarray,
    y: dict[str, np.ndarray],
    model_cfg: ModelConfig,
    cfg: Config,
    run_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
    logger: RunLogger,
) -> dict:
    cv = KFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    fold_records = []

    class_counts = class_counts_from_targets(y, TASKS)
    specs = minority_specs(
        TASKS,
        threshold_pct=args.minority_threshold,
        mode=args.minority_mode,
        class_counts=class_counts,
    )

    print(
        f"Minority specs ({args.minority_mode}, <{args.minority_threshold}%): "
        f"{len(specs)} (task, class) entries"
    )
    for spec in specs:
        print(
            f"  - {spec.task:24s} class={spec.class_id} "
            f"group={spec.group:18s} rate={spec.presence_rate_pct:.3f}%"
        )

    write_json(
        run_dir / "minority_grad_specs.json",
        specs_payload(
            specs,
            config={
                "loss": args.loss,
                "epsilon": args.epsilon,
                "ce_weight": args.ce_weight,
                "ce_use_class_weight": bool(args.ce_use_class_weight),
                "class_weight": args.class_weight,
                "grad_track": args.grad_track,
                "minority_mode": args.minority_mode,
                "minority_threshold_pct": args.minority_threshold,
            },
            class_counts=class_counts,
        ),
    )

    for fold, (train_idx, valid_idx) in enumerate(cv.split(x)):
        record = fit_fold_and_save(
            fold=fold,
            train_idx=train_idx,
            valid_idx=valid_idx,
            x=x,
            y=y,
            model_cfg=model_cfg,
            cfg=cfg,
            run_dir=run_dir,
            device=device,
            args=args,
            specs=specs,
            logger=logger,
        )
        fold_records.append(record)

    val_scores = [record["best_score"] for record in fold_records]

    manifest = {
        "name": name,
        "created_at": datetime.now().isoformat(),
        "config": {
            "parquet": str(cfg.parquet),
            "test_parquet": str(cfg.test_parquet),
            "train_sources": list(cfg.train_sources),
            "eval_sources": {key: list(value) for key, value in cfg.eval_sources.items()},
            "require_eval_sources": cfg.require_eval_sources,
            "single_component_only": cfg.single_component_only,
            "seed": cfg.seed,
            "test_size": cfg.test_size,
            "folds": cfg.folds,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "patience": cfg.patience,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "normalize": cfg.normalize,
            "compile_model": cfg.compile_model,
            "save_fp16": cfg.save_fp16,
        },
        "loss": {
            "name": args.loss,
            "epsilon": args.epsilon,
            "ce_weight": args.ce_weight if args.loss == "poly1_ce" else 0.0,
            "ce_use_class_weight": bool(args.ce_use_class_weight),
            "class_weight": args.class_weight,
            "val_loss_logits": "base",
        },
        "grad_tracking": {
            "mode": args.grad_track,
            "minority_mode": args.minority_mode,
            "minority_threshold_pct": args.minority_threshold,
            "specs": [spec.to_dict() for spec in specs],
            "files": {
                "train_log": "train_log.csv",
                "train_loss_by_task": "train_loss_by_task.csv",
                "minority_grad_by_epoch": "minority_grad_by_epoch.csv",
                "specs": "minority_grad_specs.json",
            },
        },
        "model_config": asdict(model_cfg),
        "task_config": TASKS.to_dict(),
        "input_size": int(x.shape[1]),
        "ensemble": {
            "method": "soft_voting_probability_mean",
            "fold_files": [record["path"] for record in fold_records],
        },
        "folds": fold_records,
        "val_scores": val_scores,
        "val_score_mean": float(np.mean(val_scores)),
        "val_score_std": float(np.std(val_scores)),
    }

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    )

    return manifest


def main() -> None:
    args = parse_args()
    paths = TrainPaths.from_args(args, LAUNCH_DIR)

    configure_torch_runtime()

    cfg = replace(
        Config(),
        parquet=paths.parquet,
        test_parquet=paths.test,
        save_dir=paths.out,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        folds=args.folds,
        seed=args.seed,
        num_workers=args.num_workers,
        compile_model=args.compile_model,
        save_fp16=args.save_fp16,
    )

    seed_everything(cfg.seed)
    device = device_of()
    model_cfg = ModelConfig()

    paths.out.mkdir(parents=True, exist_ok=True)
    test_df = prepare_or_load_test_parquet(cfg, TASKS)
    train_data = load_train_data_excluding_test(cfg, TASKS, test_df)
    

    x_train = train_data["x_train"]
    y_train = train_data["y_train"]

    print(f"Train samples: {len(x_train):,}")
    print(f"Features: {x_train.shape[1]}")
    print(f"Run dir: {paths.out.resolve()}")
    print(f"Loss: {args.loss} (epsilon={args.epsilon}, ce_weight={args.ce_weight})")

    log_files = ["train_log", "train_loss_by_task"]
    if args.grad_track != "off":
        log_files.append("minority_grad_by_epoch")
    logger = RunLogger(paths.out, enabled=log_files)
    print(
        "Log files: "
        + ", ".join(str(logger.path(name)) for name in logger.FILES if name in log_files)
    )

    try:
        manifest = train_cv_and_save(
            name=args.name,
            x=x_train,
            y=y_train,
            model_cfg=model_cfg,
            cfg=cfg,
            run_dir=paths.out,
            device=device,
            args=args,
            logger=logger,
        )
    finally:
        logger.close()

    print(
        f"Validation mean={manifest['val_score_mean']:.2f}, "
        f"std={manifest['val_score_std']:.2f}"
    )

    if not args.skip_eval:
        eval_results = evaluate_saved_ensemble_on_eval_sets(
            run_dir=paths.out,
            manifest=manifest,
            test_df=pl.read_parquet(paths.test),
            tasks=TASKS,
            cfg=cfg,
            device=device,
        )
        summary = save_evaluation_tables(eval_results, paths.out)
        print(summary)


if __name__ == "__main__":
    main()
