#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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
    Poly1Loss,
    evaluate_saved_ensemble_on_eval_sets,
    predict,
    save_evaluation_tables,
    summarize_metrics,
    task_metrics,
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
) -> dict[str, Poly1Loss]:
    weights = make_class_weights(y, device)

    return {
        task: Poly1Loss(
            n_classes=TASKS.num_classes[task],
            epsilon=1.0,
            weight=weights[task],
        )
        for task in TASKS.all
    }


def total_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    losses: dict[str, Poly1Loss],
) -> torch.Tensor:
    return torch.stack(
        [losses[task](outputs[task], targets[task]) for task in TASKS.all]
    ).sum()


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


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: optim.Optimizer,
    losses: dict[str, Poly1Loss],
    device: torch.device,
) -> float:
    model.train()

    total = 0.0
    steps = 0
    skipped = 0

    for x, y in loader:
        if x.shape[0] < 2:
            skipped += 1
            continue

        x = x.to(device, non_blocking=True)
        y = {task: target.to(device, non_blocking=True) for task, target in y.items()}

        optimizer.zero_grad(set_to_none=True)

        outputs = model(x)
        loss = total_loss(outputs, y, losses) + model_regularization_loss(model, device)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss.item()}")

        loss.backward()
        optimizer.step()

        total += float(loss.detach().cpu())
        steps += 1

    if skipped:
        print(f"Skipped singleton batches: {skipped}")

    if steps == 0:
        raise RuntimeError("No training batches were processed.")

    return total / steps


def evaluate_model(
    model: torch.nn.Module,
    x: np.ndarray,
    y: dict[str, np.ndarray],
    preprocessor: Preprocessor,
    cfg: Config,
    device: torch.device,
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

    y_true, y_pred, prob = predict(model, loader, TASKS, device)
    metrics = task_metrics(y_true, y_pred, TASKS)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "prob": prob,
        "metrics": metrics,
        "summary": summarize_metrics(metrics),
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
    losses = make_losses(train_y, device)

    best_score = -np.inf
    best_state = None
    wait = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, losses, device)

        valid = evaluate_model(
            model=model,
            x=x[valid_idx],
            y=valid_y,
            preprocessor=preprocessor,
            cfg=cfg,
            device=device,
        )

        score = valid["summary"]["overall_acc"]

        print(
            f"fold={fold + 1} epoch={epoch:03d} "
            f"loss={train_loss:.4f} "
            f"val_acc={valid['summary']['overall_acc']:.2f} "
            f"val_f1={valid['summary']['overall_f1']:.2f}"
        )

        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in unwrap_model(model).state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        if wait >= cfg.patience:
            print(f"early stop @ epoch {epoch}")
            break

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
    print(f"Fold {fold + 1} best val_acc={best_score:.2f}")

    return {
        "fold": int(fold + 1),
        "path": fold_path.name,
        "best_score": float(best_score),
    }


def train_cv_and_save(
    name: str,
    x: np.ndarray,
    y: dict[str, np.ndarray],
    model_cfg: ModelConfig,
    cfg: Config,
    run_dir: Path,
    device: torch.device,
) -> dict:
    cv = KFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    fold_records = []

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

    manifest = train_cv_and_save(
        name=args.name,
        x=x_train,
        y=y_train,
        model_cfg=model_cfg,
        cfg=cfg,
        run_dir=paths.out,
        device=device,
    )

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
