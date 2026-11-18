#!/usr/bin/env python
"""exp0 - audit the held-out set and export the exact logit decomposition.

This is step 1 of the agreed work plan ("核对数据与模型输出"): before any
perturbation is run, fix the sample space and export, for every held-out
spectrum and every fold,

    base logits (CNN branch)   kan logits (Contribution-KAN branch)
    full logits = base + kan   KAN bias term b(z)   alpha

for the four tasks of interest (nitro, nitro_3class, nitriles,
nitriles_3class), plus the class support of the held-out set.

It also verifies, numerically, the two identities exp1/exp2 rely on:

    full_logit == base_logit + kan_logit
    sum_l contrib_c(l) == kan_logit_c - alpha * b_c(z)

Usage
-----
    python exp0_export_logits.py --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402

TASKS_OF_INTEREST = ("nitro", "nitriles", "nitro_3class", "nitriles_3class")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--folds", type=str, default=None, help="comma separated fold ids (debug)")
    p.add_argument("--out-dir", type=Path, default=C.RESULT_DIR / "exp0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = C.parse_device(args.device)
    folds = [int(f) for f in args.folds.split(",")] if args.folds else None
    out_dir = C.ensure_dir(args.out_dir)

    ens = C.load_ensemble(device, folds=folds)
    data = C.load_analysis_data()
    print(f"[info] held-out rows = {len(data)}, folds = {ens.n_folds}, device = {device}")
    print(f"[info] fold files = {ens.fold_files}")

    X = ens.preprocess(data.raw, fold=0)            # sample_zscore -> fold independent
    n = len(X)

    # ---------------- per-fold logits for the four tasks ------------------- #
    store: dict[str, np.ndarray] = {}
    for task in TASKS_OF_INTEREST:
        base = C.fold_logits(ens, X, task, logit_mode="base")
        kan = C.fold_logits(ens, X, task, logit_mode="delta")
        store[f"{task}__base"] = base
        store[f"{task}__kan"] = kan
        store[f"{task}__full"] = base + kan
        print(f"[logits] {task:18s} {base.shape}")

    for task in TASKS_OF_INTEREST:
        _, bias = C.fold_logits_and_bias(ens, X, task, batch_size=512)
        store[f"{task}__bias"] = bias

    alpha = {task: ens.alpha(task) for task in TASKS_OF_INTEREST}

    # ---------------- identity checks ------------------------------------- #
    identity_rows = []
    for task in TASKS_OF_INTEREST:
        base, kan, full = store[f"{task}__base"], store[f"{task}__kan"], store[f"{task}__full"]
        err = float(np.abs(full - (base + kan)).max())

        # second identity needs contrib: verify on a subsample for speed
        idx = np.linspace(0, n - 1, min(256, n)).astype(int)
        contrib = C.fold_contrib(ens, X[idx], task)
        a = alpha[task].reshape(-1, 1, 1)
        lhs = contrib.sum(axis=-1)                       # (F, m, K)
        rhs = kan[:, idx, :] - a * store[f"{task}__bias"][:, idx, :]
        err2 = float(np.abs(lhs - rhs).max())
        scale = float(np.abs(rhs).max())
        identity_rows.append({
            "task": task,
            "check": "full == base + kan",
            "max_abs_err": err,
            "rel_err": err / max(1e-9, float(np.abs(full).max())),
        })
        identity_rows.append({
            "task": task,
            "check": "sum_l contrib == kan - alpha*b(z)",
            "max_abs_err": err2,
            "rel_err": err2 / max(1e-9, scale),
        })
        print(f"[identity] {task:18s} full==base+kan {err:.2e}   "
              f"sum(contrib)==kan-alpha*b {err2:.2e}")
        del contrib

    pl.DataFrame(identity_rows).write_csv(out_dir / "identity_check.csv")

    # ---------------- all-task accuracy audit ----------------------------- #
    audit_rows = []
    for f, model in enumerate(ens.models):
        with torch.inference_mode():
            out = model(torch.as_tensor(X, dtype=torch.float32, device=ens.device), logit_mode="full")
        for task, logits in out.items():
            pred = logits.argmax(dim=1).cpu().numpy()
            audit_rows.append({
                "fold": f + 1,
                "task": task,
                "acc_full": float((pred == data.labels[task]).mean()),
            })
    audit = pl.DataFrame(audit_rows).group_by("task").agg(
        pl.col("acc_full").mean().alias("acc_full_mean"),
        pl.col("acc_full").std().alias("acc_full_std"),
    ).sort("task")
    audit.write_csv(out_dir / "task_accuracy_audit.csv")

    # ---------------- save logits ----------------------------------------- #
    np.savez_compressed(
        out_dir / "fold_logits.npz",
        **{k: v.astype(np.float32) for k, v in store.items() if v is not None},
        **{f"alpha__{t}": alpha[t].astype(np.float32) for t in TASKS_OF_INTEREST},
    )

    # ---------------- row metadata ---------------------------------------- #
    meta = data.frame.select([
        c for c in ("sample_id", "source_name", "component_count", "compound_name", "smiles",
                    "nitro", "nitriles", "_eval_name")
        if c in data.frame.columns
    ]).with_columns([
        pl.Series("row_index", np.arange(n)),
        pl.Series("compound_key", data.compounds.astype(str)),
        pl.Series("nitro_label", data.labels["nitro_3class"]),
        pl.Series("nitriles_label", data.labels["nitriles_3class"]),
    ])
    meta.write_parquet(out_dir / "rows.parquet")

    # ---------------- support tables -------------------------------------- #
    support = pl.DataFrame([C.support_table(data, g) for g in C.GROUPS])
    support.write_csv(out_dir / "support_main_test.csv")
    print("\n[support / main_test]")
    print(support)

    train = pl.read_parquet(C.TRAIN_PARQUET) if C.TRAIN_PARQUET.is_file() else None
    if train is None:
        print(f"\n[support / train pool] skipped: {C.TRAIN_PARQUET} not found "
              f"(it is written by scripts/train.py, which is not part of the release)")
    else:
        train_rows = []
        for g in C.GROUPS:
            col = C.GROUPS[g]["column"]
            counts = train[col].to_numpy()
            train_rows.append({
                "group": g,
                "train_n_c0": int((counts == 0).sum()),
                "train_n_c1": int((counts == 1).sum()),
                "train_n_c2plus": int((counts >= 2).sum()),
                "train_n_total": int(len(counts)),
            })
        pl.DataFrame(train_rows).write_csv(out_dir / "support_train.csv")
        print("\n[support / train pool]")
        print(pl.DataFrame(train_rows))

    # ---------------- alpha / branch scale audit -------------------------- #
    alpha_rows = []
    for task in TASKS_OF_INTEREST:
        for f, a in enumerate(alpha[task], start=1):
            alpha_rows.append({"task": task, "fold": f, "alpha": float(a)})
    pl.DataFrame(alpha_rows).write_csv(out_dir / "alpha_per_fold.csv")
    print("\n[alpha per fold]")
    print(pl.DataFrame(alpha_rows).pivot(on="fold", index="task", values="alpha"))

    C.write_json(out_dir / "run_meta.json", {
        "eval_name": C.EVAL_NAME,
        "n_rows": int(n),
        "folds": ens.fold_files,
        "device": str(device),
        "run_dir": str(ens.run_dir),
        "tasks": list(TASKS_OF_INTEREST),
        "chemistry_band_coverage_pct": {g: C.characteristic_coverage(g) for g in C.GROUPS},
        "chem_bands": {g: C.FUNCTIONAL_GROUP_REGIONS[g] for g in C.GROUPS},
    })
    print(f"\n[out] {out_dir}")


if __name__ == "__main__":
    main()
