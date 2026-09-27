#!/usr/bin/env python
"""Paired raw-vs-digitized prediction analysis for CKAN-SpecNet.

Background
----------
`results/raw_1000_test_9_1` and `results/digital_1000_test_9_1` only store
task-level / class-level aggregate metrics, not per-sample predicted
probabilities, so they cannot answer the reviewer question "do identical spectra
receive identical predictions?". This script:

1. Re-runs inference with the same released 5-fold ensemble on the raw and on the
   digitized parquet, and exports per-sample, per-task, per-fold softmax
   probabilities (cached as .npz);
2. Aligns the two datasets by `sample_id` and computes paired agreement metrics:
   - binary prediction agreement (21 x 1000 decisions)
   - multiplicity-class agreement (12 x 1000 decisions)
   - mean absolute change in predicted probability
   - Cohen's kappa (binary) / quadratic weighted kappa (multiclass)
   - correctness transitions (both correct / raw-only / digital-only / both wrong)
   - per-sample mean number of flipped tasks and the relation between spectral
     fidelity and the number of flips
3. Writes CSV tables, figures and summary.json for the paper's supplementary
   material and the rebuttal.

Usage
-----
    python scripts/row_and_digital_comparation/analyze_raw_vs_digital.py \\
        --raw scripts/row_and_digital_comparation/row_selected_spectra.parquet \\
        --digital scripts/row_and_digital_comparation/digital_selected_spectra.parquet \\
        --run-dir models \\
        --out-dir results/anylize/raw_and_digital

Note: importing `ikan` changes the process cwd to its installation directory, so
every path must be resolved to an absolute path before `ckan_specnet` is imported
(see LAUNCH_DIR).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

LAUNCH_DIR = Path.cwd()
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl

from ckan_specnet.core import (
    EVAL_NAME_COL,
    configure_torch_runtime,
    device_of,
)
from ckan_specnet.data import Preprocessor, frame_to_xy, make_loader
from ckan_specnet.eval import (
    load_fold_model,
    load_manifest,
    predict,
    summarize_metrics,
    task_metrics,
    tasks_from_manifest,
)

# import side effect of `ikan`: it changes the cwd to site-packages/ikan
os.chdir(LAUNCH_DIR)

GROUPS = ("binary", "multiclass", "overall")
# these two parquets use `sample_id`; the released test.parquet uses `_sample_id`
ID_COL_CANDIDATES = ("sample_id", "_sample_id")


def id_column(df: pl.DataFrame) -> str:
    for name in ID_COL_CANDIDATES:
        if name in df.columns:
            return name
    raise ValueError(f"no sample-id column in {df.columns}")


# --------------------------------------------------------------------------- #
# basic helpers
# --------------------------------------------------------------------------- #
def abs_path(value: str | Path, base: Path = LAUNCH_DIR) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def cohen_kappa(a, b, k: int, weights: str | None = None) -> float:
    """Cohen's kappa (weights=None unweighted; 'quadratic'/'linear' weighted).

    Equivalent to sklearn.metrics.cohen_kappa_score, but degenerate cases are
    handled explicitly.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    observed = np.zeros((k, k), dtype=np.float64)
    np.add.at(observed, (a, b), 1.0)
    n = observed.sum()
    if n == 0:
        return float("nan")
    expected = np.outer(observed.sum(1), observed.sum(0)) / n
    idx = np.arange(k)
    if weights == "quadratic":
        w = (idx[:, None] - idx[None, :]) ** 2 / max((k - 1) ** 2, 1)
    elif weights == "linear":
        w = np.abs(idx[:, None] - idx[None, :]) / max(k - 1, 1)
    else:
        w = 1.0 - np.eye(k)
    denom = float((w * expected).sum())
    if denom <= 0:
        return float("nan")
    return float(1.0 - (w * observed).sum() / denom)


def nanmean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def nanstd(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.std(ddof=0)) if arr.size else float("nan")


# --------------------------------------------------------------------------- #
# 1. inference: export per-sample probabilities
# --------------------------------------------------------------------------- #
def infer_probabilities(
    parquet: Path,
    run_dir: Path,
    out_npz: Path,
    batch_size: int,
    num_workers: int,
    device,
) -> dict:
    """Run the 5-fold ensemble on every eval subset of the parquet; save per-fold and ensemble probabilities."""
    manifest = load_manifest(run_dir)
    tasks = tasks_from_manifest(manifest)
    df = pl.read_parquet(parquet)
    fold_files = list(manifest["ensemble"]["fold_files"])
    normalize = manifest["config"]["normalize"]

    payload: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}

    for eval_name in sorted(df[EVAL_NAME_COL].unique().to_list()):
        sub = df.filter(pl.col(EVAL_NAME_COL) == eval_name)
        x, y, _ = frame_to_xy(sub, tasks)
        print(f"[infer] {parquet.name} :: {eval_name}  x={x.shape}  folds={len(fold_files)}")

        fold_probs: list[dict[str, np.ndarray]] = []
        for fold_idx, fold_file in enumerate(fold_files, start=1):
            model, _ = load_fold_model(run_dir / fold_file, tasks)
            model = model.to(device)
            loader = make_loader(
                x=x,
                y=None,
                preprocessor=Preprocessor(normalize),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                device=device,
            )
            _, _, prob = predict(model, loader, tasks, device)
            fold_probs.append({t: np.asarray(prob[t], dtype=np.float64) for t in tasks.all})
            model.to("cpu")
            del model
            print(f"[infer]   fold {fold_idx}/{len(fold_files)} done", flush=True)

        ensemble = {
            t: np.mean([fp[t] for fp in fold_probs], axis=0) for t in tasks.all
        }

        for t in tasks.all:
            payload[f"ens__{t}"] = ensemble[t]
            payload[f"y__{t}"] = np.asarray(y[t], dtype=np.int64)
            for fold_idx, fp in enumerate(fold_probs, start=1):
                payload[f"fold{fold_idx}__{t}"] = fp[t]

        payload[f"{eval_name}__sample_id"] = np.asarray(sub[id_column(sub)].cast(pl.Utf8).to_list())
        meta[eval_name] = {
            "n_samples": int(sub.height),
            "eval_name": eval_name,
            "parquet": str(parquet),
            "run_dir": str(run_dir),
            "fold_files": fold_files,
            "normalize": normalize,
            "tasks": list(tasks.all),
            "num_classes": {t: int(tasks.num_classes[t]) for t in tasks.all},
        }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)
    meta_path = out_npz.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[infer] saved {out_npz} ({out_npz.stat().st_size / 1e6:.1f} MB)")
    return meta


# --------------------------------------------------------------------------- #
# 2. paired agreement metrics
# --------------------------------------------------------------------------- #
def paired_task_metrics(
    y_true: np.ndarray,
    p_raw: np.ndarray,
    p_dig: np.ndarray,
    num_classes: int,
) -> dict:
    """Paired decision metrics of one task (per-decision quantities, aggregated per task/group afterwards)."""
    idx = np.arange(len(y_true))
    pred_r = p_raw.argmax(1)
    pred_d = p_dig.argmax(1)

    agree = pred_r == pred_d
    top1_r = p_raw[idx, pred_r]          # = max(p_raw) (decision probability of the raw model)
    top1_d = p_dig[idx, pred_d]
    p_dig_at_r = p_dig[idx, pred_r]      # probability the digital model assigns to the raw decision

    delta_decision = np.abs(top1_r - p_dig_at_r)          # decision-probability change (paired, same class)
    delta_top1 = np.abs(top1_r - top1_d)                  # top-1 confidence change
    delta_dist = np.abs(p_raw - p_dig).mean(1)            # mean absolute change over the whole distribution
    tvd = 0.5 * np.abs(p_raw - p_dig).sum(1)              # total variation distance

    correct_r = pred_r == y_true
    correct_d = pred_d == y_true

    both_correct = correct_r & correct_d
    raw_only = correct_r & ~correct_d
    dig_only = ~correct_r & correct_d
    both_wrong = ~correct_r & ~correct_d

    flips = ~agree
    flip_w2r = flips & ~correct_r & correct_d   # wrong -> correct after digitization
    flip_r2w = flips & correct_r & ~correct_d   # correct -> wrong after digitization
    flip_ww = flips & ~correct_r & ~correct_d   # both wrong but different class
    flip_cc = flips & correct_r & correct_d     # both correct but different class

    n_pos = int((y_true == 1).sum()) if num_classes == 2 else 0
    pos_agree = float(agree[y_true == 1].mean() * 100) if n_pos else float("nan")
    neg_agree = (
        float(agree[y_true == 0].mean() * 100) if num_classes == 2 and (y_true == 0).any() else float("nan")
    )

    # binary: flip direction (1->0 lost detection / 0->1 new detection)
    flip_pos_to_neg = flips & (pred_r == 1) & (pred_d == 0)
    flip_neg_to_pos = flips & (pred_r == 0) & (pred_d == 1)

    # |dp| split into "prediction agrees / prediction flips"
    agree_mask = agree
    flip_mask = flips
    mad_decision_agree = float(delta_decision[agree_mask].mean() * 100) if agree_mask.any() else float("nan")
    mad_decision_flip = float(delta_decision[flip_mask].mean() * 100) if flip_mask.any() else float("nan")

    return {
        "n_decisions": int(len(y_true)),
        "agreement_pct": float(agree.mean() * 100),
        "disagreement_n": int(flips.sum()),
        "disagreement_pct": float(flips.mean() * 100),
        "kappa": cohen_kappa(pred_r, pred_d, num_classes),
        "qwk": cohen_kappa(pred_r, pred_d, num_classes, weights="quadratic")
        if num_classes > 2
        else float("nan"),
        "mad_decision_prob_pp": float(delta_decision.mean() * 100),
        "mad_decision_prob_p95_pp": float(np.percentile(delta_decision, 95) * 100),
        "mad_decision_prob_max_pp": float(delta_decision.max() * 100),
        "mad_decision_prob_agree_pp": mad_decision_agree,
        "mad_decision_prob_flip_pp": mad_decision_flip,
        "mad_top1_conf_pp": float(delta_top1.mean() * 100),
        "mad_distribution_pp": float(delta_dist.mean() * 100),
        "mean_tvd_pp": float(tvd.mean() * 100),
        "signed_top1_conf_pp": float((top1_d - top1_r).mean() * 100),
        "mean_conf_raw_pp": float(top1_r.mean() * 100),
        "mean_conf_dig_pp": float(top1_d.mean() * 100),
        "acc_raw_pct": float(correct_r.mean() * 100),
        "acc_dig_pct": float(correct_d.mean() * 100),
        "delta_acc_pp": float((correct_d.mean() - correct_r.mean()) * 100),
        "both_correct_pct": float(both_correct.mean() * 100),
        "raw_only_correct_pct": float(raw_only.mean() * 100),
        "dig_only_correct_pct": float(dig_only.mean() * 100),
        "both_wrong_pct": float(both_wrong.mean() * 100),
        "flips_n": int(flips.sum()),
        "flip_wrong_to_right_n": int(flip_w2r.sum()),
        "flip_right_to_wrong_n": int(flip_r2w.sum()),
        "flip_both_wrong_n": int(flip_ww.sum()),
        "flip_both_correct_n": int(flip_cc.sum()),
        "flip_pos_to_neg_n": int(flip_pos_to_neg.sum()),
        "flip_neg_to_pos_n": int(flip_neg_to_pos.sum()),
        "net_recovered_n": int(flip_w2r.sum() - flip_r2w.sum()),
        "pos_agreement_pct": pos_agree,
        "neg_agreement_pct": neg_agree,
        "n_pos": n_pos,
        "n_neg": int((y_true == 0).sum()) if num_classes == 2 else 0,
        "mean_flips_per_sample": float(flips.mean()),
    }


def aggregate_group(per_task_rows: list[dict], group_tasks: list[str]) -> dict:
    """Aggregate every task of one group into a single row (macro average + pooled agreement)."""
    rows = [r for r in per_task_rows if r["task"] in group_tasks]
    if not rows:
        return {}

    total_decisions = sum(r["n_decisions"] for r in rows)
    total_disagree = sum(r["disagreement_n"] for r in rows)
    total_flips_w2r = sum(r["flip_wrong_to_right_n"] for r in rows)
    total_flips_r2w = sum(r["flip_right_to_wrong_n"] for r in rows)

    def macro(key: str) -> float:
        return nanmean([r[key] for r in rows])

    def macro_std(key: str) -> float:
        return nanstd([r[key] for r in rows])

    return {
        "group": None,
        "n_tasks": len(rows),
        "n_decisions": total_decisions,
        "n_samples": rows[0]["n_decisions"],
        # pooled (all decisions together)
        "agreement_pct": float((total_decisions - total_disagree) / total_decisions * 100),
        "disagreement_n": total_disagree,
        "disagreement_pct": float(total_disagree / total_decisions * 100),
        # macro (average within a task first, then across tasks)
        "agreement_pct_macro": macro("agreement_pct"),
        "agreement_pct_macro_std": macro_std("agreement_pct"),
        "kappa_macro": macro("kappa"),
        "kappa_macro_std": macro_std("kappa"),
        "qwk_macro": macro("qwk"),
        "qwk_macro_std": macro_std("qwk"),
        "mad_decision_prob_pp": macro("mad_decision_prob_pp"),
        "mad_decision_prob_pp_std": macro_std("mad_decision_prob_pp"),
        "mad_decision_prob_p95_pp": macro("mad_decision_prob_p95_pp"),
        "mad_decision_prob_agree_pp": macro("mad_decision_prob_agree_pp"),
        "mad_decision_prob_flip_pp": macro("mad_decision_prob_flip_pp"),
        "mad_top1_conf_pp": macro("mad_top1_conf_pp"),
        "mad_top1_conf_pp_std": macro_std("mad_top1_conf_pp"),
        "mad_distribution_pp": macro("mad_distribution_pp"),
        "mad_distribution_pp_std": macro_std("mad_distribution_pp"),
        "mean_tvd_pp": macro("mean_tvd_pp"),
        "mean_tvd_pp_std": macro_std("mean_tvd_pp"),
        "signed_top1_conf_pp": macro("signed_top1_conf_pp"),
        "mean_conf_raw_pp": macro("mean_conf_raw_pp"),
        "mean_conf_dig_pp": macro("mean_conf_dig_pp"),
        "acc_raw_pct": macro("acc_raw_pct"),
        "acc_dig_pct": macro("acc_dig_pct"),
        "delta_acc_pp": macro("delta_acc_pp"),
        "delta_acc_pp_std": macro_std("delta_acc_pp"),
        "both_correct_pct": macro("both_correct_pct"),
        "raw_only_correct_pct": macro("raw_only_correct_pct"),
        "dig_only_correct_pct": macro("dig_only_correct_pct"),
        "both_wrong_pct": macro("both_wrong_pct"),
        "flips_n": int(sum(r["flips_n"] for r in rows)),
        "flip_wrong_to_right_n": total_flips_w2r,
        "flip_right_to_wrong_n": total_flips_r2w,
        "flip_pos_to_neg_n": int(sum(r["flip_pos_to_neg_n"] for r in rows)),
        "flip_neg_to_pos_n": int(sum(r["flip_neg_to_pos_n"] for r in rows)),
        "net_recovered_n": int(total_flips_w2r - total_flips_r2w),
        "pos_agreement_pct": macro("pos_agreement_pct"),
        "neg_agreement_pct": macro("neg_agreement_pct"),
        "mean_flips_per_sample_macro": macro("mean_flips_per_sample"),
    }


# --------------------------------------------------------------------------- #
# 3. spectral fidelity
# --------------------------------------------------------------------------- #
def spectral_fidelity(raw_df: pl.DataFrame, dig_df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    raw_id_col = id_column(raw_df)
    dig_id_col = id_column(dig_df)
    raw_map = {
        str(s): np.asarray(v, dtype=np.float64)
        for s, v in zip(raw_df[raw_id_col], raw_df["spectrum"])
    }
    dig_map = {
        str(s): np.asarray(v, dtype=np.float64)
        for s, v in zip(dig_df[dig_id_col], dig_df["spectrum"])
    }
    ids = sorted(set(raw_map) & set(dig_map))

    rows = []
    for sid in ids:
        a, b = raw_map[sid], dig_map[sid]
        resid = b - a
        ss_res = float((resid**2).sum())
        ss_tot = float(((a - a.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        denom = float(a.max() - a.min())
        rows.append(
            {
                "sample_id": sid,
                "r2": r2,
                "pearson_r": r,
                "rmse": float(np.sqrt((resid**2).mean())),
                "mae": float(np.abs(resid).mean()),
                "max_abs_err": float(np.abs(resid).max()),
                "nrmse_pct_of_range": float(np.sqrt((resid**2).mean()) / denom * 100) if denom > 0 else float("nan"),
                "cosine_similarity": float(
                    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                ),
            }
        )

    df = pl.DataFrame(rows)
    summary = {
        "n_pairs": len(rows),
        "r2_mean": float(df["r2"].mean()),
        "r2_median": float(df["r2"].median()),
        "r2_p05": float(df["r2"].quantile(0.05)),
        "r2_min": float(df["r2"].min()),
        "pearson_r_mean": float(df["pearson_r"].mean()),
        "rmse_mean": float(df["rmse"].mean()),
        "rmse_median": float(df["rmse"].median()),
        "rmse_p95": float(df["rmse"].quantile(0.95)),
        "mae_mean": float(df["mae"].mean()),
        "max_abs_err_p95": float(df["max_abs_err"].quantile(0.95)),
        "nrmse_pct_of_range_mean": float(df["nrmse_pct_of_range"].mean()),
        "cosine_similarity_mean": float(df["cosine_similarity"].mean()),
        "cosine_similarity_min": float(df["cosine_similarity"].min()),
    }
    return df, summary


# --------------------------------------------------------------------------- #
# 4. main flow
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default="data/raw_selected_spectra.parquet")
    parser.add_argument("--digital", default="data/digital_selected_spectra.parquet")
    parser.add_argument("--run-dir", default="models/ckan_specnet_5fold")
    parser.add_argument("--out-dir", default="results/anylize/raw_and_digital")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--ref-raw-summary",
        default="results/raw_1000_test_9_1/summary.csv",
        help="summary.csv of the existing (raw) run, used for reproduction checks",
    )
    parser.add_argument(
        "--ref-digital-summary",
        default="results/digital_1000_test_9_1/summary.csv",
        help="summary.csv of the existing (digitized) run, used for reproduction checks",
    )
    parser.add_argument("--reuse-cache", action="store_true", help="reuse the cached .npz files and skip inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = abs_path(args.raw)
    dig_path = abs_path(args.digital)
    run_dir = abs_path(args.run_dir)
    out_dir = abs_path(args.out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    cache_dir = out_dir / "predictions"
    for d in (out_dir, tables_dir, figures_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    configure_torch_runtime()
    import torch

    torch.set_num_threads(int(args.threads))
    if args.device == "auto":
        device = device_of()
    else:
        device = torch.device(args.device)
    print(f"[env] device={device} torch_threads={torch.get_num_threads()}")

    raw_npz = cache_dir / "raw_ensemble_probabilities.npz"
    dig_npz = cache_dir / "digital_ensemble_probabilities.npz"

    if args.reuse_cache and raw_npz.is_file() and dig_npz.is_file():
        print("[infer] reuse cached probabilities")
    else:
        infer_probabilities(raw_path, run_dir, raw_npz, args.batch_size, args.num_workers, device)
        infer_probabilities(dig_path, run_dir, dig_npz, args.batch_size, args.num_workers, device)

    raw_meta_all = json.loads(raw_npz.with_suffix(".meta.json").read_text())
    dig_meta_all = json.loads(dig_npz.with_suffix(".meta.json").read_text())
    raw_pred = np.load(raw_npz, allow_pickle=False)
    dig_pred = np.load(dig_npz, allow_pickle=False)

    raw_eval = next(iter(raw_meta_all))
    dig_eval = next(iter(dig_meta_all))
    raw_meta = raw_meta_all[raw_eval]
    dig_meta = dig_meta_all[dig_eval]
    tasks = raw_meta["tasks"]
    num_classes = {t: int(k) for t, k in raw_meta["num_classes"].items()}
    binary_tasks = [t for t in tasks if num_classes[t] == 2]
    multi_tasks = [t for t in tasks if num_classes[t] > 2]
    n_folds = len(raw_meta["fold_files"])
    print(f"[info] tasks={len(tasks)} (binary={len(binary_tasks)}, multiclass={len(multi_tasks)}), folds={n_folds}")

    # ---- paired alignment (sorted by sample_id so that both models see the same spectrum) ----
    raw_ids = [str(s) for s in raw_pred[f"{raw_eval}__sample_id"]]
    dig_ids = [str(s) for s in dig_pred[f"{dig_eval}__sample_id"]]
    common = sorted(set(raw_ids) & set(dig_ids))
    if len(common) != len(raw_ids) or len(common) != len(dig_ids):
        raise ValueError(f"sample_id mismatch: raw={len(raw_ids)} digital={len(dig_ids)} common={len(common)}")
    r_order = np.array([raw_ids.index(s) for s in common])
    d_order = np.array([dig_ids.index(s) for s in common])
    print(f"[info] aligned {len(common)} paired spectra by sample_id")

    y_true = {t: raw_pred[f"y__{t}"][r_order] for t in tasks}
    for t in tasks:
        if not np.array_equal(y_true[t], dig_pred[f"y__{t}"][d_order]):
            raise ValueError(f"label mismatch between raw/digital for task {t}")

    p_raw_ens = {t: raw_pred[f"ens__{t}"][r_order] for t in tasks}
    p_dig_ens = {t: dig_pred[f"ens__{t}"][d_order] for t in tasks}
    p_raw_fold = [
        {t: raw_pred[f"fold{f}__{t}"][r_order] for t in tasks} for f in range(1, n_folds + 1)
    ]
    p_dig_fold = [
        {t: dig_pred[f"fold{f}__{t}"][d_order] for t in tasks} for f in range(1, n_folds + 1)
    ]

    # ---- per-task metrics (ensemble + per fold) ----
    per_task_ens = []
    per_task_fold: list[list[dict]] = [[] for _ in range(n_folds)]
    for t in tasks:
        row = paired_task_metrics(y_true[t], p_raw_ens[t], p_dig_ens[t], num_classes[t])
        row["task"] = t
        row["n_classes"] = num_classes[t]
        row["group"] = "binary" if num_classes[t] == 2 else "multiclass"
        per_task_ens.append(row)
        for f in range(n_folds):
            frow = paired_task_metrics(y_true[t], p_raw_fold[f][t], p_dig_fold[f][t], num_classes[t])
            frow["task"] = t
            per_task_fold[f].append(frow)

    # per-fold group summaries -> across-fold std (same convention as the paper's +-std)
    fold_group_rows = []
    for f in range(n_folds):
        for group, group_tasks in (("binary", binary_tasks), ("multiclass", multi_tasks), ("overall", tasks)):
            agg = aggregate_group(per_task_fold[f], group_tasks)
            agg["group"] = group
            agg["fold"] = f + 1
            fold_group_rows.append(agg)
    fold_df = pl.DataFrame(fold_group_rows)

    def fold_std(group: str, key: str) -> float:
        sub = fold_df.filter(pl.col("group") == group)
        return float(sub[key].std(ddof=0)) if sub.height and key in sub.columns else float("nan")

    group_rows = []
    for group, group_tasks in (("binary", binary_tasks), ("multiclass", multi_tasks), ("overall", tasks)):
        agg = aggregate_group(per_task_ens, group_tasks)
        agg["group"] = group
        agg["agreement_pct_fold_std"] = fold_std(group, "agreement_pct")
        agg["kappa_macro_fold_std"] = fold_std(group, "kappa_macro")
        agg["qwk_macro_fold_std"] = fold_std(group, "qwk_macro")
        agg["mad_decision_prob_pp_fold_std"] = fold_std(group, "mad_decision_prob_pp")
        agg["mad_top1_conf_pp_fold_std"] = fold_std(group, "mad_top1_conf_pp")
        agg["mad_distribution_pp_fold_std"] = fold_std(group, "mad_distribution_pp")
        agg["mean_tvd_pp_fold_std"] = fold_std(group, "mean_tvd_pp")
        agg["agreement_pct_macro_fold_std"] = fold_std(group, "agreement_pct_macro")
        group_rows.append(agg)

    # ---- per-sample flip statistics ----
    flip_counts = np.zeros(len(common), dtype=np.int64)
    for t in tasks:
        flip_counts += (p_raw_ens[t].argmax(1) != p_dig_ens[t].argmax(1)).astype(np.int64)

    flip_dist = (
        pl.DataFrame({"n_flipped_tasks": flip_counts})
        .group_by("n_flipped_tasks")
        .agg(pl.len().alias("n_spectra"))
        .sort("n_flipped_tasks")
        .with_columns((pl.col("n_spectra") / len(common) * 100).alias("pct_spectra"))
    )

    # ---- per-decision |dp| (for the distribution figure) ----
    delta_dec: dict[str, list[np.ndarray]] = {"binary": [], "multiclass": []}
    for t in tasks:
        pred_r = p_raw_ens[t].argmax(1)
        rows_i = np.arange(len(pred_r))
        change = np.abs(p_raw_ens[t][rows_i, pred_r] - p_dig_ens[t][rows_i, pred_r]) * 100
        delta_dec["binary" if num_classes[t] == 2 else "multiclass"].append(change)
    delta_dec = {k: np.concatenate(v) for k, v in delta_dec.items()}

    # ---- spectral fidelity ----
    raw_df = pl.read_parquet(raw_path)
    raw_df = raw_df.with_columns(pl.col(id_column(raw_df)).cast(pl.Utf8))
    dig_df = pl.read_parquet(dig_path)
    dig_df = dig_df.with_columns(pl.col(id_column(dig_df)).cast(pl.Utf8))
    fidelity_df, fidelity_summary = spectral_fidelity(raw_df, dig_df)
    fidelity_df = fidelity_df.join(
        pl.DataFrame(
            {
                "sample_id": common,
                "aligned_row": np.arange(len(common)),
                "n_flipped_tasks": flip_counts,
            }
        ),
        on="sample_id",
        how="inner",
    ).sort("aligned_row")

    # correlation between fidelity and the number of flips (Spearman)
    from scipy import stats as scipy_stats

    spearman_r2 = scipy_stats.spearmanr(fidelity_df["r2"], fidelity_df["n_flipped_tasks"])
    spearman_rmse = scipy_stats.spearmanr(fidelity_df["rmse"], fidelity_df["n_flipped_tasks"])
    correlation_summary = {
        "spearman_r2_vs_flips_rho": float(spearman_r2.statistic),
        "spearman_r2_vs_flips_p": float(spearman_r2.pvalue),
        "spearman_rmse_vs_flips_rho": float(spearman_rmse.statistic),
        "spearman_rmse_vs_flips_p": float(spearman_rmse.pvalue),
    }

    mean_flips = float(flip_counts.mean())
    per_sample_summary = {
        "n_spectra": int(len(common)),
        "n_tasks": len(tasks),
        "mean_flipped_tasks_per_spectrum": mean_flips,
        "median_flipped_tasks_per_spectrum": float(np.median(flip_counts)),
        "pct_spectra_zero_flips": float((flip_counts == 0).mean() * 100),
        "pct_spectra_le1_flip": float((flip_counts <= 1).mean() * 100),
        "pct_spectra_le3_flips": float((flip_counts <= 3).mean() * 100),
        "max_flipped_tasks": int(flip_counts.max()),
    }

    # ---- reproduction check: compare with results/*_1000_test_9_1/summary.csv ----
    verification_rows = []
    for label, probs, ref_path in (
        ("raw", p_raw_ens, abs_path(args.ref_raw_summary)),
        ("digital", p_dig_ens, abs_path(args.ref_digital_summary)),
    ):
        metrics, _ = task_metrics(y_true, {t: probs[t].argmax(1) for t in tasks}, _tasks_from(tasks, num_classes))
        summary = summarize_metrics(metrics)
        row = {"model": label, "reference_summary": str(ref_path), "reference_found": ref_path.is_file()}
        row.update(summary)
        if ref_path.is_file():
            ref = pl.read_csv(ref_path)
            shared = [c for c in summary if c in ref.columns and ref.height == 1]
            diffs = {
                f"max_abs_diff__{c}": abs(float(ref[c][0]) - float(summary[c]))
                for c in shared
                if np.isfinite(float(summary[c])) and np.isfinite(float(ref[c][0]))
            }
            row["n_metrics_compared"] = len(diffs)
            row["max_abs_diff_overall"] = max(diffs.values()) if diffs else float("nan")
            row.update(diffs)
        verification_rows.append(row)
    verification_df = pl.DataFrame(verification_rows)

    # ---- save tables ----
    per_task_df = pl.DataFrame(per_task_ens).select(
        [
            "task",
            "group",
            "n_classes",
            "n_decisions",
            "agreement_pct",
            "disagreement_n",
            "disagreement_pct",
            "kappa",
            "qwk",
            "mad_decision_prob_pp",
            "mad_decision_prob_p95_pp",
            "mad_decision_prob_agree_pp",
            "mad_decision_prob_flip_pp",
            "mad_top1_conf_pp",
            "mad_distribution_pp",
            "mean_tvd_pp",
            "signed_top1_conf_pp",
            "mean_conf_raw_pp",
            "mean_conf_dig_pp",
            "acc_raw_pct",
            "acc_dig_pct",
            "delta_acc_pp",
            "both_correct_pct",
            "raw_only_correct_pct",
            "dig_only_correct_pct",
            "both_wrong_pct",
            "flips_n",
            "flip_wrong_to_right_n",
            "flip_right_to_wrong_n",
            "flip_pos_to_neg_n",
            "flip_neg_to_pos_n",
            "net_recovered_n",
        ]
    )
    per_task_df.write_csv(tables_dir / "tableS6_paired_agreement_per_task.csv")

    group_df = pl.DataFrame(group_rows)
    group_df.write_csv(tables_dir / "tableS5_paired_agreement_summary.csv")

    fold_df.write_csv(tables_dir / "tableS5b_paired_agreement_per_fold.csv")
    flip_dist.write_csv(tables_dir / "tableS10_flip_count_distribution.csv")
    fidelity_df.write_csv(tables_dir / "tableS9_spectral_fidelity_per_sample.csv")
    verification_df.write_csv(tables_dir / "tableS11_verification_vs_saved_summary.csv")

    # binary: agreement split by presence / absence
    binary_df = per_task_df.filter(pl.col("group") == "binary")
    binary_rows = []
    for t in binary_tasks:
        row = next(r for r in per_task_ens if r["task"] == t)
        binary_rows.append(
            {
                "task": t,
                "n_pos": row["n_pos"],
                "n_neg": row["n_neg"],
                "agreement_pct": row["agreement_pct"],
                "agreement_pos_pct": row["pos_agreement_pct"],
                "agreement_neg_pct": row["neg_agreement_pct"],
                "kappa": row["kappa"],
                "mad_decision_prob_pp": row["mad_decision_prob_pp"],
                "acc_raw_pct": row["acc_raw_pct"],
                "acc_dig_pct": row["acc_dig_pct"],
                "delta_acc_pp": row["delta_acc_pp"],
                "flips_n": row["flips_n"],
                "flip_lost_detection_n": row["flip_pos_to_neg_n"],
                "flip_gained_detection_n": row["flip_neg_to_pos_n"],
            }
        )
    pl.DataFrame(binary_rows).write_csv(tables_dir / "tableS7_binary_presence_vs_absence.csv")

    # multiclass: agreement per true class
    multiclass_rows = []
    for t in multi_tasks:
        yt = y_true[t]
        pr = p_raw_ens[t].argmax(1)
        pd_ = p_dig_ens[t].argmax(1)
        agree = pr == pd_
        for c in range(num_classes[t]):
            mask = yt == c
            if not mask.any():
                continue
            multiclass_rows.append(
                {
                    "task": t,
                    "true_class": c,
                    "n": int(mask.sum()),
                    "agreement_pct": float(agree[mask].mean() * 100),
                    "acc_raw_pct": float((pr[mask] == c).mean() * 100),
                    "acc_dig_pct": float((pd_[mask] == c).mean() * 100),
                }
            )
    pl.DataFrame(multiclass_rows).write_csv(tables_dir / "tableS8_multiclass_agreement_by_true_class.csv")

    # ---- summary.json ----
    key_numbers = {
        "config": {
            "raw_parquet": str(raw_path),
            "digital_parquet": str(dig_path),
            "run_dir": str(run_dir),
            "n_samples": len(common),
            "n_tasks": len(tasks),
            "n_binary_tasks": len(binary_tasks),
            "n_multiclass_tasks": len(multi_tasks),
            "n_folds": n_folds,
            "binary_tasks": binary_tasks,
            "multiclass_tasks": multi_tasks,
        },
        "groups": {row["group"]: row for row in group_rows},
        "per_sample": per_sample_summary,
        "spectral_fidelity": fidelity_summary,
        "fidelity_vs_flips": correlation_summary,
        "per_task": {r["task"]: r for r in per_task_ens},
    }
    (out_dir / "summary.json").write_text(json.dumps(key_numbers, ensure_ascii=False, indent=2, default=str))

    # ---- figures ----
    make_figures(per_task_df, group_rows, fidelity_df, flip_counts, delta_dec, tables_dir, figures_dir)

    # ---- Markdown tables ready to paste into the paper ----
    export_markdown_tables(
        tables_dir=tables_dir,
        per_task_ens=per_task_ens,
        group_rows=group_rows,
        binary_rows=binary_rows,
        multiclass_rows=multiclass_rows,
        fidelity_summary=fidelity_summary,
        per_sample_summary=per_sample_summary,
        flip_dist=flip_dist,
        verification_df=verification_df,
    )

    # ---- console summary ----
    print("\n================ main results ================")
    for row in group_rows:
        qwk_txt = "" if not np.isfinite(row["qwk_macro"]) else f" | QWK={row['qwk_macro']:.4f}"
        print(
            f"{row['group']:>10s} | decisions={row['n_decisions']:>6d} | "
            f"agreement={row['agreement_pct']:.3f}%±{row['agreement_pct_fold_std']:.3f} "
            f"(macro {row['agreement_pct_macro']:.3f}%) | "
            f"kappa={row['kappa_macro']:.4f}{qwk_txt} | madΔp={row['mad_decision_prob_pp']:.3f}pp | "
            f"Δacc={row['delta_acc_pp']:+.3f}pp"
        )
    print(f"mean flipped tasks per spectrum: {mean_flips:.3f} / {len(tasks)}")
    print(f"spectral fidelity: R2={fidelity_summary['r2_mean']:.4f}, RMSE={fidelity_summary['rmse_mean']:.4f}")
    print(f"\nSaved tables/figures to: {out_dir}")


def _tasks_from(tasks: list[str], num_classes: dict[str, int]):
    from ckan_specnet.core import TaskCatalog

    binary = tuple(t for t in tasks if num_classes[t] == 2)
    ternary = tuple(t.replace("_3class", "") for t in tasks if num_classes[t] == 3)
    quaternary = tuple(t.replace("_4class", "") for t in tasks if num_classes[t] == 4)
    return TaskCatalog(binary=binary, ternary=ternary, quaternary=quaternary, num_classes=num_classes, all=tuple(tasks))


def _fmt(value, precision: int = 2, dash: str = "—") -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return dash
    if not np.isfinite(value):
        return dash
    return f"{value:.{precision}f}"


def _pm(value, std, precision: int = 2) -> str:
    if not np.isfinite(float(value)):
        return "—"
    if std is None or not np.isfinite(float(std)):
        return f"{float(value):.{precision}f}"
    return f"{float(value):.{precision}f} ± {float(std):.{precision}f}"


def export_markdown_tables(
    tables_dir: Path,
    per_task_ens: list[dict],
    group_rows: list[dict],
    binary_rows: list[dict],
    multiclass_rows: list[dict],
    fidelity_summary: dict,
    per_sample_summary: dict,
    flip_dist: pl.DataFrame,
    verification_df: pl.DataFrame,
) -> None:
    """Write Markdown tables that can be pasted straight into the paper / the reply."""
    groups = {row["group"]: row for row in group_rows}

    # ---------------- Table S5 ----------------
    lines = [
        "**Table S5.** Paired prediction agreement between raw and digitized spectra "
        "(1,000 paired spectra, 5-fold ensemble). Agreement, Cohen's κ and QWK are reported as "
        "mean ± standard deviation across the five folds; the remaining columns are computed from "
        "the ensemble predictions.",
        "",
        "| Decision set | Decisions | Agreement (%) | Cohen's κ | QWK | Mean \\|Δp\\| (pp) | Mean TVD (pp) | Δ Accuracy (pp) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    labels = {
        "binary": "Binary functional groups (21 tasks × 1,000)",
        "multiclass": "Multiplicity classes (12 tasks × 1,000)",
        "overall": "All tasks (33 tasks × 1,000)",
    }
    for key in ("binary", "multiclass", "overall"):
        row = groups[key]
        lines.append(
            "| {label} | {n:,} | {agree} | {kappa} | {qwk} | {mad} | {tvd} | {dacc} |".format(
                label=labels[key],
                n=row["n_decisions"],
                agree=_pm(row["agreement_pct"], row["agreement_pct_fold_std"], 3),
                kappa=_pm(row["kappa_macro"], row["kappa_macro_fold_std"], 4),
                qwk=_pm(row["qwk_macro"], row["qwk_macro_fold_std"], 4),
                mad=_fmt(row["mad_decision_prob_pp"], 3),
                tvd=_fmt(row["mean_tvd_pp"], 3),
                dacc=_fmt(row["delta_acc_pp"], 3),
            )
        )
    lines += [
        "",
        "Agreement = fraction of decisions for which the two inputs produce the same predicted class. "
        "Mean \\|Δp\\| = mean absolute change in the softmax probability of the raw-spectrum decision "
        "(percentage points). TVD = total variation distance between the two predicted distributions. "
        "κ is Cohen's κ between the two sets of predictions; QWK is its quadratic-weighted version "
        "(multiplicity tasks only). Δ Accuracy = digitized − raw accuracy.",
    ]
    (tables_dir / "tableS5_paired_agreement_summary.md").write_text("\n".join(lines) + "\n")

    # ---------------- Table S6 ----------------
    rows_sorted = sorted(per_task_ens, key=lambda r: (r["group"], r["agreement_pct"]))
    type_label = {"binary": "Binary", "multiclass": "Multiplicity"}

    lines = [
        "**Table S6.** Per-task paired prediction agreement between raw and digitized spectra "
        "(1,000 paired spectra, five-fold ensemble). All values are reported to three decimal places.",
        "",
        "| Task | Type | Agreement (%) | Changed decisions (/1,000) | Cohen's κ | QWK | Mean \\|Δp\\| (pp) | Δ Accuracy (pp) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r['task']} | {type_label[r['group']]} | {_fmt(r['agreement_pct'], 3)} | "
            f"{r['disagreement_n']} | {_fmt(r['kappa'], 3)} | {_fmt(r['qwk'], 3)} | "
            f"{_fmt(r['mad_decision_prob_pp'], 3)} | {_fmt(r['delta_acc_pp'], 3)} |"
        )
    worst = min(rows_sorted, key=lambda r: r["agreement_pct"])
    perfect = [r["task"] for r in rows_sorted if r["disagreement_n"] == 0]
    lines += [
        "",
        "**How to read this table**",
        "",
        "- **Task**: functional-group task name; **Type**: `Binary` = presence/absence of "
        "the functional group (2 classes); `Multiplicity` = number of functional groups "
        "(3 or 4 classes, i.e. a multiclass task).",
        "- **Agreement (%)**: fraction of spectra for which the raw and the digitized "
        "spectrum give **the same predicted class** (higher = more consistent); there are "
        "1,000 spectra, hence 1,000 decisions per task.",
        "- **Changed decisions (/1,000)**: number of the 1,000 spectra whose prediction "
        "changed (= 1,000 - agreement count).",
        "- **Cohen's kappa**: true agreement after removing chance agreement (closer to 1 = "
        "more consistent; > 0.9 is essentially identical); **QWK** is the class-order-aware "
        "weighted version, meaningful only for Multiplicity tasks, and shown as \"-\" for "
        "Binary tasks.",
        "- **Mean \\|dp\\| (pp)**: the probability the model assigns to **the class it chose "
        "itself**, averaged absolute change caused by digitization, in percentage points (pp); "
        "smaller means the model's confidence is less affected by digitization.",
        "- **d Accuracy (pp)**: **accuracy on digitized spectra - accuracy on raw spectra**, "
        "computed in a paired way on **the same model and the same 1,000 spectra**, in "
        "**percentage points (pp)**; positive = the task is more accurate on digitized "
        "spectra, negative = worse, 0 = identical accuracy. **Identical accuracy does not "
        "imply identical predictions** (correct and broken decisions can cancel out), so it "
        "must be read together with Agreement. Because every task has only 1,000 spectra, "
        "changing one prediction moves the accuracy by 0.1 pp, so all values in this column "
        "are multiples of 0.1.",
        "",
        "**Formulas (metric definitions)**",
        "",
        "- `Agreement (%) = (1/N) * sum 1[yhat_raw = yhat_dig] * 100`, N = 1,000.",
        "- `d Accuracy (pp) = [ (1/N) * sum 1(yhat_dig = y) - (1/N) * sum 1(yhat_raw = y) ] * 100`, "
        "where y is the true label; the unit is pp.",
        "- `Mean |Δp| (pp) = (1/N) · Σ |p_raw(ŷ_raw) − p_dig(ŷ_raw)| × 100`。",
        "",
        "**Table footnote (English, ready to paste into the manuscript):**",
        "",
        "> Δ Accuracy (pp) is the difference between the accuracy obtained on digitized spectra and on "
        "raw spectra (digitized − raw), computed per task on the same 1,000 paired spectra with the same "
        "five-fold ensemble and expressed in percentage points. Positive values indicate that the task is "
        "more accurate on digitized spectra. Because each task contains 1,000 spectra, a single changed "
        "decision corresponds to 0.1 pp. Note that an unchanged accuracy does not imply unchanged "
        "predictions, since corrected and broken decisions can cancel out. Group means across tasks: "
        "−0.033 pp for the 21 binary tasks and 0.000 pp for the 12 multiplicity tasks, i.e. an order of "
        "magnitude below the fold-to-fold variability of the model itself (±0.13 pp).",
        "",
        f"**Summary:** across the 33 tasks the agreement ranges from {_fmt(worst['agreement_pct'], 3)}% "
        f"({worst['task']}, {worst['disagreement_n']} changed decisions) to 100.000%; "
        f"{len(perfect)} tasks ({', '.join(perfect)}) give identical predictions for all 1,000 spectra.",
    ]
    (tables_dir / "tableS6_paired_agreement_per_task.md").write_text("\n".join(lines) + "\n")

    # companion CSV: the same 8 columns as the md (the 32-column wide table is no longer written)
    def _num(value, precision: int = 3):
        value = float(value)
        return round(value, precision) if np.isfinite(value) else None

    # companion CSV: the same 8 columns as the md (the 32-column wide table is no longer
    # written); all values are formatted with 3 decimals,
    # so that Excel / Word show exactly the same values as the md.
    def _num(value, precision: int = 3) -> str:
        value = float(value)
        return f"{value:.{precision}f}" if np.isfinite(value) else ""

    pl.DataFrame(
        [
            {
                "task": r["task"],
                "type": type_label[r["group"]],
                "agreement_pct": _num(r["agreement_pct"]),
                "changed_decisions_per_1000": r["disagreement_n"],
                "cohens_kappa": _num(r["kappa"]),
                "qwk": _num(r["qwk"]),
                "mean_abs_delta_p_pp": _num(r["mad_decision_prob_pp"]),
                "delta_accuracy_pp": _num(r["delta_acc_pp"]),
            }
            for r in rows_sorted
        ]
    ).write_csv(tables_dir / "tableS6_paired_agreement_per_task.csv")

    # ---------------- Table S7 ----------------
    lines = [
        "**Table S7.** Binary tasks: agreement on presence vs. absence decisions and the direction of flips.",
        "",
        "| Task | n (presence) | n (absence) | Agreement (%) | Agreement on presence (%) | Agreement on absence (%) | Cohen's κ | Flips (n) | Presence lost (1→0) | Presence gained (0→1) | Δ Acc (pp) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(binary_rows, key=lambda x: x["agreement_pct"]):
        lines.append(
            f"| {r['task']} | {r['n_pos']} | {r['n_neg']} | {_fmt(r['agreement_pct'], 2)} | "
            f"{_fmt(r['agreement_pos_pct'], 2)} | {_fmt(r['agreement_neg_pct'], 2)} | {_fmt(r['kappa'], 3)} | "
            f"{r['flips_n']} | {r['flip_lost_detection_n']} | {r['flip_gained_detection_n']} | {_fmt(r['delta_acc_pp'], 2)} |"
        )
    (tables_dir / "tableS7_binary_presence_vs_absence.md").write_text("\n".join(lines) + "\n")

    # ---------------- Table S8 ----------------
    lines = [
        "**Table S8.** Multiplicity tasks: agreement broken down by true count level.",
        "",
        "| Task | True count level | n | Agreement (%) | Accuracy raw (%) | Accuracy digitized (%) |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(multiclass_rows, key=lambda x: (x["task"], x["true_class"])):
        lines.append(
            f"| {r['task']} | C{r['true_class']} | {r['n']} | {_fmt(r['agreement_pct'], 2)} | "
            f"{_fmt(r['acc_raw_pct'], 2)} | {_fmt(r['acc_dig_pct'], 2)} |"
        )
    (tables_dir / "tableS8_multiclass_agreement_by_true_class.md").write_text("\n".join(lines) + "\n")

    # ---------------- Table S9 ----------------
    lines = [
        "**Table S9.** Spectral fidelity of the digitization on the 1,000 paired spectra "
        "(intensity normalised to [0, 1]).",
        "",
        "| Statistic | Value |",
        "|---|---|",
        f"| Paired spectra | {fidelity_summary['n_pairs']:,} |",
        f"| Per-spectrum R², mean | {_fmt(fidelity_summary['r2_mean'], 4)} |",
        f"| Per-spectrum R², median | {_fmt(fidelity_summary['r2_median'], 4)} |",
        f"| Per-spectrum R², 5th percentile | {_fmt(fidelity_summary['r2_p05'], 4)} |",
        f"| Per-spectrum R², minimum | {_fmt(fidelity_summary['r2_min'], 4)} |",
        f"| Pearson r, mean | {_fmt(fidelity_summary['pearson_r_mean'], 4)} |",
        f"| RMSE, mean | {_fmt(fidelity_summary['rmse_mean'], 4)} |",
        f"| RMSE, 95th percentile | {_fmt(fidelity_summary['rmse_p95'], 4)} |",
        f"| MAE, mean | {_fmt(fidelity_summary['mae_mean'], 4)} |",
        f"| NRMSE (% of full intensity range), mean | {_fmt(fidelity_summary['nrmse_pct_of_range_mean'], 2)} |",
        f"| Cosine similarity, mean | {_fmt(fidelity_summary['cosine_similarity_mean'], 4)} |",
        f"| Cosine similarity, minimum | {_fmt(fidelity_summary['cosine_similarity_min'], 4)} |",
        "",
        "RMSE/MAE are expressed in normalised intensity units; NRMSE is RMSE divided by the "
        "full intensity range of the spectrum.",
    ]
    (tables_dir / "tableS9_spectral_fidelity_summary.md").write_text("\n".join(lines) + "\n")

    # ---------------- Table S10 ----------------
    lines = [
        "**Table S10.** Number of tasks whose prediction changes for a single digitized spectrum "
        f"(out of {per_sample_summary['n_tasks']}).",
        "",
        "| Flipped tasks per spectrum | Spectra (n) | Spectra (%) |",
        "|---|---|---|",
    ]
    for row in flip_dist.iter_rows(named=True):
        lines.append(f"| {row['n_flipped_tasks']} | {row['n_spectra']} | {_fmt(row['pct_spectra'], 1)} |")
    lines += [
        "",
        f"Spectra with no changed decision: {_fmt(per_sample_summary['pct_spectra_zero_flips'], 1)}%; "
        f"at most one changed decision: {_fmt(per_sample_summary['pct_spectra_le1_flip'], 1)}%; "
        f"mean changed decisions per spectrum: {_fmt(per_sample_summary['mean_flipped_tasks_per_spectrum'], 3)}.",
    ]
    (tables_dir / "tableS10_flip_count_distribution.md").write_text("\n".join(lines) + "\n")

    # ---------------- Verification ----------------
    lines = [
        "**Table S11.** Reproduction check: overall metrics recomputed in this analysis vs. the "
        "metrics stored in `results/raw_1000_test_9_1/summary.csv` and "
        "`results/digital_1000_test_9_1/summary.csv`.",
        "",
        "| Model | Metrics compared | Max absolute difference |",
        "|---|---|---|",
    ]
    for row in verification_df.iter_rows(named=True):
        lines.append(
            f"| {row['model']} | {row.get('n_metrics_compared', '—')} | {_fmt(row.get('max_abs_diff_overall'), 12)} |"
        )
    lines += ["", "A maximum absolute difference of 0 confirms that the paired analysis uses exactly the same "
                  "predictions as the tables already reported."]
    (tables_dir / "tableS11_verification.md").write_text("\n".join(lines) + "\n")


def make_figures(per_task_df, group_rows, fidelity_df, flip_counts, delta_dec, tables_dir, figures_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200})

    # Fig S1: per-task agreement (x-axis zoomed to 95-100%, otherwise the differences are invisible)
    df = per_task_df.sort(["group", "agreement_pct"])
    colors = {"binary": "#2b6cb0", "multiclass": "#c05621"}
    fig, ax = plt.subplots(figsize=(7.6, 7.8))
    ypos = np.arange(df.height)
    ax.barh(ypos, df["agreement_pct"].to_list(), color=[colors[g] for g in df["group"].to_list()], height=0.72)
    for y, (agree, n_flip) in enumerate(zip(df["agreement_pct"].to_list(), df["disagreement_n"].to_list())):
        ax.text(agree + 0.06, y, f"{n_flip}", va="center", fontsize=7, color="#2d3748")
    overall = next(r for r in group_rows if r["group"] == "overall")
    ax.axvline(overall["agreement_pct"], color="k", ls="--", lw=1,
               label=f"pooled overall = {overall['agreement_pct']:.2f}%")
    ax.set_yticks(ypos, df["task"].to_list(), fontsize=8)
    ax.set_xlim(95.0, 100.55)
    ax.set_xlabel("raw vs digitized prediction agreement (%), zoomed to 95–100%")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[g]) for g in ("binary", "multiclass")]
    ax.legend(handles + [plt.Line2D([0], [0], color="k", ls="--", lw=1)],
              ["binary tasks (21)", "multiplicity tasks (12)",
               f"pooled overall = {overall['agreement_pct']:.2f}%"],
              loc="lower left", fontsize=8)
    ax.set_title("Per-task prediction agreement between raw and digitized spectra\n"
                 "(numbers next to bars: changed decisions out of 1,000)")
    fig.tight_layout()
    fig.savefig(figures_dir / "figS1_per_task_agreement.png")
    plt.close(fig)

    # Fig S2: per-task |Δp| (decision probability) vs agreement
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for group in ("binary", "multiclass"):
        sub = per_task_df.filter(pl.col("group") == group)
        ax.scatter(sub["mad_decision_prob_pp"], sub["agreement_pct"], s=26,
                   color=colors[group], label=f"{group} tasks", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("mean |Δp| of the raw decision (percentage points)")
    ax.set_ylabel("prediction agreement (%)")
    ax.legend(fontsize=8)
    ax.set_title("Agreement vs. probability shift")
    fig.tight_layout()
    fig.savefig(figures_dir / "figS2_agreement_vs_delta_prob.png")
    plt.close(fig)

    # Fig S3: flip count distribution
    vals, counts = np.unique(flip_counts, return_counts=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(vals, counts / counts.sum() * 100, color="#4a5568")
    ax.set_xlabel("number of flipped tasks per spectrum (out of 33)")
    ax.set_ylabel("spectra (%)")
    ax.set_title("How many decisions change for a single digitized spectrum")
    fig.tight_layout()
    fig.savefig(figures_dir / "figS3_flip_count_distribution.png")
    plt.close(fig)

    # Fig S4: flip direction (most correctness transitions are "both correct/both wrong"; a stacked plot hides the differences)
    groups = [r for r in group_rows if r["group"] in ("binary", "multiclass")]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    ax = axes[0]
    x = np.arange(len(groups))
    w = 0.36
    w2r = [g["flip_wrong_to_right_n"] for g in groups]
    r2w = [g["flip_right_to_wrong_n"] for g in groups]
    ax.bar(x - w / 2, w2r, w, label="wrong → right (recovered)", color="#2f855a")
    ax.bar(x + w / 2, r2w, w, label="right → wrong (broken)", color="#c53030")
    for xi, (a, b) in enumerate(zip(w2r, r2w)):
        ax.text(xi - w / 2, a + 1, str(a), ha="center", fontsize=8)
        ax.text(xi + w / 2, b + 1, str(b), ha="center", fontsize=8)
    ax.set_xticks(x, [f"{g['group']}\n({g['flips_n']} changed decisions)" for g in groups])
    ax.set_ylabel("changed decisions (n)")
    ax.set_ylim(0, max(w2r + r2w) * 1.25)
    ax.legend(fontsize=8)
    ax.set_title("Direction of the changed decisions")

    ax = axes[1]
    binary_df = per_task_df.filter(pl.col("group") == "binary")
    lost = int(binary_df["flip_pos_to_neg_n"].sum())
    gained = int(binary_df["flip_neg_to_pos_n"].sum())
    ax.bar(["presence lost\n(pred 1 → 0)", "presence gained\n(pred 0 → 1)"], [lost, gained],
           color=["#c53030", "#2f855a"], width=0.55)
    for xi, value in enumerate([lost, gained]):
        ax.text(xi, value + 1, str(value), ha="center", fontsize=9)
    ax.set_ylabel("changed decisions (n)")
    ax.set_ylim(0, max(lost, gained) * 1.25)
    ax.set_title("Binary tasks: functional-group presence")
    fig.suptitle("Effect of digitization on the final prediction", fontsize=10)
    fig.tight_layout()
    fig.savefig(figures_dir / "figS4_correctness_transitions.png")
    plt.close(fig)

    # Fig S6: per-decision |dp| distribution
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), sharey=True)
    bins = np.arange(0, 21, 0.5)
    for ax, group in zip(axes, ("binary", "multiclass")):
        ax.hist(np.clip(delta_dec[group], 0, 20), bins=bins, color="#2b6cb0" if group == "binary" else "#c05621")
        ax.set_yscale("log")
        ax.set_xlabel("|Δp| of the raw decision (percentage points)")
        ax.set_title(f"{group} ({len(delta_dec[group]):,} decisions)")
    axes[0].set_ylabel("decisions (log scale)")
    fig.suptitle("Absolute change in the predicted probability (clipped at 20 pp)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figures_dir / "figS6_decision_probability_change.png")
    plt.close(fig)

    # Fig S5: spectral fidelity R2 vs flips
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(fidelity_df["r2"], fidelity_df["n_flipped_tasks"], s=14, alpha=0.5, color="#2b6cb0")
    ax.set_xlabel("per-spectrum R² (raw vs digitized)")
    ax.set_ylabel("flipped tasks per spectrum")
    ax.set_title("Digitization fidelity vs. number of changed decisions")
    fig.tight_layout()
    fig.savefig(figures_dir / "figS5_fidelity_vs_flips.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
