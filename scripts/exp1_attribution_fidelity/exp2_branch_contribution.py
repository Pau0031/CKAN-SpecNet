#!/usr/bin/env python
"""exp2 - How much does the Contribution-KAN branch contribute to the decision?

Question 2 of the revision plan:
    to what extent does the Contribution-KAN branch participate in the final
    prediction, and does that participation change when the task moves from
    binary recognition (present / absent) to multiplicity (count = 1 vs >= 2)?

The decomposition is architectural, not statistical -- it is exact:

    full_logit_c = base_logit_c + KAN_logit_c
    m_full = m_base + m_KAN                       (any logit contrast)
    m_KAN  = sum_l dM(l) + alpha*(b_hi - b_lo)    (spatial part + bias part)

so every number below is read off the trained model with no refitting.  The
branch ratio is a descriptive quantity, *not* a probability, and "removing the
KAN branch" is a diagnostic counterfactual of the fused model -- it is not the
separately trained CNN-only ablation model.

Reported for both tasks and both classes:
    branch contribution ratio   |m_KAN| / (|m_base| + |m_KAN|)
    direction agreement         sign(m_KAN) == sign(m_base)
    error correction            P(full correct | base wrong)
    error introduction          P(full wrong  | base correct)
    class dependence            count=1 vs count>=2 (multiplicity),
                               absent vs present  (binary)
    margin AUC                  c2+ vs c1, and present vs absent
plus the spatial / bias split of m_KAN and the agreement between the binary and
the multiplicity spatial evidence maps (same compounds).

Usage
-----
    python exp2_branch_contribution.py --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import common as C  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--folds", type=str, default=None)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=C.RESULT_DIR / "exp2")
    p.add_argument("--skip-enrichment", action="store_true",
                   help="skip the KAN spatial-evidence enrichment part")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def margins(base: np.ndarray, kan: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-fold ``(m_base, m_KAN)`` for the requested contrast."""
    if kind == "multi":
        return base[..., 2] - base[..., 1], kan[..., 2] - kan[..., 1]
    return base[..., 1] - base[..., 0], kan[..., 1] - kan[..., 0]


def describe(name: str, value: float, ci=(-1.0, -1.0), n: int = 0, n_clusters: int = 0) -> dict:
    return {"metric": name, "value": float(value), "ci_lo": float(ci[0]),
            "ci_hi": float(ci[1]), "n": int(n), "n_clusters": int(n_clusters)}


def analyse_task(data, exp0, group: str, kind: str, task: str, strata: dict,
                 population: np.ndarray, boot: int, seed: int) -> tuple[list[dict], pl.DataFrame]:
    """All branch-contribution numbers for one (group, task, task-type)."""
    base = exp0[f"{task}__base"]              # (F, N, K)
    kan = exp0[f"{task}__kan"]
    full = exp0[f"{task}__full"]
    bias = exp0[f"{task}__bias"]
    alpha = exp0[f"alpha__{task}"]            # (F,)

    m_base_f, m_kan_f = margins(base, kan, kind)
    m_full_f = m_base_f + m_kan_f

    m_base, m_kan = m_base_f.mean(axis=0), m_kan_f.mean(axis=0)
    m_full = m_full_f.mean(axis=0)

    if kind == "multi":
        hi, lo = 2, 1
    else:
        hi, lo = 1, 0
    bias_part_f = alpha[:, None] * (bias[..., hi] - bias[..., lo])
    bias_part = bias_part_f.mean(axis=0)
    spatial_part = m_kan - bias_part

    pred_base = base.mean(axis=0).argmax(axis=1)
    pred_kan = kan.mean(axis=0).argmax(axis=1)
    pred_full = full.mean(axis=0).argmax(axis=1)
    y = data.labels[task]

    ok_base = pred_base == y
    ok_kan = pred_kan == y
    ok_full = pred_full == y

    ratio = np.abs(m_kan) / (np.abs(m_base) + np.abs(m_kan) + 1e-9)
    agree = np.sign(m_kan) == np.sign(m_base)
    changed = pred_full != pred_base

    per_sample = pl.DataFrame({
        "group": group, "task": task, "task_type": kind,
        "row_index": np.arange(len(y)),
        "compound_key": data.compounds.astype(str),
        "true": y,
        "count": data.frame[C.GROUPS[group]["column"]].to_numpy(),
        "m_base": m_base, "m_kan": m_kan, "m_full": m_full,
        "m_kan_spatial": spatial_part, "m_kan_bias": bias_part,
        "branch_ratio": ratio,
        "sign_agree": agree,
        "pred_base": pred_base, "pred_kan": pred_kan, "pred_full": pred_full,
        "ok_base": ok_base, "ok_kan": ok_kan, "ok_full": ok_full,
        "pred_changed": changed,
    })

    rows = []
    for stratum_name, mask in strata.items():
        if mask.sum() < 5:
            continue
        codes = C.cluster_index(data.compounds[mask])[0]
        b = boot

        def add(metric, value, ci, n=int(mask.sum())):
            rows.append({"group": group, "task": task, "task_type": kind,
                         "stratum": stratum_name, **describe(metric, value, ci, n, len(set(codes)))})

        add("n_samples", float(mask.sum()), (-1, -1))
        add("acc_full", float(ok_full[mask].mean()),
            _ci(C.bootstrap_rate, ok_full[mask], codes, b, seed))
        add("acc_base", float(ok_base[mask].mean()),
            _ci(C.bootstrap_rate, ok_base[mask], codes, b, seed))
        add("acc_kan_alone", float(ok_kan[mask].mean()),
            _ci(C.bootstrap_rate, ok_kan[mask], codes, b, seed))
        add("pred_changed_rate", float(changed[mask].mean()),
            _ci(C.bootstrap_rate, changed[mask], codes, b, seed))

        st = C.bootstrap_mean(ratio[mask], codes, b, seed)
        add("branch_ratio_mean", st["mean"], (st["ci_lo"], st["ci_hi"]))
        add("branch_ratio_median", float(np.median(ratio[mask])), _median_ci(ratio[mask], codes, b, seed))
        st = C.bootstrap_rate(agree[mask], codes, b, seed)
        add("sign_agreement", st["mean"], (st["ci_lo"], st["ci_hi"]))

        base_wrong = ~ok_base[mask]
        base_right = ok_base[mask]
        if base_wrong.sum() >= 5:
            st = C.bootstrap_rate(ok_full[mask][base_wrong], codes[base_wrong], b, seed)
            add("error_correction_rate", st["mean"], (st["ci_lo"], st["ci_hi"]),
                n=int(base_wrong.sum()))
            add("n_base_wrong", float(base_wrong.sum()), (-1, -1), n=int(base_wrong.sum()))
        if base_right.sum() >= 5:
            st = C.bootstrap_rate(~ok_full[mask][base_right], codes[base_right], b, seed)
            add("error_introduction_rate", st["mean"], (st["ci_lo"], st["ci_hi"]),
                n=int(base_right.sum()))
            add("n_base_right", float(base_right.sum()), (-1, -1), n=int(base_right.sum()))

        st = C.bootstrap_mean(spatial_part[mask], codes, b, seed)
        add("m_kan_spatial_mean", st["mean"], (st["ci_lo"], st["ci_hi"]))
        st = C.bootstrap_mean(bias_part[mask], codes, b, seed)
        add("m_kan_bias_mean", st["mean"], (st["ci_lo"], st["ci_hi"]))
        share = np.abs(bias_part) / (np.abs(m_kan) + 1e-9)
        add("bias_share_of_m_kan_median", float(np.median(share[mask])),
            _median_ci(share[mask], codes, b, seed))

        add("corr_m_base_m_kan_pearson",
            float(np.corrcoef(m_base[mask], m_kan[mask])[0, 1]) if mask.sum() > 2 else float("nan"),
            (-1, -1))

    # AUCs on the task population: count>=2 vs count=1 for the multiplicity task,
    # present vs absent for the binary task
    sub = np.flatnonzero(population)
    codes_sub = C.cluster_index(data.compounds[sub])[0]
    label = (y[sub] == (2 if kind == "multi" else 1)).astype(int)
    for name, score in (("auc_full", m_full[sub]), ("auc_base", m_base[sub]),
                        ("auc_kan", m_kan[sub])):
        st = C.bootstrap_stat(lambda idx, s=score: C.auc_score(label[idx], s[idx]),
                              len(sub), codes_sub, max(200, boot // 4), seed)
        rows.append({"group": group, "task": task, "task_type": kind, "stratum": "population",
                     **describe(name, st["value"], (st["ci_lo"], st["ci_hi"]), len(sub),
                                st["n_clusters"])})
        rows[-1]["n_positive"] = int(label.sum())

    return rows, per_sample


def _ci(fn, values, codes, boot, seed) -> tuple[float, float]:
    st = fn(values, codes, boot, seed)
    return (st["ci_lo"], st["ci_hi"])


def _median_ci(values: np.ndarray, codes: np.ndarray, boot: int, seed: int) -> tuple[float, float]:
    """Cluster bootstrap CI of the median."""
    n_clusters = codes.max() + 1
    members = [np.flatnonzero(codes == c) for c in range(n_clusters)]
    rng = np.random.default_rng(seed)
    out = np.empty(boot, dtype=np.float64)
    for b in range(boot):
        idx = np.concatenate([members[c] for c in rng.integers(0, n_clusters, size=n_clusters)])
        out[b] = np.median(values[idx])
    return (float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975)))


# --------------------------------------------------------------------------- #
def spatial_enrichment(data, ens, group: str, idx: np.ndarray, task_bin: str,
                       task_multi: str, boot: int, seed: int) -> tuple[pl.DataFrame, dict]:
    """KAN spatial evidence inside the literature bands: binary vs multiplicity.

    Same compounds, same 30 cm^-1 windows, two different questions:
        binary       evidence supporting "present" (M_1 - M_0)
        multiplicity evidence supporting "count>=2" (s * [M_2 - M_1])
    """
    X = ens.preprocess(data.raw[idx], fold=0)
    counts = data.frame[C.GROUPS[group]["column"]].to_numpy()[idx]
    sign = C.support_sign(counts)
    win = C.make_windows(30.0)
    in_band = C.characteristic_windows(group, win)
    coverage = C.characteristic_coverage(group)

    contrib_bin = C.fold_contrib(ens, X, task_bin).mean(axis=0)      # (n, 2, L)
    map_bin = contrib_bin[:, 1, :] - contrib_bin[:, 0, :]
    contrib_multi = C.fold_contrib(ens, X, task_multi).mean(axis=0)  # (n, 3, L)
    map_multi = sign[:, None] * (contrib_multi[:, 2, :] - contrib_multi[:, 1, :])

    win_bin = np.abs(C.window_sum(map_bin, win))
    win_multi = np.abs(C.window_sum(map_multi, win))
    codes = C.cluster_index(data.compounds[idx])[0]
    rows = []
    for label, win_mass in (("binary_present_vs_absent", win_bin),
                            ("multiplicity_c2plus_vs_c1", win_multi)):
        share = win_mass[:, in_band].sum(axis=1) / np.maximum(win_mass.sum(axis=1), 1e-12)
        st = C.bootstrap_mean(share, codes, boot, seed)
        rows.append({"group": group, "metric": "evidence_share_in_chem_bands",
                     "view": label, "value": st["mean"], "ci_lo": st["ci_lo"],
                     "ci_hi": st["ci_hi"], "enrichment_factor": st["mean"] * 100.0 / coverage,
                     "chem_coverage_pct": coverage, "n": st["n"]})

    rho = C.spearman_rows(C.window_sum(map_bin, win), C.window_sum(map_multi, win))
    st = C.bootstrap_mean(rho, codes, boot, seed)
    rows.append({"group": group, "metric": "window_rank_correlation_binary_vs_multi",
                 "view": "both", "value": st["mean"], "ci_lo": st["ci_lo"],
                 "ci_hi": st["ci_hi"], "enrichment_factor": float("nan"),
                 "chem_coverage_pct": coverage, "n": st["n"]})

    top_multi = np.argmax(win_multi, axis=1)
    top_bin = np.argmax(win_bin, axis=1)

    # identity: the spatial part recovered from contrib must equal m_KAN - alpha*b
    # (both terms are averaged over the five folds *after* the product, since
    #  alpha and b differ per fold)
    exp0 = np.load(C.RESULT_DIR / "exp0" / "fold_logits.npz")
    kan_task = exp0[f"{task_multi}__kan"][:, idx]
    bias_task = exp0[f"{task_multi}__bias"][:, idx]
    alpha = exp0[f"alpha__{task_multi}"]
    m_kan = sign * (kan_task[..., 2] - kan_task[..., 1]).mean(axis=0)
    bias_part = (alpha[:, None] * (bias_task[..., 2] - bias_task[..., 1])).mean(axis=0)
    # map_multi already carries the support sign, so: sum_l map = m_KAN - s*alpha*db
    spatial_err = float(np.abs(map_multi.sum(axis=1) - (m_kan - sign * bias_part)).max())

    summary = {
        "chem_coverage_pct": coverage,
        "n_in_band_windows": int(in_band.sum()),
        "n_windows": int(win.n),
        "multi_top_window_center_median_cm": float(np.median(win.centers[top_multi])),
        "multi_top_window_in_band_frac": float(in_band[top_multi].mean()),
        "binary_top_window_center_median_cm": float(np.median(win.centers[top_bin])),
        "binary_top_window_in_band_frac": float(in_band[top_bin].mean()),
        "window_rank_corr_mean": float(rho.mean()),
        "spatial_vs_m_kan_minus_bias_max_abs_err": spatial_err,
    }
    return pl.DataFrame(rows), summary


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    started = time.time()
    out_dir = C.ensure_dir(args.out_dir)
    folds = [int(f) for f in args.folds.split(",")] if args.folds else None

    data = C.load_analysis_data()
    exp0 = dict(np.load(C.RESULT_DIR / "exp0" / "fold_logits.npz"))

    compounds = data.compounds.astype(str)
    all_rows, all_samples = [], []
    enrichment_frames, enrichment_summary = [], {}
    ens = None

    for group, spec in C.GROUPS.items():
        counts = data.frame[spec["column"]].to_numpy()
        present = counts >= 1
        multi_idx = np.flatnonzero(present)
        c2plus = counts >= 2

        # ---- multiplicity task (count = 1 vs count >= 2) ------------------- #
        strata_multi = {
            "all_present": present,
            "c1": counts == 1,
            "c2plus": c2plus,
        }
        rows, samples = analyse_task(data, exp0, group, "multi", spec["multi"],
                                     strata_multi, present, args.boot, args.seed)
        all_rows += rows
        all_samples.append(samples)

        # ---- binary task (present vs absent) ----------------------------- #
        strata_bin = {
            "all": np.ones(len(counts), dtype=bool),
            "absent": counts == 0,
            "present": present,
        }
        rows, samples = analyse_task(data, exp0, group, "binary", spec["binary"],
                                     strata_bin, np.ones(len(counts), dtype=bool),
                                     args.boot, args.seed)
        all_rows += rows
        all_samples.append(samples)

        # ---- KAN spatial evidence: binary vs multiplicity ------------------ #
        if not args.skip_enrichment:
            if ens is None:
                ens = C.load_ensemble(C.parse_device(args.device))
            frame, summary = spatial_enrichment(data, ens, group, multi_idx,
                                                spec["binary"], spec["multi"],
                                                args.boot, args.seed)
            enrichment_frames.append(frame)
            enrichment_summary[group] = summary

    metrics = pl.DataFrame(all_rows).sort(["group", "task_type", "stratum", "metric"])
    metrics.write_csv(out_dir / "exp2_metrics.csv")
    pl.concat(all_samples).write_parquet(out_dir / "exp2_per_sample.parquet")

    if enrichment_frames:
        pl.concat(enrichment_frames).write_csv(out_dir / "exp2_enrichment.csv")

    # ---- per-fold robustness --------------------------------------------- #
    fold_rows = []
    for group, spec in C.GROUPS.items():
        counts = data.frame[spec["column"]].to_numpy()
        for task, kind, mask in ((spec["multi"], "multi", counts >= 1),
                                 (spec["binary"], "binary", np.ones(len(counts), dtype=bool))):
            base, kan = exp0[f"{task}__base"], exp0[f"{task}__kan"]
            mb, mk = margins(base, kan, kind)
            for f in range(mb.shape[0]):
                pb = base[f].argmax(axis=1) == data.labels[task]
                pf = (base[f] + kan[f]).argmax(axis=1) == data.labels[task]
                fold_rows.append({
                    "group": group, "task": task, "task_type": kind, "fold": f + 1,
                    "n": int(mask.sum()),
                    "acc_base": float(pb[mask].mean()),
                    "acc_full": float(pf[mask].mean()),
                    "branch_ratio_median": float(np.median(
                        np.abs(mk[f][mask]) / (np.abs(mb[f][mask]) + np.abs(mk[f][mask]) + 1e-9))),
                    "sign_agreement": float((np.sign(mk[f][mask]) == np.sign(mb[f][mask])).mean()),
                    "alpha": float(exp0[f"alpha__{task}"][f]),
                })
    pl.DataFrame(fold_rows).write_csv(out_dir / "exp2_per_fold.csv")

    # ---- identity check --------------------------------------------------- #
    id_rows = []
    for group, spec in C.GROUPS.items():
        for task in (spec["binary"], spec["multi"]):
            base, kan, full = exp0[f"{task}__base"], exp0[f"{task}__kan"], exp0[f"{task}__full"]
            kind = "multi" if task.endswith("3class") else "binary"
            mb, mk = margins(base, kan, kind)
            id_rows.append({
                "group": group, "task": task,
                "check": "full_logit == base_logit + kan_logit",
                "max_abs_err": float(np.abs(base + kan - full).max()),
            })
            id_rows.append({
                "group": group, "task": task,
                "check": "m_full == m_base + m_KAN",
                "max_abs_err": float(np.abs(mb + mk - (margins(full, np.zeros_like(full), kind)[0])).max()),
            })
    pl.DataFrame(id_rows).write_csv(out_dir / "exp2_identity_check.csv")

    C.write_json(out_dir / "exp2_meta.json", {
        "folds": folds or "all",
        "boot": args.boot, "seed": args.seed,
        "enrichment": enrichment_summary,
        "runtime_s": round(time.time() - started, 1),
    })

    print("\n[exp2] branch contribution summary (median branch ratio / sign agreement / "
          "correction / introduction)")
    show = metrics.filter(pl.col("metric").is_in(
        ["branch_ratio_median", "sign_agreement", "error_correction_rate",
         "error_introduction_rate", "acc_base", "acc_full", "auc_full", "auc_base", "auc_kan"]))
    print(show.select(["group", "task_type", "stratum", "metric", "value", "ci_lo", "ci_hi", "n"]))
    print(f"\n[out] {out_dir}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
