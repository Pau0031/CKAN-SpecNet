#!/usr/bin/env python
"""exp1 - Spectral attribution fidelity for nitro / nitriles multiplicity.

Question 1 of the revision plan:
    do the wavenumber regions pointed at by Contribution-KAN actually drive the
    final fused decision, when the model has to separate count = 1 from
    count >= 2?

Design (fixed before the experiment, no hyper-parameter is re-selected)
----------------------------------------------------------------------
sample space   held-out ``main_test`` spectra with count >= 1 (multiplicity is
               only defined there).  *Every* count>=2 spectrum and *every*
               count=1 spectrum is used, so no "cherry-picked correct examples".
selection      windows of 30 cm^-1 (sensitivity 20 / 40 cm^-1) tiling the axis;
               coverage levels 1 / 2 / 4 / 6 / 10 % -> k = 1,2,4,7,11 windows.
methods        Contribution-KAN : s * [M_2(l) - M_1(l)],  s = +1 for count>=2,
                                  -1 for count=1  (support-aligned evidence)
               Grad-CAM++/CAM   : CAM of the fused margin score s*(o_2 - o_1)
               peak intensity   : mean |x(l)| inside the window
               chem. bands      : literature regions (ckan_specnet.regions) first
               random           : random windows, repeated
perturbation   A) local smooth interpolation between the band edges
               B) local baseline replacement estimated from both flanks
               Both act on the *preprocessed* spectrum and the whole
               CKAN-SpecNet is re-run afterwards (CNN features, dynamic KAN
               coefficients, final logits) -- never a shortcut on the stored
               contributions.
metric         D(k) = m_y(x) - m_y(x_perturbed),  m_y = s*(o_2 - o_1)
               plus the 3-class true-class margin, the true-class probability,
               and the flip rate of correctly classified samples.
inference      the five folds are all evaluated; the reported prediction is the
               ensemble mean (as submitted), CIs come from a bootstrap that
               resamples *compounds* (SMILES), never the five fold predictions
               of one spectrum.

Usage
-----
    python exp1_attribution_fidelity.py --group nitro    --device cuda:0
    python exp1_attribution_fidelity.py --group nitriles --device cuda:1
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

METHODS = ("kan", "gradcampp", "peak", "chem", "random")
BLOCK_KEYS = ["width_cm", "coverage_pct", "k_windows", "frac_covered_pct", "perturb", "renorm"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--group", type=str, required=True, choices=list(C.GROUPS))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--folds", type=str, default=None, help="comma separated fold ids (debug)")
    p.add_argument("--widths", type=str, default="30,20,40", help="window widths in cm^-1")
    p.add_argument("--coverages", type=str, default="1,2,4,6,10")
    p.add_argument("--methods", type=str, default=",".join(METHODS))
    p.add_argument("--perturbs", type=str, default="interp,baseline")
    p.add_argument("--rand-reps", type=int, default=20, help="random-region repeats, 30 cm^-1")
    p.add_argument("--rand-reps-sens", type=int, default=10, help="random repeats, 20/40 cm^-1")
    p.add_argument("--renorm-check", action="store_true",
                   help="also run the re-normalised variant (width 30 only)")
    p.add_argument("--boot", type=int, default=2000, help="bootstrap replicates")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=C.RESULT_DIR / "exp1")
    p.add_argument("--limit", type=int, default=None, help="debug: use only the first N samples")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def selection_for(method: str, *, maps: dict, win: C.Windows, k: int, n: int,
                  intensity: np.ndarray, in_band: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Per-sample window indices for one selection method."""
    if method in maps:
        return C.select_windows(maps[method], win, k)
    if method == "peak":
        return C.select_windows(intensity, win, k)
    if method == "chem":
        return C.select_chemistry(n, win, k, intensity, in_band)
    if method == "random":
        return C.select_random(n, win, k, rng)
    raise ValueError(method)


def selection_diagnostics(sel: np.ndarray, in_band: np.ndarray,
                          intensity_win: np.ndarray, kan_ref: np.ndarray) -> dict:
    """Chemistry overlap / band strength / agreement-with-KAN of one selection."""
    inband = in_band[sel].mean(axis=1)
    strength = np.take_along_axis(intensity_win, sel, axis=1).mean(axis=1)
    jaccard = np.empty(len(sel), dtype=np.float64)
    for i in range(len(sel)):
        a, b = set(np.atleast_1d(sel[i]).tolist()), set(kan_ref[i].tolist())
        jaccard[i] = len(a & b) / max(1, len(a | b))
    return {
        "sel_inband_frac": float(inband.mean()),
        "sel_intensity": float(strength.mean()),
        "sel_jaccard_kan": float(jaccard.mean()),
    }


def group_conditions(cond: pl.DataFrame) -> dict:
    """Group sweep rows by (method-free) window/perturbation block."""
    blocks: dict[tuple, dict] = {}
    for i, row in enumerate(cond.iter_rows(named=True)):
        key = tuple(row[k] for k in BLOCK_KEYS)
        block = blocks.setdefault(key, {"meta": dict(zip(BLOCK_KEYS, key)), "methods": {}})
        block["methods"].setdefault(row["method"], []).append(i)
    return blocks


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    started = time.time()
    device = C.parse_device(args.device)
    folds = [int(f) for f in args.folds.split(",")] if args.folds else None
    widths = [float(w) for w in args.widths.split(",")]
    coverages = [float(c) for c in args.coverages.split(",")]
    methods = [m for m in args.methods.split(",") if m]
    perturbs = [p for p in args.perturbs.split(",") if p]

    group = args.group
    spec = C.GROUPS[group]
    task = spec["multi"]
    out_dir = C.ensure_dir(args.out_dir)

    ens = C.load_ensemble(device, folds=folds)
    data = C.load_analysis_data()

    idx = C.group_samples(data, group, min_count=1)
    if args.limit:
        idx = idx[: args.limit]
    n = len(idx)

    counts = data.frame[spec["column"]].to_numpy()[idx]
    y_multi = data.labels[task][idx]
    raw = data.raw[idx]
    compounds = data.compounds[idx]
    sign = C.support_sign(counts)

    print(f"[info] group={group} n={n}  count=1: {(counts == 1).sum()}  "
          f"count>=2: {(counts >= 2).sum()}  compounds: {len(set(compounds))}")
    print(f"[info] folds={ens.fold_files} widths={widths} coverages={coverages}")

    X = ens.preprocess(raw, fold=0)

    # ---------------- unperturbed reference state -------------------------- #
    exp0 = np.load(C.RESULT_DIR / "exp0" / "fold_logits.npz")
    full_folds = exp0[f"{task}__full"][:, idx, :]           # (F, n, 3)
    base_folds = exp0[f"{task}__base"][:, idx, :]
    if folds:
        full_folds, base_folds = full_folds[folds], base_folds[folds]

    m_before_fold = sign[None, :] * (full_folds[..., 2] - full_folds[..., 1])   # (F, n)
    m_before = m_before_fold.mean(axis=0)
    full_before = full_folds.mean(axis=0)
    base_before = base_folds.mean(axis=0)
    pred_before = full_before.argmax(axis=1)
    correct = pred_before == y_multi
    prob_before = _softmax(full_before)
    p_true_before = prob_before[np.arange(n), y_multi]
    mtrue_before = C.true_class_margin(full_before, y_multi)
    pred_base = base_before.argmax(axis=1)

    print(f"[info] unperturbed: acc(full)={correct.mean():.4f} acc(base)="
          f"{(pred_base == y_multi).mean():.4f} mean m_y={m_before.mean():.3f} "
          f"(c1 {m_before[counts == 1].mean():.3f} / c2+ {m_before[counts >= 2].mean():.3f})")

    # ---------------- attribution maps ------------------------------------- #
    t0 = time.time()
    contrib = C.fold_contrib(ens, X, task)                          # (F, n, 3, L)
    if folds:
        contrib = contrib[folds]
    contrib_mean = contrib.mean(axis=0)
    kan_map = (sign[:, None] * (contrib_mean[:, 2, :] - contrib_mean[:, 1, :])).astype(np.float32)
    del contrib, contrib_mean

    cam_pp, collapsed_pp = C.gradcam_maps(ens, X, task, hi=2, lo=1, sign=sign, method="gradcam++")
    zero_pp = float(collapsed_pp.mean())
    print(f"[attr] maps ready in {time.time() - t0:.1f}s  "
          f"collapsed-map rate gradcam={zero_pp:.3f}")
    intensity = np.abs(X)

    maps = {"kan": kan_map, "gradcampp": cam_pp}
    collapsed = {"gradcampp": collapsed_pp}
    if "gradcam" in methods:            # only computed when explicitly requested
        cam_std, collapsed_std = C.gradcam_maps(ens, X, task, hi=2, lo=1, sign=sign,
                                                method="gradcam")
        maps["gradcam"] = cam_std
        collapsed["gradcam"] = collapsed_std
    np.savez_compressed(out_dir / f"exp1_{group}_maps.npz",
                        kan=kan_map.astype(np.float16),
                        gradcampp=cam_pp.astype(np.float16),
                        intensity=intensity.astype(np.float16),
                        X=X.astype(np.float16))

    # ---------------- how different are the methods' rankings? -------------- #
    corr_rows = []
    for width in widths:
        win = C.make_windows(width)
        in_band_w = C.characteristic_windows(group, win)
        scores = {
            "kan": C.window_sum(kan_map, win),
            "gradcampp": C.window_sum(cam_pp, win),
            "peak": C.window_sum(intensity, win),
        }
        if "gradcam" in maps:
            scores["gradcam"] = C.window_sum(maps["gradcam"], win)
        for label, a, b in (("kan_vs_peak", "kan", "peak"),
                            ("kan_vs_gradcampp", "kan", "gradcampp"),
                            ("gradcampp_vs_peak", "gradcampp", "peak")):
            rho = C.spearman_rows(scores[a], scores[b])
            corr_rows.append({"group": group, "width_cm": width, "comparison": label,
                              "spearman_mean": float(rho.mean()),
                              "spearman_median": float(np.median(rho)),
                              "n": int(len(rho))})
        k = C.coverage_to_k(win, 4.0)
        sel_kan = C.select_windows(kan_map, win, k)
        for method in ("gradcampp", "peak"):
            sel_other = (C.select_windows(maps[method], win, k) if method in maps
                         else C.select_windows(intensity, win, k))
            corr_rows.append({"group": group, "width_cm": width,
                              "comparison": f"kan_vs_{method}_topk_jaccard@4pct",
                              "spearman_mean": C.topk_overlap(sel_kan, sel_other),
                              "spearman_median": float("nan"), "n": int(n)})
        for method in ("kan", "gradcampp", "peak"):
            ref_map = maps[method] if method in maps else intensity
            corr_rows.append({"group": group, "width_cm": width,
                              "comparison": f"{method}_mass_in_chem_bands",
                              "spearman_mean": float(C.mass_in_bands(ref_map, win, in_band_w)),
                              "spearman_median": float("nan"), "n": int(n)})
    pl.DataFrame(corr_rows).write_csv(out_dir / f"exp1_{group}_map_diagnostics.csv")

    # ---------------- condition sweep -------------------------------------- #
    cond_rows: list[dict] = []
    rec_D, rec_D_fold, rec_m_after, rec_pred, rec_d_true, rec_d_prob = [], [], [], [], [], []
    rng = np.random.default_rng(args.seed + 1000 * list(C.GROUPS).index(group))

    for width in widths:
        win = C.make_windows(width)
        in_band = C.characteristic_windows(group, win)
        intensity_win = C.window_sum(intensity, win)
        reps = args.rand_reps if abs(width - 30.0) < 1e-6 else args.rand_reps_sens
        renorm_variants = ([False, True]
                           if (args.renorm_check and abs(width - 30.0) < 1e-6) else [False])
        kan_ref = C.select_windows(maps["kan"], win, 1)             # overwritten per k below

        for cov in coverages:
            k = C.coverage_to_k(win, cov)
            frac = C.actual_fraction(win, k)
            kan_ref = C.select_windows(maps["kan"], win, k)

            for method in methods:
                n_rep = reps if method == "random" else 1
                for rep in range(n_rep):
                    sel = selection_for(method, maps=maps, win=win, k=k, n=n,
                                        intensity=intensity, in_band=in_band, rng=rng)
                    masks = C.masks_from_selection(sel, win)
                    diag = selection_diagnostics(sel, in_band, intensity_win, kan_ref)

                    for perturb in perturbs:
                        for renorm in renorm_variants:
                            Xp = C.perturb(X, masks, kind=perturb, renormalize=renorm)
                            logits_folds = C.fold_logits(ens, Xp, task)
                            if folds:
                                logits_folds = logits_folds[folds]
                            full_after = logits_folds.mean(axis=0)
                            m_after_fold = sign[None, :] * (logits_folds[..., 2] - logits_folds[..., 1])
                            m_after = m_after_fold.mean(axis=0)

                            pred_after = full_after.argmax(axis=1)
                            prob_after = _softmax(full_after)

                            rec_D.append((m_before - m_after).astype(np.float32))
                            rec_D_fold.append((m_before_fold - m_after_fold).astype(np.float16))
                            rec_m_after.append(m_after.astype(np.float32))
                            rec_pred.append(pred_after.astype(np.int8))
                            rec_d_true.append((mtrue_before
                                               - C.true_class_margin(full_after, y_multi)).astype(np.float32))
                            rec_d_prob.append((p_true_before
                                               - prob_after[np.arange(n), y_multi]).astype(np.float32))
                            cond_rows.append({
                                "cond_id": len(cond_rows),
                                "method": method,
                                "repeat": rep,
                                "width_cm": width,
                                "coverage_pct": cov,
                                "k_windows": k,
                                "frac_covered_pct": frac,
                                "perturb": perturb,
                                "renorm": renorm,
                                **diag,
                            })

            print(f"[sweep] width={width:>4.0f} coverage={cov:>4.1f}% k={k:<3d} "
                  f"conds={len(cond_rows):<5d} ({time.time() - started:.0f}s)")

    cond = pl.DataFrame(cond_rows)
    D = np.stack(rec_D)
    D_fold = np.stack(rec_D_fold)          # (n_cond, F, n)
    m_after = np.stack(rec_m_after)
    pred_after = np.stack(rec_pred)
    d_true = np.stack(rec_d_true)
    d_prob = np.stack(rec_d_prob)
    print(f"[sweep] {len(cond)} conditions, D matrix {D.shape}, "
          f"{time.time() - started:.0f}s")

    blocks = group_conditions(cond)

    # ---------------- random-region spread --------------------------------- #
    spread_rows = []
    for meta, method_blocks in ((b["meta"], b["methods"]) for b in blocks.values()):
        members = method_blocks.get("random", [])
        if len(members) < 2:
            continue
        means = [float(D[m].mean()) for m in members]
        spread_rows.append({**meta, "n_reps": len(members),
                            "D_mean_std_over_reps": float(np.std(means)),
                            "D_mean_min": float(np.min(means)),
                            "D_mean_max": float(np.max(means))})
    if spread_rows:
        pl.DataFrame(spread_rows).write_csv(out_dir / f"exp1_{group}_random_spread.csv")

    # ---------------- bootstrap aggregation -------------------------------- #
    subsets = {"all": np.ones(n, dtype=bool), "correct": correct, "incorrect": ~correct}
    summary_rows, paired_rows = [], []
    curves: dict[tuple, dict] = {}

    print(f"[stats] bootstrapping {len(blocks)} blocks x {args.boot} replicates ...")
    for block in blocks.values():
        meta = block["meta"]
        for method, members in block["methods"].items():
            method_subsets = dict(subsets)
            # a collapsed CAM carries no attribution: report those samples apart
            if method in collapsed and collapsed[method].any() and (~collapsed[method]).sum() > 5:
                method_subsets["non_degenerate"] = ~collapsed[method]
            for subset_name, subset in method_subsets.items():
                if subset.sum() == 0:
                    continue
                codes_sub = C.cluster_index(compounds[subset])[0]
                d_mean = D[members][:, subset].mean(axis=0)
                base = {"group": group, "subset": subset_name, "method": method, **meta,
                        "n_repeats": len(members)}

                st = C.bootstrap_mean(d_mean, codes_sub, args.boot, args.seed)
                summary_rows.append({**base, "metric": "D_mean", "value": st["mean"],
                                     "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"], "n": st["n"],
                                     "statistic": "mean"})
                summary_rows.append({**base, "metric": "D_median", "value": float(np.median(d_mean)),
                                     "ci_lo": float("nan"), "ci_hi": float("nan"),
                                     "n": int(subset.sum()), "statistic": "median"})
                curves.setdefault((subset_name, method, meta["width_cm"], meta["perturb"],
                                   meta["renorm"]), {})[meta["coverage_pct"]] = st["mean"]

                flips = (pred_after[members][:, subset] != pred_before[subset][None, :]).mean(axis=0)
                st_flip = C.bootstrap_rate(flips, codes_sub, args.boot, args.seed)
                summary_rows.append({**base, "metric": "flip_rate", "value": st_flip["mean"],
                                     "ci_lo": st_flip["ci_lo"], "ci_hi": st_flip["ci_hi"],
                                     "n": st_flip["n"], "statistic": "rate"})
                flips_count = ((pred_after[members][:, subset] != pred_before[subset][None, :])
                               & (pred_after[members][:, subset] != 0)).mean(axis=0)
                st_count = C.bootstrap_rate(flips_count, codes_sub, args.boot, args.seed)
                summary_rows.append({**base, "metric": "flip_to_other_count_rate",
                                     "value": st_count["mean"], "ci_lo": st_count["ci_lo"],
                                     "ci_hi": st_count["ci_hi"], "n": st_count["n"],
                                     "statistic": "rate"})
                for metric, values in (("d_true_class_margin", d_true),
                                       ("d_true_class_prob", d_prob)):
                    vals = values[members][:, subset].mean(axis=0)
                    st_m = C.bootstrap_mean(vals, codes_sub, args.boot, args.seed)
                    summary_rows.append({**base, "metric": metric, "value": st_m["mean"],
                                         "ci_lo": st_m["ci_lo"], "ci_hi": st_m["ci_hi"],
                                         "n": st_m["n"], "statistic": "mean"})

    summary = pl.DataFrame(summary_rows)
    summary.write_csv(out_dir / f"exp1_{group}_summary.csv")

    # ---------------- paired tests (KAN vs every other method) -------------- #
    for block in blocks.values():
        if "kan" not in block["methods"]:
            continue
        meta = block["meta"]
        d_kan = D[block["methods"]["kan"]].mean(axis=0)
        for method, members in block["methods"].items():
            if method == "kan":
                continue
            d_other = D[members].mean(axis=0)
            for subset_name, subset in (("all", np.ones(n, dtype=bool)), ("correct", correct)):
                if subset.sum() < 3:
                    continue
                codes_sub = C.cluster_index(compounds[subset])[0]
                test = C.bootstrap_paired(d_kan[subset], d_other[subset], codes_sub,
                                          args.boot, args.seed)
                paired_rows.append({"group": group, "subset": subset_name, **meta,
                                    "method": method, "n_repeats": len(members), **test})
    pl.DataFrame(paired_rows).write_csv(out_dir / f"exp1_{group}_paired_tests.csv")

    # ---------------- curve-level AUC -------------------------------------- #
    auc_rows = []
    for (subset_name, method, width, perturb, renorm), points in curves.items():
        xs = np.array(sorted(points))
        ys = np.array([points[x] for x in xs])
        auc_rows.append({
            "group": group, "subset": subset_name, "method": method, "width_cm": width,
            "perturb": perturb, "renorm": renorm,
            "D_auc": float(ys.mean()),
            "D_at_max_coverage": float(ys[-1]),
            "coverage_max_pct": float(xs[-1]),
            "n_points": int(len(xs)),
        })
    auc = pl.DataFrame(auc_rows)
    auc.sort(["width_cm", "perturb", "subset", "method"]).write_csv(
        out_dir / f"exp1_{group}_auc.csv")

    # ---------------- per-fold robustness ---------------------------------- #
    fold_rows = []
    for block in blocks.values():
        meta = block["meta"]
        for method, members in block["methods"].items():
            d_fold = D_fold[members].astype(np.float32).mean(axis=0)      # (F, n)
            for f in range(d_fold.shape[0]):
                fold_rows.append({"group": group, "method": method, **meta, "fold": f + 1,
                                  "D_mean": float(d_fold[f].mean())})
    pl.DataFrame(fold_rows).write_csv(out_dir / f"exp1_{group}_perfold.csv")

    # ---------------- task-level stats (Table X) --------------------------- #
    stat_rows = []
    full_task = exp0[f"{task}__full"].mean(axis=0)
    base_task = exp0[f"{task}__base"].mean(axis=0)
    for scope, mask in (("main_test_all", np.ones(len(full_task), dtype=bool)),
                        ("analysis_set", np.isin(np.arange(len(full_task)), idx))):
        y = data.labels[task][mask]
        pred = full_task[mask].argmax(axis=1)
        base_pred = base_task[mask].argmax(axis=1)
        f1 = {}
        for c in (1, 2):
            tp = int(((pred == c) & (y == c)).sum())
            fp = int(((pred == c) & (y != c)).sum())
            fn = int(((pred != c) & (y == c)).sum())
            f1[c] = 2 * tp / max(1, 2 * tp + fp + fn)
        stat_rows.append({
            "group": group, "task": task, "scope": scope,
            "n": int(mask.sum()), "n_c1": int((y == 1).sum()), "n_c2plus": int((y == 2).sum()),
            "n_compounds": int(len(set(data.compounds[mask]))),
            "acc_full": float((pred == y).mean()),
            "acc_base": float((base_pred == y).mean()),
            "f1_c1": f1[1], "f1_c2plus": f1[2], "macro_f1_c1_c2": float(np.mean([f1[1], f1[2]])),
            "auc_count_margin_full": C.auc_score((y == 2).astype(int),
                                                 full_task[mask][:, 2] - full_task[mask][:, 1]),
            "auc_count_margin_base": C.auc_score((y == 2).astype(int),
                                                 base_task[mask][:, 2] - base_task[mask][:, 1]),
        })
    pl.DataFrame(stat_rows).write_csv(out_dir / f"exp1_{group}_task_stats.csv")

    # ---------------- persist --------------------------------------------- #
    np.savez_compressed(
        out_dir / f"exp1_{group}_per_sample.npz",
        D=D.astype(np.float16),
        D_fold=D_fold,
        m_after=m_after.astype(np.float16),
        pred_after=pred_after,
        d_true_margin=d_true.astype(np.float16),
        d_true_prob=d_prob.astype(np.float16),
        sample_idx=idx,
    )
    cond.write_parquet(out_dir / f"exp1_{group}_conditions.parquet")
    pl.DataFrame({
        "sample_idx": idx,
        "compound_key": compounds,
        "count": counts,
        "class": y_multi,
        "pred_before": pred_before,
        "pred_before_base": pred_base,
        "correct": correct,
        "m_before": m_before,
        "mtrue_before": mtrue_before,
        "p_true_before": p_true_before,
        "source_name": data.frame["source_name"].to_numpy()[idx],
    }).write_parquet(out_dir / f"exp1_{group}_samples.parquet")

    C.write_json(out_dir / f"exp1_{group}_meta.json", {
        "group": group, "task": task, "n": int(n),
        "n_c1": int((counts == 1).sum()), "n_c2plus": int((counts >= 2).sum()),
        "n_compounds": int(len(set(compounds))),
        "folds": ens.fold_files, "widths_cm": widths, "coverages_pct": coverages,
        "methods": methods, "perturbs": perturbs,
        "rand_reps": {"30": args.rand_reps, "other": args.rand_reps_sens},
        "renorm_check": bool(args.renorm_check),
        "boot": args.boot, "seed": args.seed,
        "zero_map_rate": {"gradcam": zero_pp},
        "n_conditions": int(len(cond)),
        "chem_bands": C.FUNCTIONAL_GROUP_REGIONS[group],
        "chem_band_coverage_pct": C.characteristic_coverage(group),
        "unperturbed": {
            "acc_full": float(correct.mean()),
            "acc_base": float((pred_base == y_multi).mean()),
            "mean_m_true": float(m_before.mean()),
            "mean_m_true_c1": float(m_before[counts == 1].mean()),
            "mean_m_true_c2plus": float(m_before[counts >= 2].mean()),
        },
        "runtime_s": round(time.time() - started, 1),
    })

    print("\n[exp1 done] primary curve (width 30, interp, all samples)")
    print(summary.filter((pl.col("width_cm") == 30.0) & (pl.col("perturb") == "interp")
                         & (pl.col("renorm") == False)  # noqa: E712
                         & (pl.col("subset") == "all")
                         & (pl.col("metric") == "D_mean"))
          .select(["method", "coverage_pct", "value", "ci_lo", "ci_hi"])
          .sort("coverage_pct", "method"))
    print(f"\n[out] {out_dir}  ({time.time() - started:.0f}s)")


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()
