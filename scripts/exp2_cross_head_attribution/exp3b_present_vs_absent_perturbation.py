#!/usr/bin/env python
"""exp3b - the same perturbation, now with the count = 0 spectra as an explicit stratum.

Motivation
----------
exp1 / exp1b / exp2 / exp3 all restricted the sample space to count >= 1, because
multiplicity (1 vs >= 2) is only defined there.  The teacher's follow-up is to put
the count = 0 spectra back in and ask what the *same* perturbation does to
"absent vs present".

For exp3 this is natural: its primary read-out is the margin of the **independent
binary head** (`nitro` / `nitriles`, trained on present/absent only), which is
defined for every spectrum.  Adding count = 0 turns that head into a real
present-vs-absent detector and gives the perturbation experiment a **negative
control**: if the literature band really carries the functional-group signal,
removing it must hurt the present spectra far more than the absent ones.

What changes with respect to exp3
---------------------------------
1. **Sample space**  all 7066 held-out spectra: nitro 390 present + 6676 absent,
   nitriles 288 present + 6778 absent.  The class ratio is ~1:17, so results are
   always reported **per stratum**; a pooled mean over all spectra would simply
   restate the absent stratum.
2. **Support alignment**  the binary read-out needs a sign now:
   ``s_bin = +1`` for count >= 1, ``s_bin = -1`` for count = 0, so
   ``m_bin = s_bin * (o_1 - o_0)`` and ``m_pres = s_bin * (logsumexp(o_1,o_2) - o_0)``
   are "support for the true presence state".  The binary-head KAN map and the
   binary-head Grad-CAM target use the same sign.
3. **The count margin `m_cnt` stays a count >= 1 quantity** (there is no count
   direction for an absent spectrum); `D_cnt` and the `KAN (count head)`
   selection are therefore restricted to the present stratum, while every other
   selection and both presence read-outs are evaluated on all spectra.
4. **Flip analysis**  present stratum: present -> absent; absent stratum:
   absent -> present.
5. The present-versus-absent contrast is bootstrapped as a **stratified**
   difference (clusters resampled within the whole set, strata kept apart).

Usage
-----
    python exp3b_present_vs_absent_perturbation.py --group nitro    --device cuda:0
    python exp3b_present_vs_absent_perturbation.py --group nitriles --device cuda:1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ``common.py`` (ensemble loading, analysis windows, the two perturbation
# operators, the cluster bootstrap) lives in the exp1 folder and is shared here.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "exp1_attribution_fidelity"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import common as C  # noqa: E402

METHODS = ("kan_bin", "kan_cnt", "cam_bin", "peak", "chem", "random")
BLOCK_KEYS = ["width_cm", "coverage_pct", "k_windows", "frac_covered_pct", "perturb"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--group", type=str, required=True, choices=list(C.GROUPS))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--folds", type=str, default=None)
    p.add_argument("--widths", type=str, default="30")
    p.add_argument("--coverages", type=str, default="1,2,4,6,10")
    p.add_argument("--perturbs", type=str, default="interp,baseline")
    p.add_argument("--rand-reps", type=int, default=20)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=C.RESULT_DIR / "exp3b")
    p.add_argument("--limit", type=int, default=None, help="debug: first N samples")
    return p.parse_args()


def selection_for(method: str, *, maps: dict, win: C.Windows, k: int, n: int,
                  intensity: np.ndarray, in_band: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    if method in maps:
        return C.select_windows(maps[method], win, k)
    if method == "peak":
        return C.select_windows(intensity, win, k)
    if method == "chem":
        return C.select_chemistry(n, win, k, intensity, in_band)
    if method == "random":
        return C.select_random(n, win, k, rng)
    raise ValueError(method)


def bootstrap_stratified(a: np.ndarray, codes_a: np.ndarray,
                        b: np.ndarray, codes_b: np.ndarray,
                        n_boot: int = 2000, seed: int = 0) -> dict:
    """Cluster bootstrap of ``mean(a) - mean(b)`` on two independent strata.

    ``codes_a`` / ``codes_b`` must come from the same cluster labelling (two
    disjoint subsets of it); every replicate resamples the union of clusters and
    keeps the two strata apart, so the dependence structure of each stratum is
    preserved.
    """
    n_clusters = max(int(codes_a.max()), int(codes_b.max())) + 1
    sum_a = np.bincount(codes_a, weights=a, minlength=n_clusters)
    cnt_a = np.bincount(codes_a, minlength=n_clusters).astype(np.float64)
    sum_b = np.bincount(codes_b, weights=b, minlength=n_clusters)
    cnt_b = np.bincount(codes_b, minlength=n_clusters).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a = sum_a[idx].sum(axis=1) / cnt_a[idx].sum(axis=1)
        mean_b = sum_b[idx].sum(axis=1) / cnt_b[idx].sum(axis=1)
    diff = mean_a - mean_b
    diff = diff[np.isfinite(diff)]
    p_left = float((diff <= 0).mean())
    p_right = float((diff >= 0).mean())
    return {
        "diff": float(a.mean() - b.mean()),
        "ci_lo": float(np.quantile(diff, 0.025)),
        "ci_hi": float(np.quantile(diff, 0.975)),
        "p_two_sided": float(min(1.0, 2.0 * min(p_left, p_right))),
        "n_boot_valid": int(diff.size),
    }


def main() -> None:  # noqa: C901
    args = parse_args()
    started = time.time()
    device = C.parse_device(args.device)
    folds = [int(f) for f in args.folds.split(",")] if args.folds else None
    widths = [float(w) for w in args.widths.split(",")]
    coverages = [float(c) for c in args.coverages.split(",")]
    perturbs = [p for p in args.perturbs.split(",") if p]

    group = args.group
    spec = C.GROUPS[group]
    task_bin, task_multi = spec["binary"], spec["multi"]
    out_dir = C.ensure_dir(args.out_dir)

    ens = C.load_ensemble(device, folds=folds)
    data = C.load_analysis_data()

    idx = np.arange(len(data))                      # every held-out spectrum
    if args.limit:
        idx = idx[: args.limit]
    n = len(idx)

    counts = data.frame[spec["column"]].to_numpy()[idx]
    y_bin = data.labels[task_bin][idx]
    y_multi = data.labels[task_multi][idx]
    raw = data.raw[idx]
    compounds = data.compounds[idx]

    present = counts >= 1
    absent = ~present
    sign_bin = np.where(present, 1.0, -1.0).astype(np.float32)     # support for presence
    sign_cnt = C.support_sign(counts)                              # only used on present
    print(f"[info] group={group} n={n}  present={int(present.sum())} "
          f"(c1 {(counts == 1).sum()}, c2+ {(counts >= 2).sum()})  absent={int(absent.sum())}")
    print(f"[info] clusters: present {len(set(compounds[present]))}, "
          f"absent {len(set(compounds[absent]))}")

    X = ens.preprocess(raw, fold=0)

    # ---------------- unperturbed state ------------------------------------ #
    exp0 = np.load(C.RESULT_DIR / "exp0" / "fold_logits.npz")
    bin_full = exp0[f"{task_bin}__full"][:, idx, :]
    mul_full = exp0[f"{task_multi}__full"][:, idx, :]
    if folds:
        bin_full, mul_full = bin_full[folds], mul_full[folds]

    def presence_readouts(bin_folds: np.ndarray, mul_folds: np.ndarray) -> dict:
        out = {
            "bin_fold": sign_bin[None, :] * C.binary_margin(bin_folds),
            "pres3_fold": sign_bin[None, :] * C.presence_margin(mul_folds),
            "cnt_fold": sign_cnt[None, :] * C.multi_margin(mul_folds),
        }
        for key in ("bin", "pres3", "cnt"):
            out[key] = out[f"{key}_fold"].mean(axis=0)
        return out

    ref = presence_readouts(bin_full, mul_full)
    pred_bin_before = bin_full.mean(axis=0).argmax(axis=1)
    pred_mul_before = mul_full.mean(axis=0).argmax(axis=1)
    print(f"[info] unperturbed m_bin: present {ref['bin'][present].mean():.2f} / "
          f"absent {ref['bin'][absent].mean():.2f} | m_pres3: present "
          f"{ref['pres3'][present].mean():.2f} / absent {ref['pres3'][absent].mean():.2f}")
    print(f"[info] binary-head acc {float((pred_bin_before == y_bin).mean()):.4f} "
          f"(present {float((pred_bin_before[present] == y_bin[present]).mean()):.4f}, "
          f"absent {float((pred_bin_before[absent] == y_bin[absent]).mean()):.4f})")

    # ---------------- attribution maps ------------------------------------- #
    t0 = time.time()
    contrib_bin = C.fold_contrib(ens, X, task_bin)
    if folds:
        contrib_bin = contrib_bin[folds]
    cm = contrib_bin.mean(axis=0)
    kan_bin_map = (sign_bin[:, None] * (cm[:, 1, :] - cm[:, 0, :])).astype(np.float32)
    del contrib_bin, cm

    contrib_mul = C.fold_contrib(ens, X, task_multi)
    if folds:
        contrib_mul = contrib_mul[folds]
    cm = contrib_mul.mean(axis=0)
    kan_cnt_map = (sign_cnt[:, None] * (cm[:, 2, :] - cm[:, 1, :])).astype(np.float32)
    del contrib_mul, cm

    cam_bin, collapsed = C.gradcam_maps(ens, X, task_bin, hi=1, lo=0, sign=sign_bin,
                                        method="gradcam++")
    print(f"[attr] maps ready in {time.time() - t0:.1f}s  collapsed={float(collapsed.mean()):.3f}")

    intensity = np.abs(X)
    maps = {"kan_bin": kan_bin_map, "kan_cnt": kan_cnt_map, "cam_bin": cam_bin}
    np.savez_compressed(out_dir / f"exp3b_{group}_maps.npz",
                        kan_bin=kan_bin_map.astype(np.float16),
                        kan_cnt=kan_cnt_map.astype(np.float16),
                        cam_bin=cam_bin.astype(np.float32),
                        intensity=intensity.astype(np.float16), X=X.astype(np.float16))

    # KAN evidence inside the literature bands, per stratum
    diag_rows = []
    for width in widths:
        win = C.make_windows(width)
        in_band = C.characteristic_windows(group, win)
        for name, m in (("kan_bin", "kan_bin"), ("kan_cnt", "kan_cnt"), ("cam_bin", "cam_bin")):
            for stratum, mask in (("present", present), ("absent", absent)):
                diag_rows.append({"group": group, "width_cm": width,
                                  "metric": f"mass_in_chem_bands_{name}",
                                  "stratum": stratum,
                                  "value": float(C.mass_in_bands(maps[m][mask], win, in_band)),
                                  "n": int(mask.sum())})
        for stratum, mask in (("present", present), ("absent", absent)):
            rho = C.spearman_rows(C.window_sum(maps["kan_bin"][mask], win),
                                  C.window_sum(maps["kan_cnt"][mask], win))
            diag_rows.append({"group": group, "width_cm": width,
                              "metric": "spearman_kan_bin_vs_kan_cnt", "stratum": stratum,
                              "value": float(rho.mean()), "n": int(mask.sum())})
    pl.DataFrame(diag_rows).write_csv(out_dir / f"exp3b_{group}_map_diagnostics.csv")

    # ---------------- sweep ------------------------------------------------ #
    cond_rows: list[dict] = []
    rec = {k: [] for k in ("D_bin", "D_pres3", "D_cnt")}
    rec_fold = {k: [] for k in ("D_bin", "D_pres3")}
    rec_pred = {"bin": [], "multi": []}
    rng = np.random.default_rng(args.seed + 31337 * list(C.GROUPS).index(group))

    for width in widths:
        win = C.make_windows(width)
        in_band = C.characteristic_windows(group, win)
        intensity_win = C.window_sum(intensity, win)
        for cov in coverages:
            k = C.coverage_to_k(win, cov)
            frac = C.actual_fraction(win, k)
            kan_ref = C.select_windows(maps["kan_bin"], win, k)
            for method in METHODS:
                n_rep = args.rand_reps if method == "random" else 1
                for rep in range(n_rep):
                    sel = selection_for(method, maps=maps, win=win, k=k, n=n,
                                        intensity=intensity, in_band=in_band, rng=rng)
                    masks = C.masks_from_selection(sel, win)
                    jac = np.empty(n)
                    for i in range(n):
                        a = set(np.atleast_1d(sel[i]).tolist())
                        b = set(np.atleast_1d(kan_ref[i]).tolist())
                        jac[i] = len(a & b) / max(1, len(a | b))
                    diag = {
                        "sel_inband_frac": float(in_band[sel].mean()),
                        "sel_intensity": float(np.take_along_axis(intensity_win, sel, axis=1).mean()),
                        "sel_jaccard_kan_bin": float(jac.mean()),
                    }
                    for perturb in perturbs:
                        Xp = C.perturb(X, masks, kind=perturb, renormalize=False)
                        logits = C.fold_logits_tasks(ens, Xp, [task_bin, task_multi])
                        bin_folds, mul_folds = logits[task_bin], logits[task_multi]
                        if folds:
                            bin_folds, mul_folds = bin_folds[folds], mul_folds[folds]
                        after = presence_readouts(bin_folds, mul_folds)
                        for key in ("bin", "pres3", "cnt"):
                            rec[f"D_{key}"].append((ref[key] - after[key]).astype(np.float32))
                        for key in ("bin", "pres3"):
                            rec_fold[f"D_{key}"].append(
                                (ref[f"{key}_fold"] - after[f"{key}_fold"]).astype(np.float16))
                        rec_pred["bin"].append(bin_folds.mean(axis=0).argmax(axis=1).astype(np.int8))
                        rec_pred["multi"].append(mul_folds.mean(axis=0).argmax(axis=1).astype(np.int8))
                        cond_rows.append({"cond_id": len(cond_rows), "method": method,
                                          "repeat": rep, "width_cm": width,
                                          "coverage_pct": cov, "k_windows": k,
                                          "frac_covered_pct": frac, "perturb": perturb, **diag})
            print(f"[sweep] width={width:>4.0f} coverage={cov:>4.1f}% conds={len(cond_rows):<5d} "
                  f"({time.time() - started:.0f}s)", flush=True)

    cond = pl.DataFrame(cond_rows)
    D = {k: np.stack(v) for k, v in rec.items()}
    D_fold = {k: np.stack(v) for k, v in rec_fold.items()}
    pred_after = {k: np.stack(v) for k, v in rec_pred.items()}
    print(f"[sweep] {len(cond)} conditions, {time.time() - started:.0f}s", flush=True)

    # ---------------- flip rates ------------------------------------------- #
    flip_bin = pred_after["bin"] != pred_bin_before[None, :]
    flip_mul = pred_after["multi"] != pred_mul_before[None, :]
    flip_rows = []
    for i, row in enumerate(cond.iter_rows(named=True)):
        meta = {k: row[k] for k in ("method", "repeat", *BLOCK_KEYS)}
        flip_rows.append({**meta, "stratum": "present", "readout": "bin_flip",
                          "rate": float(flip_bin[i][present].mean())})
        flip_rows.append({**meta, "stratum": "present", "readout": "multi_flip_to_absent",
                          "rate": float((flip_mul[i][present] & (pred_after["multi"][i][present] == 0)).mean())})
        flip_rows.append({**meta, "stratum": "absent", "readout": "bin_flip",
                          "rate": float(flip_bin[i][absent].mean())})
    flips = pl.DataFrame(flip_rows)
    flips.write_csv(out_dir / f"exp3b_{group}_flips.csv")

    # ---------------- blocks ----------------------------------------------- #
    blocks: dict[tuple, dict] = {}
    for i, row in enumerate(cond.iter_rows(named=True)):
        key = tuple(row[k] for k in BLOCK_KEYS)
        block = blocks.setdefault(key, {"meta": dict(zip(BLOCK_KEYS, key)), "methods": {}})
        block["methods"].setdefault(row["method"], []).append(i)

    # ---------------- bootstrap, per stratum -------------------------------- #
    present_codes = C.cluster_index(compounds[present])[0]
    absent_codes = C.cluster_index(compounds[absent])[0]
    print(f"[stats] bootstrapping {len(blocks)} blocks x {args.boot} reps ...", flush=True)
    summary_rows, contrast_rows = [], []
    for block in blocks.values():
        meta = block["meta"]
        for method, members in block["methods"].items():
            for key in ("bin", "pres3", "cnt"):
                d_all = D[f"D_{key}"][members]
                strata = [("present", present), ("absent", absent)]
                if key == "cnt":
                    strata = [("present", present)]        # count margin needs count >= 1
                for stratum, mask in strata:
                    vals = d_all[:, mask].mean(axis=0)
                    codes = present_codes if stratum == "present" else absent_codes
                    st = C.bootstrap_mean(vals, codes, args.boot, args.seed)
                    summary_rows.append({
                        "group": group, "method": method, "readout": key,
                        "stratum": stratum, **meta, "n_repeats": len(members),
                        "value": st["mean"], "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"],
                        "n": st["n"], "n_clusters": st["n_clusters"]})
                if key in ("bin", "pres3"):
                    a = d_all[:, present].mean(axis=0)
                    b = d_all[:, absent].mean(axis=0)
                    test = bootstrap_stratified(a, present_codes, b, absent_codes,
                                                args.boot, args.seed)
                    contrast_rows.append({"group": group, "method": method, "readout": key,
                                          **meta, "n_repeats": len(members), **test})
    summary = pl.DataFrame(summary_rows)
    summary.write_csv(out_dir / f"exp3b_{group}_summary.csv")
    pl.DataFrame(contrast_rows).write_csv(out_dir / f"exp3b_{group}_contrast.csv")

    # ---------------- curves / AUC ------------------------------------------ #
    curve_rows, auc_rows = [], []
    for (method, key, stratum, width, perturb), points in _curves(summary).items():
        xs = np.array(sorted(points))
        ys = np.array([points[x] for x in xs])
        for x, y in zip(xs, ys):
            curve_rows.append({"group": group, "method": method, "readout": key,
                               "stratum": stratum, "width_cm": width, "perturb": perturb,
                               "coverage_pct": float(x), "D_mean": float(y)})
        auc_rows.append({"group": group, "method": method, "readout": key,
                         "stratum": stratum, "width_cm": width, "perturb": perturb,
                         "D_auc": float(ys.mean())})
    pl.DataFrame(curve_rows).write_csv(out_dir / f"exp3b_{group}_curves.csv")
    pl.DataFrame(auc_rows).sort(["perturb", "readout", "stratum", "method"]).write_csv(
        out_dir / f"exp3b_{group}_auc.csv")

    # ---------------- per fold --------------------------------------------- #
    fold_rows = []
    for block in blocks.values():
        meta = block["meta"]
        for method, members in block["methods"].items():
            for key in ("bin", "pres3"):
                d_fold = D_fold[f"D_{key}"][members].astype(np.float32).mean(axis=0)
                for f in range(d_fold.shape[0]):
                    for stratum, mask in (("present", present), ("absent", absent)):
                        fold_rows.append({"group": group, "method": method, "readout": key,
                                          "stratum": stratum, **meta, "fold": f + 1,
                                          "D_mean": float(d_fold[f][mask].mean())})
    pl.DataFrame(fold_rows).write_csv(out_dir / f"exp3b_{group}_perfold.csv")

    # ---------------- persist ---------------------------------------------- #
    np.savez_compressed(out_dir / f"exp3b_{group}_per_sample.npz",
                        D_bin=D["D_bin"].astype(np.float16),
                        D_pres3=D["D_pres3"].astype(np.float16),
                        D_cnt=D["D_cnt"].astype(np.float16),
                        pred_after_bin=pred_after["bin"],
                        pred_after_multi=pred_after["multi"],
                        sample_idx=idx)
    cond.write_parquet(out_dir / f"exp3b_{group}_conditions.parquet")
    pl.DataFrame({
        "sample_idx": idx, "compound_key": compounds, "count": counts,
        "class": y_multi, "bin_label": y_bin, "present": present,
        "pred_bin": pred_bin_before, "pred_multi": pred_mul_before,
        "sign_bin": sign_bin, "m_bin": ref["bin"], "m_pres3": ref["pres3"], "m_cnt": ref["cnt"],
        "source_name": data.frame["source_name"].to_numpy()[idx],
    }).write_parquet(out_dir / f"exp3b_{group}_samples.parquet")

    C.write_json(out_dir / f"exp3b_{group}_meta.json", {
        "group": group, "task_binary": task_bin, "task_multi": task_multi,
        "n": int(n), "n_present": int(present.sum()), "n_absent": int(absent.sum()),
        "n_c1": int((counts == 1).sum()), "n_c2plus": int((counts >= 2).sum()),
        "n_clusters_present": int(len(set(compounds[present]))),
        "n_clusters_absent": int(len(set(compounds[absent]))),
        "folds": ens.fold_files, "widths_cm": widths, "coverages_pct": coverages,
        "methods": list(METHODS), "perturbs": perturbs, "boot": args.boot,
        "n_conditions": int(len(cond)),
        "chem_bands": C.FUNCTIONAL_GROUP_REGIONS[group],
        "chem_band_coverage_pct": C.characteristic_coverage(group),
        "unperturbed": {
            "m_bin_present": float(ref["bin"][present].mean()),
            "m_bin_absent": float(ref["bin"][absent].mean()),
            "m_pres3_present": float(ref["pres3"][present].mean()),
            "m_pres3_absent": float(ref["pres3"][absent].mean()),
            "m_cnt_present": float(ref["cnt"][present].mean()),
            "acc_binary_head": float((pred_bin_before == y_bin).mean()),
            "acc_binary_head_present": float((pred_bin_before[present] == y_bin[present]).mean()),
            "acc_binary_head_absent": float((pred_bin_before[absent] == y_bin[absent]).mean()),
        },
        "runtime_s": round(time.time() - started, 1),
    })

    sel = summary.filter((pl.col("width_cm") == 30.0) & (pl.col("perturb") == "interp")
                         & (pl.col("coverage_pct") == 4.0))
    print("\n[exp3b done] width 30 / interp / 4 % — per stratum")
    print(sel.select(["method", "readout", "stratum", "value", "ci_lo", "ci_hi"])
          .sort(["readout", "stratum", "method"]))
    print(f"\n[out] {out_dir}  ({time.time() - started:.0f}s)")


def _curves(summary: pl.DataFrame) -> dict:
    out: dict[tuple, dict] = {}
    for row in summary.iter_rows(named=True):
        key = (row["method"], row["readout"], row["stratum"], row["width_cm"], row["perturb"])
        out.setdefault(key, {})[row["coverage_pct"]] = row["value"]
    return out


if __name__ == "__main__":
    main()
