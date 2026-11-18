#!/usr/bin/env python
"""exp3 - Independent binary head vs multiplicity head under the *same* perturbation.

Motivation (teacher's item 3, "独立 binary head 的谱学扰动对比")
-------------------------------------------------------------
The manuscript's central claim is a transition  recognition -> multiplicity :
the model can tell *whether* a functional group is present from its
characteristic IR band, but the count (1 vs >= 2) is not recoverable from that
band.  exp1b showed this with the *three-class* head, using
``logsumexp(o1, o2) - o0`` as a presence proxy.  A referee can object that the
logsumexp of a head trained on three classes is not an independent
"recognition" measurement.

CKAN-SpecNet is a multi-task network: every functional group has **two
independent output heads**, trained on two independent label sets

    binary head        ``nitro`` / ``nitriles``           present (1) vs absent (0)
    multiplicity head  ``nitro_3class`` / ``nitriles_3class`` 0 / 1 / >=2

so the independent binary head can be used as the recognition witness **without
retraining and without touching a single weight** -- and it is the five-fold
average, exactly the released model.

Design
------
sample space  same held-out count>=1 spectra as exp1 (nitro 390 / nitriles 288)
selection     30 cm^-1 windows (sensitivity 20/40 cm^-1), coverage 1/2/4/6/10 %
              -> k = 1,2,4,7,11 windows
              kan_bin : KAN evidence of the *binary* head, M_1(l) - M_0(l)
                        (every sample is present, so no support sign is needed)
              kan_cnt : KAN evidence of the multiplicity head, the exp1 map
                        s * (M_2(l) - M_1(l)), s = +1 for count>=2 / -1 for count=1
              cam_bin : Grad-CAM of the binary head margin o_1 - o_0
              peak    : mean |x| inside the window
              chem    : literature bands first (peak-intensity fill)
              random  : random windows, 20 repeats
perturbation  operator A (local interpolation) and B (lower-envelope baseline
              replacement), applied to the preprocessed spectrum; the *whole*
              network is re-run afterwards
read-out      one forward pass gives **three margins at the same time**:
              m_bin   = o_1 - o_0                       (independent binary head)
              m_pres3 = logsumexp(o_1,o_2) - o_0        (multiplicity head, presence)
              m_cnt   = s * (o_2 - o_1)                 (multiplicity head, count)
              D = m(x) - m(x_pert);  plus the present -> absent flip rate of the
              binary head and the two flip modes of the three-class head.
inference     compound-clustered bootstrap (2000 reps), paired tests both
              *within* a condition (D_bin vs D_cnt, same perturbation) and
              *between* selections (kan_bin vs every other method on D_bin).

Everything is a diagnostic of the released five-fold model in ``models/``;
nothing is trained.

Usage
-----
    python exp3_binary_head_perturbation.py --group nitro    --device cuda:0
    python exp3_binary_head_perturbation.py --group nitriles --device cuda:1
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
READOUTS = ("bin", "pres3", "cnt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--group", type=str, required=True, choices=list(C.GROUPS))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--folds", type=str, default=None, help="comma separated fold ids (debug)")
    p.add_argument("--widths", type=str, default="30,20,40")
    p.add_argument("--coverages", type=str, default="1,2,4,6,10")
    p.add_argument("--perturbs", type=str, default="interp,baseline")
    p.add_argument("--rand-reps", type=int, default=20)
    p.add_argument("--rand-reps-sens", type=int, default=10)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=C.RESULT_DIR / "exp3")
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


def main() -> None:  # noqa: C901 - straight-line experiment
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

    idx = C.group_samples(data, group, min_count=1)
    if args.limit:
        idx = idx[: args.limit]
    n = len(idx)

    counts = data.frame[spec["column"]].to_numpy()[idx]
    y_multi = data.labels[task_multi][idx]
    y_bin = data.labels[task_bin][idx]
    raw = data.raw[idx]
    compounds = data.compounds[idx]
    sign = C.support_sign(counts)

    print(f"[info] group={group} n={n}  count=1: {(counts == 1).sum()}  "
          f"count>=2: {(counts >= 2).sum()}  compounds: {len(set(compounds))}")
    print(f"[info] binary head='{task_bin}'  multiplicity head='{task_multi}'  "
          f"folds={ens.fold_files}")

    X = ens.preprocess(raw, fold=0)

    # ---------------- unperturbed state of the three margins ---------------- #
    exp0 = np.load(C.RESULT_DIR / "exp0" / "fold_logits.npz")
    bin_full = exp0[f"{task_bin}__full"][:, idx, :]        # (F, n, 2)
    mul_full = exp0[f"{task_multi}__full"][:, idx, :]      # (F, n, 3)
    mul_base = exp0[f"{task_multi}__base"][:, idx, :]
    if folds:
        bin_full, mul_full, mul_base = bin_full[folds], mul_full[folds], mul_base[folds]

    def margins(bin_folds: np.ndarray, mul_folds: np.ndarray) -> dict[str, np.ndarray]:
        """Per-fold and fold-mean read-outs of the three margins, ``(F,n)`` + ``(n,)``."""
        out = {}
        out["bin_fold"] = C.binary_margin(bin_folds)
        out["pres3_fold"] = C.presence_margin(mul_folds)
        out["cnt_fold"] = sign[None, :] * C.multi_margin(mul_folds)
        for key in ("bin", "pres3", "cnt"):
            out[key] = out[f"{key}_fold"].mean(axis=0)
        return out

    ref = margins(bin_full, mul_full)
    bin_before = bin_full.mean(axis=0)
    mul_before = mul_full.mean(axis=0)
    pred_bin_before = bin_before.argmax(axis=1)
    pred_mul_before = mul_before.argmax(axis=1)
    bin_correct = pred_bin_before == y_bin
    mul_correct = pred_mul_before == y_multi
    print(f"[info] unperturbed m_bin={ref['bin'].mean():.3f} "
          f"m_pres3={ref['pres3'].mean():.3f} m_cnt={ref['cnt'].mean():.3f} | "
          f"acc bin={bin_correct.mean():.4f} multi={mul_correct.mean():.4f}")

    # ---------------- attribution maps ------------------------------------- #
    t0 = time.time()
    contrib_bin = C.fold_contrib(ens, X, task_bin)                 # (F, n, 2, L)
    if folds:
        contrib_bin = contrib_bin[folds]
    kan_bin_map = (contrib_bin.mean(axis=0)[:, 1, :]
                   - contrib_bin.mean(axis=0)[:, 0, :]).astype(np.float32)
    del contrib_bin

    contrib_mul = C.fold_contrib(ens, X, task_multi)               # (F, n, 3, L)
    if folds:
        contrib_mul = contrib_mul[folds]
    contrib_mul = contrib_mul.mean(axis=0)
    kan_cnt_map = (sign[:, None] * (contrib_mul[:, 2, :] - contrib_mul[:, 1, :])).astype(np.float32)
    del contrib_mul

    cam_bin, collapsed_bin = C.gradcam_maps(ens, X, task_bin, hi=1, lo=0,
                                            sign=np.ones(n, dtype=np.float32),
                                            method="gradcam++")
    print(f"[attr] maps ready in {time.time() - t0:.1f}s  "
          f"collapsed-map rate cam_bin={float(collapsed_bin.mean()):.3f}")

    intensity = np.abs(X)
    maps = {"kan_bin": kan_bin_map, "kan_cnt": kan_cnt_map, "cam_bin": cam_bin}
    np.savez_compressed(out_dir / f"exp3_{group}_maps.npz",
                        kan_bin=kan_bin_map.astype(np.float16),
                        kan_cnt=kan_cnt_map.astype(np.float16),
                        cam_bin=cam_bin.astype(np.float32),   # binary CAM can exceed float16
                        intensity=intensity.astype(np.float16),
                        X=X.astype(np.float16))

    # ---------------- do the two KAN maps see the same wavenumbers? --------- #
    diag_rows = []
    for width in widths:
        win = C.make_windows(width)
        in_band = C.characteristic_windows(group, win)
        scores = {m: C.window_sum(maps[m], win) for m in maps}
        scores["peak"] = C.window_sum(intensity, win)
        for a, b in (("kan_bin", "kan_cnt"), ("kan_bin", "peak"),
                     ("kan_cnt", "peak"), ("kan_bin", "cam_bin")):
            rho = C.spearman_rows(scores[a], scores[b])
            diag_rows.append({"group": group, "width_cm": width,
                              "metric": f"spearman_{a}_vs_{b}",
                              "value": float(rho.mean()), "n": n})
        for m in ("kan_bin", "kan_cnt", "cam_bin"):
            diag_rows.append({"group": group, "width_cm": width,
                              "metric": f"mass_in_chem_bands_{m}",
                              "value": float(C.mass_in_bands(maps[m], win, in_band)),
                              "n": n})
        k4 = C.coverage_to_k(win, 4.0)
        sel_bin = C.select_windows(maps["kan_bin"], win, k4)
        sel_cnt = C.select_windows(maps["kan_cnt"], win, k4)
        diag_rows.append({"group": group, "width_cm": width,
                          "metric": "topk_jaccard@4pct_kan_bin_vs_kan_cnt",
                          "value": C.topk_overlap(sel_bin, sel_cnt), "n": n})
    pl.DataFrame(diag_rows).write_csv(out_dir / f"exp3_{group}_map_diagnostics.csv")

    # ---------------- condition sweep -------------------------------------- #
    cond_rows: list[dict] = []
    rec = {key: [] for key in ("D_bin", "D_pres3", "D_cnt")}
    rec_fold = {key: [] for key in ("D_bin", "D_pres3", "D_cnt")}
    rec_pred = {"bin": [], "multi": []}
    rng = np.random.default_rng(args.seed + 7919 * list(C.GROUPS).index(group))

    for width in widths:
        win = C.make_windows(width)
        in_band = C.characteristic_windows(group, win)
        intensity_win = C.window_sum(intensity, win)
        reps = args.rand_reps if abs(width - 30.0) < 1e-6 else args.rand_reps_sens
        kan_bin_ref = C.select_windows(maps["kan_bin"], win, 1)

        for cov in coverages:
            k = C.coverage_to_k(win, cov)
            frac = C.actual_fraction(win, k)
            kan_bin_ref = C.select_windows(maps["kan_bin"], win, k)

            for method in METHODS:
                n_rep = reps if method == "random" else 1
                for rep in range(n_rep):
                    sel = selection_for(method, maps=maps, win=win, k=k, n=n,
                                        intensity=intensity, in_band=in_band, rng=rng)
                    masks = C.masks_from_selection(sel, win)
                    jac = np.empty(n)
                    for i in range(n):
                        a = set(np.atleast_1d(sel[i]).tolist())
                        b = set(np.atleast_1d(kan_bin_ref[i]).tolist())
                        jac[i] = len(a & b) / max(1, len(a | b))
                    diag = {
                        "sel_inband_frac": float(in_band[sel].mean()),
                        "sel_intensity": float(np.take_along_axis(intensity_win, sel, axis=1).mean()),
                        "sel_jaccard_kan_bin": float(jac.mean()),
                    }

                    for perturb in perturbs:
                        Xp = C.perturb(X, masks, kind=perturb, renormalize=False)
                        logits = C.fold_logits_tasks(ens, Xp, [task_bin, task_multi])
                        bin_folds = logits[task_bin]
                        mul_folds = logits[task_multi]
                        if folds:
                            bin_folds, mul_folds = bin_folds[folds], mul_folds[folds]
                        after = margins(bin_folds, mul_folds)

                        for key in ("bin", "pres3", "cnt"):
                            rec[f"D_{key}"].append((ref[key] - after[key]).astype(np.float32))
                            rec_fold[f"D_{key}"].append(
                                (ref[f"{key}_fold"] - after[f"{key}_fold"]).astype(np.float16))
                        rec_pred["bin"].append(bin_folds.mean(axis=0).argmax(axis=1).astype(np.int8))
                        rec_pred["multi"].append(mul_folds.mean(axis=0).argmax(axis=1).astype(np.int8))

                        cond_rows.append({
                            "cond_id": len(cond_rows), "method": method, "repeat": rep,
                            "width_cm": width, "coverage_pct": cov, "k_windows": k,
                            "frac_covered_pct": frac, "perturb": perturb, **diag,
                        })

            print(f"[sweep] width={width:>4.0f} coverage={cov:>4.1f}% k={k:<3d} "
                  f"conds={len(cond_rows):<5d} ({time.time() - started:.0f}s)")

    cond = pl.DataFrame(cond_rows)
    D = {key: np.stack(rec[key]) for key in ("D_bin", "D_pres3", "D_cnt")}
    D_fold = {key: np.stack(rec_fold[key]) for key in ("D_bin", "D_pres3", "D_cnt")}
    pred_after = {key: np.stack(rec_pred[key]) for key in ("bin", "multi")}
    print(f"[sweep] {len(cond)} conditions, D_bin {D['D_bin'].shape}, "
          f"{time.time() - started:.0f}s")

    # ---------------- flip rates ------------------------------------------- #
    flip_rows = []
    flip_bin = (pred_after["bin"] == 0)                       # present -> absent
    flip_mul = pred_after["multi"] != pred_mul_before[None, :]
    flip_to_absent = flip_mul & (pred_after["multi"] == 0)
    flip_to_count = flip_mul & (pred_after["multi"] != 0)
    for i, row in enumerate(cond.iter_rows(named=True)):
        flip_rows.append({**{key: row[key] for key in ("method", "repeat", *BLOCK_KEYS)},
                          "readout": "bin_present_to_absent",
                          "rate": float(flip_bin[i].mean())})
        for name, arr in (("multi_flip", flip_mul), ("multi_flip_to_absent", flip_to_absent),
                          ("multi_flip_to_other_count", flip_to_count)):
            flip_rows.append({**{key: row[key] for key in ("method", "repeat", *BLOCK_KEYS)},
                              "readout": name, "rate": float(arr[i].mean())})
    flips = pl.DataFrame(flip_rows)
    flips.write_csv(out_dir / f"exp3_{group}_flips.csv")

    # ---------------- blocks ----------------------------------------------- #
    blocks: dict[tuple, dict] = {}
    for i, row in enumerate(cond.iter_rows(named=True)):
        key = tuple(row[k] for k in BLOCK_KEYS)
        block = blocks.setdefault(key, {"meta": dict(zip(BLOCK_KEYS, key)), "methods": {}})
        block["methods"].setdefault(row["method"], []).append(i)

    # ---------------- bootstrap summary ------------------------------------ #
    codes_all = C.cluster_index(compounds)[0]
    print(f"[stats] bootstrapping {len(blocks)} blocks x {args.boot} reps ...")
    summary_rows = []
    curves: dict[tuple, dict] = {}
    for block in blocks.values():
        meta = block["meta"]
        for method, members in block["methods"].items():
            for key in ("bin", "pres3", "cnt"):
                vals = D[f"D_{key}"][members].mean(axis=0)
                st = C.bootstrap_mean(vals, codes_all, args.boot, args.seed)
                base = {"group": group, "method": method, "readout": key, **meta,
                        "n_repeats": len(members), "value": st["mean"],
                        "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"], "n": st["n"],
                        "n_clusters": st["n_clusters"]}
                summary_rows.append(base)
                curves.setdefault((method, key, meta["width_cm"], meta["perturb"]), {})[
                    meta["coverage_pct"]] = st["mean"]
    summary = pl.DataFrame(summary_rows)
    summary.write_csv(out_dir / f"exp3_{group}_summary.csv")

    # ---------------- paired tests ----------------------------------------- #
    paired_rows = []
    for block in blocks.values():
        meta = block["meta"]
        members_by_method = block["methods"]
        # (i) within one condition: recognition vs multiplicity read-out
        for method, members in members_by_method.items():
            d_bin = D["D_bin"][members].mean(axis=0)
            d_cnt = D["D_cnt"][members].mean(axis=0)
            d_pres = D["D_pres3"][members].mean(axis=0)
            for label, a, b in (("D_bin_minus_D_cnt", d_bin, d_cnt),
                                ("D_pres3_minus_D_cnt", d_pres, d_cnt),
                                ("D_bin_minus_D_pres3", d_bin, d_pres)):
                test = C.bootstrap_paired(a, b, codes_all, args.boot, args.seed)
                paired_rows.append({"group": group, "contrast": label, "method": method,
                                    "reference": "same_condition", **meta,
                                    "n_repeats": len(members), **test})
        # (ii) between selections on the same read-out
        if "kan_bin" in members_by_method:
            ref_bin = D["D_bin"][members_by_method["kan_bin"]].mean(axis=0)
            for method, members in members_by_method.items():
                if method == "kan_bin":
                    continue
                other = D["D_bin"][members].mean(axis=0)
                test = C.bootstrap_paired(ref_bin, other, codes_all, args.boot, args.seed)
                paired_rows.append({"group": group, "contrast": "kan_bin_vs_method",
                                    "method": method, "reference": "D_bin", **meta,
                                    "n_repeats": len(members), **test})
        if "kan_cnt" in members_by_method:
            ref_cnt = D["D_cnt"][members_by_method["kan_cnt"]].mean(axis=0)
            for method, members in members_by_method.items():
                if method == "kan_cnt":
                    continue
                other = D["D_cnt"][members].mean(axis=0)
                test = C.bootstrap_paired(ref_cnt, other, codes_all, args.boot, args.seed)
                paired_rows.append({"group": group, "contrast": "kan_cnt_vs_method",
                                    "method": method, "reference": "D_cnt", **meta,
                                    "n_repeats": len(members), **test})
    paired = pl.DataFrame(paired_rows)
    paired.write_csv(out_dir / f"exp3_{group}_paired.csv")

    # ---------------- curves / AUC ----------------------------------------- #
    curve_rows, auc_rows = [], []
    for (method, key, width, perturb), points in curves.items():
        xs = np.array(sorted(points))
        ys = np.array([points[x] for x in xs])
        for x, y in zip(xs, ys):
            curve_rows.append({"group": group, "method": method, "readout": key,
                               "width_cm": width, "perturb": perturb,
                               "coverage_pct": float(x), "D_mean": float(y)})
        auc_rows.append({"group": group, "method": method, "readout": key,
                         "width_cm": width, "perturb": perturb,
                         "D_auc": float(ys.mean()), "D_at_max_coverage": float(ys[-1]),
                         "coverage_max_pct": float(xs[-1])})
    pl.DataFrame(curve_rows).write_csv(out_dir / f"exp3_{group}_curves.csv")
    pl.DataFrame(auc_rows).sort(["width_cm", "perturb", "readout", "method"]).write_csv(
        out_dir / f"exp3_{group}_auc.csv")

    # ---------------- per-fold (five models) -------------------------------- #
    fold_rows = []
    for block in blocks.values():
        meta = block["meta"]
        for method, members in block["methods"].items():
            for key in ("bin", "cnt"):
                d_fold = D_fold[f"D_{key}"][members].astype(np.float32).mean(axis=0)
                for f in range(d_fold.shape[0]):
                    fold_rows.append({"group": group, "method": method, "readout": key,
                                      **meta, "fold": f + 1,
                                      "D_mean": float(d_fold[f].mean())})
    pl.DataFrame(fold_rows).write_csv(out_dir / f"exp3_{group}_perfold.csv")

    # ---------------- persist ---------------------------------------------- #
    np.savez_compressed(
        out_dir / f"exp3_{group}_per_sample.npz",
        D_bin=D["D_bin"].astype(np.float16),
        D_pres3=D["D_pres3"].astype(np.float16),
        D_cnt=D["D_cnt"].astype(np.float16),
        pred_after_bin=pred_after["bin"], pred_after_multi=pred_after["multi"],
        sample_idx=idx,
    )
    cond.write_parquet(out_dir / f"exp3_{group}_conditions.parquet")
    pl.DataFrame({
        "sample_idx": idx, "compound_key": compounds, "count": counts,
        "class": y_multi, "bin_label": y_bin,
        "pred_bin": pred_bin_before, "pred_multi": pred_mul_before,
        "bin_correct": bin_correct, "multi_correct": mul_correct,
        "m_bin": ref["bin"], "m_pres3": ref["pres3"], "m_cnt": ref["cnt"],
        "source_name": data.frame["source_name"].to_numpy()[idx],
    }).write_parquet(out_dir / f"exp3_{group}_samples.parquet")

    C.write_json(out_dir / f"exp3_{group}_meta.json", {
        "group": group, "task_binary": task_bin, "task_multi": task_multi,
        "n": int(n), "n_c1": int((counts == 1).sum()), "n_c2plus": int((counts >= 2).sum()),
        "n_compounds": int(len(set(compounds))),
        "folds": ens.fold_files, "widths_cm": widths, "coverages_pct": coverages,
        "methods": list(METHODS), "perturbs": perturbs, "boot": args.boot,
        "n_conditions": int(len(cond)),
        "chem_bands": C.FUNCTIONAL_GROUP_REGIONS[group],
        "chem_band_coverage_pct": C.characteristic_coverage(group),
        "unperturbed": {
            "m_bin": float(ref["bin"].mean()),
            "m_pres3": float(ref["pres3"].mean()),
            "m_cnt": float(ref["cnt"].mean()),
            "acc_binary_head": float(bin_correct.mean()),
            "acc_multiplicity_head": float(mul_correct.mean()),
            "n_pred_absent_by_binary_head": int((pred_bin_before == 0).sum()),
        },
        "collapsed_map_rate_cam_bin": float(collapsed_bin.mean()),
        "runtime_s": round(time.time() - started, 1),
    })

    sel = summary.filter((pl.col("width_cm") == 30.0) & (pl.col("perturb") == "interp"))
    print("\n[exp3 done] width 30 / interp / 4 % coverage")
    print(sel.filter(pl.col("coverage_pct") == 4.0)
          .select(["method", "readout", "value", "ci_lo", "ci_hi"])
          .sort(["readout", "method"]))
    print(f"\n[out] {out_dir}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
