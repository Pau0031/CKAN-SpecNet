#!/usr/bin/env python
"""exp1b - one chemical region, two questions: presence vs multiplicity.

The cleanest form of the "binary evidence is not multiplicity evidence" test.

For each functional group we take the *complete* literature IR region that
defines it (``ckan_specnet.regions``: 9 windows of 30 cm^-1 for nitro, 2 for
nitriles), remove it (operator A and B), re-run the whole five-fold ensemble and
read out **two different logit margins on the same perturbation**:

    presence margin   m_pres = logsumexp(o_1, o_2) - o_0      "is the group there?"
    multiplicity      m_cnt  = s * (o_2 - o_1)                "one or several?"

A region that supports recognition must move ``m_pres``; a region that supports
counting must move ``m_cnt``.  Random regions of the same width are the control.

Usage
-----
    python exp1b_presence_vs_count_margin.py --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import common as C  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--rand-reps", type=int, default=20)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=C.RESULT_DIR / "exp1")
    return p.parse_args()


def presence_margin(logits: np.ndarray) -> np.ndarray:
    """``logsumexp(o_1, o_2) - o_0``: present vs absent, count class marginalised out."""
    a, b, c = logits[..., 1], logits[..., 2], logits[..., 0]
    return np.logaddexp(a, b) - c


def main() -> None:
    args = parse_args()
    device = C.parse_device(args.device)
    ens = C.load_ensemble(device)
    data = C.load_analysis_data()
    win = C.make_windows(30.0)

    rows = []
    for group, spec in C.GROUPS.items():
        task = spec["multi"]
        idx = C.group_samples(data, group, min_count=1)
        counts = data.frame[spec["column"]].to_numpy()[idx]
        sign = C.support_sign(counts)
        compounds = data.compounds[idx]
        codes = C.cluster_index(compounds)[0]

        X = ens.preprocess(data.raw[idx], fold=0)
        logits = C.fold_logits(ens, X, task).mean(axis=0)
        m_pres0 = presence_margin(logits)
        m_cnt0 = sign * (logits[:, 2] - logits[:, 1])
        pred0 = logits.argmax(axis=1)
        y = data.labels[task][idx]
        correct = pred0 == y

        in_band = C.characteristic_windows(group, win)
        n_band = int(in_band.sum())
        band_idx = np.flatnonzero(in_band)
        span = win.width_pts * len(band_idx) * C.STEP_CM

        rng = np.random.default_rng(args.seed)
        selections = {
            "literature_band": np.tile(band_idx, (len(idx), 1)),
        }
        for rep in range(args.rand_reps):
            selections[f"random_same_width_r{rep:02d}"] = C.select_random(
                len(idx), win, len(band_idx), rng)

        for name, sel in selections.items():
            for kind in ("interp", "baseline"):
                masks = C.masks_from_selection(sel, win)
                Xp = C.perturb(X, masks, kind=kind)
                after = C.fold_logits(ens, Xp, task).mean(axis=0)
                m_pres1 = presence_margin(after)
                m_cnt1 = sign * (after[:, 2] - after[:, 1])
                pred1 = after.argmax(axis=1)

                d_pres = m_pres0 - m_pres1
                d_cnt = m_cnt0 - m_cnt1
                flip = correct & (pred1 != pred0)
                to_absent = flip & (pred1 == 0)
                to_count = flip & (pred1 != 0)

                st_p = C.bootstrap_mean(d_pres, codes, args.boot, args.seed)
                st_c = C.bootstrap_mean(d_cnt, codes, args.boot, args.seed)
                st_f = C.bootstrap_rate(to_absent[correct].astype(float),
                                        C.cluster_index(compounds[correct])[0],
                                        args.boot, args.seed)
                st_g = C.bootstrap_rate(to_count[correct].astype(float),
                                        C.cluster_index(compounds[correct])[0],
                                        args.boot, args.seed)
                base = {"group": group, "task": task, "region": name, "perturb": kind,
                        "n": int(len(idx)), "n_correct": int(correct.sum()),
                        "n_windows": int(len(band_idx)),
                        "axis_coverage_pct": 100.0 * span / C.AXIS_SPAN}
                rows.append({**base, "metric": "presence_margin_drop",
                             "value": st_p["mean"], "ci_lo": st_p["ci_lo"],
                             "ci_hi": st_p["ci_hi"]})
                rows.append({**base, "metric": "count_margin_drop",
                             "value": st_c["mean"], "ci_lo": st_c["ci_lo"],
                             "ci_hi": st_c["ci_hi"]})
                rows.append({**base, "metric": "flip_to_absent_rate",
                             "value": st_f["mean"], "ci_lo": st_f["ci_lo"],
                             "ci_hi": st_f["ci_hi"]})
                rows.append({**base, "metric": "flip_to_other_count_rate",
                             "value": st_g["mean"], "ci_lo": st_g["ci_lo"],
                             "ci_hi": st_g["ci_hi"]})
        print(f"[{group}] band = {n_band} windows ({span:.0f} cm^-1, "
              f"{100.0 * span / C.AXIS_SPAN:.2f} % of axis), n = {len(idx)}")

    frame = pl.DataFrame(rows)
    out = C.ensure_dir(args.out_dir)
    frame.write_csv(out / "exp1b_presence_vs_count.csv")

    show = frame.filter(pl.col("region") == "literature_band").sort(["group", "perturb", "metric"])
    print("\n[literature band only]")
    print(show.select(["group", "perturb", "metric", "value", "ci_lo", "ci_hi"]))

    ctrl = frame.filter(pl.col("region").str.starts_with("random")).group_by(
        ["group", "perturb", "metric"]).agg(
        pl.col("value").mean().alias("value"),
        pl.col("value").std().alias("std_over_reps"),
    ).sort(["group", "perturb", "metric"])
    print("\n[random region of the same width]")
    print(ctrl)
    print(f"\n[out] {out / 'exp1b_presence_vs_count.csv'}")


if __name__ == "__main__":
    main()
