#!/usr/bin/env python
"""exp5 - scale-free re-statistics of the perturbation effects (teacher's remark).

Problem
-------
exp3 / exp3b read two *different task heads* on the same perturbation and compared
the **absolute** logit-margin drops:

    literature band removed, 4 % coverage:  D_bin = 8.72 (nitro)  vs  D_cnt = 0.99

which is a ratio of 8.8x (and 31x for nitriles).  That ratio is not interpretable:
the binary head and the count head have different output scales, different class
distributions and different training objectives, so a logit of 1 is not the same
amount of "evidence" in the two heads.  **Nothing is retrained here** -- the same
perturbed forward passes are re-used; only the read-out is put on a common scale.

What is recomputed
------------------
For every head and every analysis population we estimate the scale of the
*unperturbed* margin (mean, standard deviation, median absolute value, IQR) and
report the perturbation effect in units of that scale:

    D_z      = D / sd(m)                 (the teacher's suggestion: per-head z-scale)
    D_rel    = D / median|m|             (robust scale)
    D_pct_i  = D_i / |m_i|               (per-spectrum relative drop, then averaged)
    flip     = fraction of spectra whose prediction changes   (0-1 probability scale)

Cross-head statements are then made either on the z-scale (D_z, which answers "how
many of this head's own margin widths does the perturbation move") or on the flip
scale (which is already probability and needs no normalisation at all).  Inside one
head the method ranking is unchanged by the normalisation and remains the primary
evidence.

Outputs (``results/exp5/``)
----------------------------
    exp5_scales.csv            per head / stratum / group: mean, sd, median|m|, IQR
    exp5_effects.csv           effect of every selection method on every scale
    exp5_head_contrast.csv     same spectra, two heads: paired z-scale contrast
    exp5_flips.csv             flip rates of both heads (probability scale)
    exp5_summary.md            the tables quoted in readme3.md

Usage
-----
    python exp5_normalized_effects.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``common.py`` (ensemble loading, analysis windows, the two perturbation
# operators, the cluster bootstrap) lives in the exp1 folder and is shared here.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "exp1_attribution_fidelity"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import common as C  # noqa: E402

RES = C.RESULT_DIR
OUT = C.ensure_dir(RES / "exp5")
GROUPS = ["nitro", "nitriles"]
HEADS = {"bin": "m_bin", "pres3": "m_pres3", "cnt": "m_cnt"}
METHODS = ["chem", "kan_cnt", "kan_bin", "peak", "cam_bin", "random"]
LABEL = {"chem": "文献特征带", "kan_cnt": "KAN（计数头）", "kan_bin": "KAN（二分类头）",
         "peak": "峰强度", "cam_bin": "Grad-CAM（二分类头）", "random": "随机"}
BOOT = 2000


def load_conditions(tag: str) -> pl.DataFrame:
    return pl.read_parquet(RES / tag / f"{tag}_conditions.parquet")


def select(cond: pl.DataFrame, method: str, coverage: float, perturb: str = "interp",
           width: float = 30.0) -> np.ndarray:
    m = ((cond["method"] == method) & (cond["coverage_pct"] == coverage)
         & (cond["perturb"] == perturb) & (cond["width_cm"] == width))
    return np.flatnonzero(m.to_numpy())


def boot_mean(values: np.ndarray, codes: np.ndarray, n_boot: int = BOOT,
              seed: int = 0) -> dict:
    st = C.bootstrap_mean(values, codes, n_boot, seed)
    return {"value": st["mean"], "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"]}


def boot_paired(a: np.ndarray, b: np.ndarray, codes: np.ndarray,
                n_boot: int = BOOT, seed: int = 0) -> dict:
    t = C.bootstrap_paired(a, b, codes, n_boot, seed)
    return {"diff": t["diff"], "ci_lo": t["ci_lo"], "ci_hi": t["ci_hi"],
            "p": t["p_two_sided"]}


def scales_for(m: np.ndarray, mask: np.ndarray) -> dict:
    v = m[mask]
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)),
        "median_abs": float(np.median(np.abs(v))),
        "iqr": float(np.quantile(v, 0.75) - np.quantile(v, 0.25)),
        "n": int(mask.sum()),
    }


def main() -> None:
    scale_rows, effect_rows, contrast_rows, flip_rows = [], [], [], []

    for group in GROUPS:
        smp = pl.read_parquet(RES / "exp3b" / f"exp3b_{group}_samples.parquet")
        cond = load_conditions(f"exp3b_{group}") if False else pl.read_parquet(
            RES / "exp3b" / f"exp3b_{group}_conditions.parquet")
        D = np.load(RES / "exp3b" / f"exp3b_{group}_per_sample.npz")
        present = smp["present"].to_numpy()
        clusters = smp["compound_key"].to_numpy().astype(str)
        codes_all = C.cluster_index(clusters)[0]
        codes_present = C.cluster_index(clusters[present])[0]
        codes_absent = C.cluster_index(clusters[~present])[0]
        margins = {h: smp[col].to_numpy().astype(np.float64) for h, col in HEADS.items()}
        pred_before = {"bin": smp["pred_bin"].to_numpy(),
                       "multi": smp["pred_multi"].to_numpy()}

        # ---------------- scales of the unperturbed margin ------------------ #
        scales: dict[tuple[str, str], dict] = {}
        for head, col in HEADS.items():
            strata = ("present", "absent") if head == "bin" else ("present",)
            if head == "pres3":
                strata = ("present", "absent")
            for stratum in strata:
                mask = present if stratum == "present" else ~present
                st = scales_for(margins[head], mask)
                scales[(head, stratum)] = st
                scale_rows.append({"group": group, "head": head, "stratum": stratum, **st})

        # ---------------- effects on three scales, per head/stratum --------- #
        for coverage in (1.0, 2.0, 4.0, 6.0, 10.0):
            for method in METHODS:
                idx = select(cond, method, coverage)
                if idx.size == 0:
                    continue
                for head in ("bin", "pres3", "cnt"):
                    strata = ("present", "absent") if head in ("bin", "pres3") else ("present",)
                    for stratum in strata:
                        mask = present if stratum == "present" else ~present
                        d = D[f"D_{head}"][idx][:, mask].mean(axis=0)
                        sc = scales[(head, stratum)]
                        z = d / sc["sd"]
                        rel = d / sc["median_abs"]
                        m_abs = np.abs(margins[head][mask])
                        pct = np.mean(d / np.maximum(m_abs, 1e-6))
                        codes = codes_present if stratum == "present" else codes_absent
                        st_z = boot_mean(z, codes)
                        effect_rows.append({
                            "group": group, "head": head, "stratum": stratum,
                            "method": method, "coverage_pct": coverage,
                            "perturb": "interp", "width_cm": 30.0,
                            "D": float(d.mean()), "D_z": float(z.mean()),
                            "D_z_lo": st_z["ci_lo"], "D_z_hi": st_z["ci_hi"],
                            "D_rel": float(rel.mean()), "D_pct": float(pct),
                            "sd": sc["sd"], "median_abs": sc["median_abs"],
                            "n": int(mask.sum())})

        # ---------------- same spectra, two heads: paired z-contrast -------- #
        for coverage in (1.0, 2.0, 4.0, 6.0, 10.0):
            for method in METHODS:
                idx = select(cond, method, coverage)
                if idx.size == 0:
                    continue
                for head in ("bin", "pres3"):
                    d_h = D[f"D_{head}"][idx][:, present].mean(axis=0)
                    d_c = D["D_cnt"][idx][:, present].mean(axis=0)
                    z_h = d_h / scales[(head, "present")]["sd"]
                    z_c = d_c / scales[("cnt", "present")]["sd"]
                    contrast_rows.append({
                        "group": group, "head": head, "method": method,
                        "coverage_pct": coverage, "perturb": "interp", "width_cm": 30.0,
                        "D_z_head": float(z_h.mean()), "D_z_cnt": float(z_c.mean()),
                        **{f"diff_{k}": v for k, v in
                           boot_paired(z_h, z_c, codes_present).items()},
                        "ratio_descriptive": float(z_h.mean() / z_c.mean())
                        if abs(z_c.mean()) > 1e-9 else float("nan"),
                        "n": int(present.sum())})

        # ---------------- flip rates (probability scale) -------------------- #
        for coverage in (1.0, 2.0, 4.0, 6.0, 10.0):
            for method in METHODS:
                idx = select(cond, method, coverage)
                if idx.size == 0:
                    continue
                for head, key in (("bin", "pred_after_bin"), ("multi", "pred_after_multi")):
                    after = D[key][idx]
                    before = pred_before[head][None, :]
                    flags = {
                        "flip_any": after != before,
                        "flip_to_absent": (after != before) & (after == 0),
                        "flip_to_other_count": (after != before) & (after != 0),
                    }
                    for stratum, mask, codes in (("present", present, codes_present),
                                                 ("absent", ~present, codes_absent)):
                        for what, flag in flags.items():
                            if head == "bin" and what != "flip_any":
                                continue
                            st = boot_mean(flag.mean(axis=0)[mask], codes)
                            flip_rows.append({
                                "group": group, "head": head, "stratum": stratum,
                                "metric": what, "method": method,
                                "coverage_pct": coverage, "perturb": "interp",
                                "width_cm": 30.0, "flip_rate": st["value"],
                                "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"],
                                "n": int(mask.sum())})

    scales_df = pl.DataFrame(scale_rows).sort(["group", "head", "stratum"])
    effects_df = pl.DataFrame(effect_rows).sort(
        ["group", "head", "stratum", "coverage_pct", "method"])
    contrast_df = pl.DataFrame(contrast_rows).sort(
        ["group", "head", "coverage_pct", "method"])
    flips_df = pl.DataFrame(flip_rows).sort(
        ["group", "head", "stratum", "coverage_pct", "method"])

    scales_df.write_csv(OUT / "exp5_scales.csv")
    effects_df.write_csv(OUT / "exp5_effects.csv")
    contrast_df.write_csv(OUT / "exp5_head_contrast.csv")
    flips_df.write_csv(OUT / "exp5_flips.csv")

    write_summary(scales_df, effects_df, contrast_df, flips_df)
    print(f"[exp5 done] -> {OUT}")
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(200)
    print(effects_df.filter((pl.col("coverage_pct") == 4.0) & (pl.col("group") == "nitro"))
          .select(["head", "stratum", "method", "D", "D_z", "D_rel"]).head(40))


def write_summary(scales: pl.DataFrame, effects: pl.DataFrame,
                  contrast: pl.DataFrame, flips: pl.DataFrame) -> None:
    L: list[str] = ["# exp5 — 尺度无关的效应量重算（自动生成）\n",
                    "所有数字来自 exp3b 的同一批扰动结果，仅把读数换算到各头自身的 margin 尺度。\n"]

    L.append("\n## 表 1 未扰动 margin 的分布尺度\n")
    L.append("| 官能团 | 头 | 层 | 均值 | 标准差 | median\\|m\\| | IQR | n |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in scales.iter_rows(named=True):
        L.append(f"| {r['group']} | {r['head']} | {r['stratum']} | {r['mean']:.2f} | "
                 f"{r['sd']:.2f} | {r['median_abs']:.2f} | {r['iqr']:.2f} | {r['n']} |")

    for group in GROUPS:
        L.append(f"\n## 表 2-{group} 4 % 覆盖下的效应量（present 层）\n")
        L.append("| 头 | 选区方法 | D（logit 原尺度） | D_z = D/sd | 95 % CI | D_rel = D/median\\|m\\| | D_pct（逐谱相对下降） |")
        L.append("|---|---|---|---|---|---|---|")
        sub = effects.filter((effects["group"] == group) & (effects["coverage_pct"] == 4.0)
                             & (effects["stratum"] == "present"))
        for head in ("bin", "cnt", "pres3"):
            for r in sub.filter(sub["head"] == head).iter_rows(named=True):
                L.append(f"| {head} | {LABEL[r['method']]} | {r['D']:.2f} | "
                         f"**{r['D_z']:.3f}** | {r['D_z_lo']:.3f}–{r['D_z_hi']:.3f} | "
                         f"{r['D_rel']:.3f} | {100 * r['D_pct']:.1f} % |")

        L.append(f"\n## 表 3-{group} 同一批谱上两个头的 z 尺度配对差（present 层，4 %）\n")
        L.append("| 对比 | 选区方法 | 识别头 D_z | 计数头 D_z | 配对差 | 95 % CI | p |")
        L.append("|---|---|---|---|---|---|---|")
        sub = contrast.filter((contrast["group"] == group) & (contrast["coverage_pct"] == 4.0))
        for head in ("bin", "pres3"):
            for r in sub.filter(sub["head"] == head).iter_rows(named=True):
                L.append(f"| {head} − cnt | {LABEL[r['method']]} | {r['D_z_head']:.3f} | "
                         f"{r['D_z_cnt']:.3f} | {r['diff_diff']:+.3f} | "
                         f"{r['diff_ci_lo']:.3f}–{r['diff_ci_hi']:.3f} | "
                         f"{r['diff_p']:.3g} |")

    L.append("\n## 表 4 翻转率（概率尺度，4 % 覆盖）\n")
    L.append("| 官能团 | 头 | 层 | 指标 | 选区方法 | 比率 | 95 % CI |")
    L.append("|---|---|---|---|---|---|---|")
    sub = flips.filter(flips["coverage_pct"] == 4.0)
    for r in sub.iter_rows(named=True):
        L.append(f"| {r['group']} | {r['head']} | {r['stratum']} | {r['metric']} | "
                 f"{LABEL[r['method']]} | "
                 f"{100 * r['flip_rate']:.1f} % | "
                 f"{100 * r['ci_lo']:.1f}–{100 * r['ci_hi']:.1f} |")

    (OUT / "exp5_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
