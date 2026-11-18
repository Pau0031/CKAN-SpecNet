#!/usr/bin/env python
"""Figures and summary table for the revision experiments.

    figX_attribution_fidelity   (a) nitro: margin degradation vs perturbed fraction
                                (b) nitriles: same
                                (c) where the flips go at 4% coverage: towards
                                    "absent" (0) or towards the other count class
                                (d) representative spectra: KAN / Grad-CAM++ maps and
                                    the windows each method perturbs
    figXS_flip_and_robustness   (SI) flip decomposition over all coverages, window-width
                                sensitivity, perturbation-operator sensitivity
    figY_branch_contribution    (a) branch ratio  (b) direction / correction / introduction
                                (c) count-margin AUC of full / base / KAN
                                (d) m_base vs m_KAN per spectrum
                                (e) KAN evidence inside the literature bands:
                                    binary vs multiplicity on the same compounds
    tableX_summary.csv/.md      support, F1, AUC, perturbation effect, branch statistics

Everything is read from ``results/exp0..exp2``; no model is run here.

Usage
-----
    python make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402

RES = C.RESULT_DIR
FIG_DIR = RES / "figs"
TAB_DIR = RES / "tables"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.4,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

METHOD_ORDER = ["kan", "gradcampp", "peak", "chem", "random"]
GROUPS = ("nitro", "nitriles")


def released_per_class():
    """Released per-class metrics, when ``scripts/evaluate.py`` has been run.

    The summary table quotes the released evaluation next to the interpretability
    numbers.  ``scripts/evaluate.py ... --out results/reproduce`` writes exactly
    that file; a run directory that keeps the evaluation tables next to the
    weights is accepted as well.  When neither exists the two ``released_*``
    columns stay empty instead of aborting the figure pipeline.
    """
    for path in (C.RUN_DIR / "main_test_per_class_metrics.csv",
                 RES / "reproduce" / "main_test_per_class_metrics.csv",
                 RES / "released" / "main_test_per_class_metrics.csv"):
        path = Path(path)
        if path.is_file():
            return pd.read_csv(path)
    print("[table] released per-class metrics not found; run scripts/evaluate.py "
          "first (e.g. --out results/reproduce) to fill the released_* columns")
    return None


# --------------------------------------------------------------------------- #
def load_exp1(group: str) -> dict:
    d = RES / "exp1"
    out = {
        "summary": pd.read_csv(d / f"exp1_{group}_summary.csv"),
        "auc": pd.read_csv(d / f"exp1_{group}_auc.csv"),
        "paired": pd.read_csv(d / f"exp1_{group}_paired_tests.csv"),
        "task": pd.read_csv(d / f"exp1_{group}_task_stats.csv"),
        "diag": pd.read_csv(d / f"exp1_{group}_map_diagnostics.csv"),
        "samples": pd.read_parquet(d / f"exp1_{group}_samples.parquet"),
        "cond": pd.read_parquet(d / f"exp1_{group}_conditions.parquet"),
    }
    with np.load(d / f"exp1_{group}_per_sample.npz") as z:
        out["D"] = z["D"].astype(np.float32)
    with np.load(d / f"exp1_{group}_maps.npz") as z:
        out["maps"] = {k: z[k].astype(np.float32) for k in z.files}
    spread = d / f"exp1_{group}_random_spread.csv"
    out["spread"] = pd.read_csv(spread) if spread.is_file() else None
    return out


def load_exp2() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(RES / "exp2" / "exp2_metrics.csv")
    p = RES / "exp2" / "exp2_enrichment.csv"
    e = pd.read_csv(p) if p.is_file() else pd.DataFrame()
    return m, e


def primary(exp1: dict, subset: str = "all", width: float = 30.0,
            perturb: str = "interp", renorm: bool = False) -> pd.DataFrame:
    s = exp1["summary"]
    return s[(s.subset == subset) & (s.width_cm == width) & (s.perturb == perturb)
             & (s.renorm == renorm)]


def _panel_label(ax, letter: str, dx: float = -0.16, dy: float = 1.08) -> None:
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top")


# --------------------------------------------------------------------------- #
def panel_curves(ax, exp1: dict, group: str, coverage_note: str = "") -> None:
    s = primary(exp1)[primary(exp1)["metric"] == "D_mean"]
    for method in METHOD_ORDER:
        blk = s[s.method == method].sort_values("frac_covered_pct")
        if blk.empty:
            continue
        color = C.METHOD_COLORS[method]
        ax.plot(blk.frac_covered_pct, blk.value, marker=C.METHOD_MARKERS[method], ms=3.4,
                lw=1.5, color=color, label=C.METHOD_LABELS[method],
                zorder=6 if method == "kan" else 3)

        ax.fill_between(blk.frac_covered_pct, blk.ci_lo, blk.ci_hi, color=color,
                        alpha=0.13, lw=0, zorder=1)
    ax.axhline(0.0, color="0.55", lw=0.8, ls=":")
    ax.set_ylim(*({"nitro": (-0.55, 4.7), "nitriles": (-0.95, 5.7)}[group]))
    ax.set_xlabel("Perturbed fraction of the spectral axis (%)")
    ax.set_ylabel(r"Margin degradation  $D(k)$")
    ax.set_title(f"{C.GROUPS[group]['short']} ({C.GROUPS[group]['multi'].replace('_', ' ')})",
                 loc="left", pad=6)


def panel_flip_decomposition(ax, exp1: dict, group: str, coverage: float = 4.0) -> None:
    s = primary(exp1, subset="correct")
    rows = []
    for method in METHOD_ORDER:
        blk = s[(s.method == method) & (s.coverage_pct == coverage)]
        if blk.empty:
            continue
        total = float(blk[blk["metric"] == "flip_rate"].value.iloc[0])
        to_count = float(blk[blk["metric"] == "flip_to_other_count_rate"].value.iloc[0])
        rows.append((method, total - to_count, to_count))
    if not rows:
        return
    x = np.arange(len(rows))
    to_absent = np.array([r[1] for r in rows]) * 100
    to_count = np.array([r[2] for r in rows]) * 100
    ax.bar(x, to_absent, width=0.62, color="#B0B7BC", label="flip to “absent” (class 0)")
    ax.bar(x, to_count, width=0.62, bottom=to_absent, color="#7B241C",
           label="flip to the other count class")
    for i, (m, a, c) in enumerate(rows):
        ax.text(i, a + c + 1.0, f"{100*(a+c):.0f}", ha="center", fontsize=6.6, color="0.25")
    ax.set_xticks(x)
    ax.set_xticklabels([C.METHOD_LABELS[r[0]] for r in rows], rotation=20, ha="right",
                       fontsize=7.2)
    ax.set_ylabel("samples whose prediction\nflips (%)")
    ax.set_title(f"{C.GROUPS[group]['short']}: where {coverage:.0f}% perturbations flip the model",
                 loc="left", fontsize=8.6, pad=5)
    ax.set_ylim(0, 72)
    for i, (m, a, c) in enumerate(rows):
        ax.text(i, 100 * (a + c) + 1.4, f"{100*(a+c):.0f}", ha="center", fontsize=6.4,
                color="0.3", va="bottom")


def pick_representative(exp1: dict, cls: int, width: float = 30.0,
                        coverage: float = 4.0) -> int:
    cond, sam = exp1["cond"], exp1["samples"]
    row = cond[(cond.method == "kan") & (cond.width_cm == width)
               & (cond.coverage_pct == coverage) & (cond.repeat == 0)
               & (cond.perturb == "interp") & (cond.renorm == False)]  # noqa: E712
    if row.empty:
        return int(np.flatnonzero(sam["class"].to_numpy() == cls)[0])
    d = exp1["D"][int(row.cond_id.iloc[0])]
    sel = (sam["class"].to_numpy() == cls) & sam["correct"].to_numpy()
    if not sel.any():
        sel = sam["class"].to_numpy() == cls
    idx = np.flatnonzero(sel)
    return int(idx[np.argmin(np.abs(d[idx] - float(np.median(d[sel]))))])


def panel_example(ax_spec, ax_map, exp1: dict, group: str, sample: int,
                  width: float = 30.0, coverage: float = 4.0) -> dict:
    sam, cond, maps = exp1["samples"].iloc[sample], exp1["cond"], exp1["maps"]
    grid = C.GRID
    x = maps["X"][sample]
    win = C.make_windows(width)
    k = C.coverage_to_k(win, coverage)
    kan_scores = C.window_sum(maps["kan"][sample][None, :], win)[0]
    cam_scores = C.window_sum(maps["gradcampp"][sample][None, :], win)[0]
    sel_kan = C.top_k(kan_scores, k)
    sel_cam = C.top_k(cam_scores, k)

    def cond_d(method):
        r = cond[(cond.method == method) & (cond.width_cm == width)
                 & (cond.coverage_pct == coverage) & (cond.repeat == 0)
                 & (cond.perturb == "interp") & (cond.renorm == False)]  # noqa: E712
        return float(exp1["D"][int(r.cond_id.iloc[0]), sample]) if not r.empty else np.nan

    ax_spec.plot(grid, x, color="0.35", lw=0.8, zorder=3)
    for w in sel_kan:
        ax_spec.axvspan(grid[win.starts[w]], grid[win.stops[w] - 1],
                        color=C.METHOD_COLORS["kan"], alpha=0.22, lw=0, zorder=1)
    for w in sel_cam:
        if w in set(sel_kan.tolist()):
            continue
        ax_spec.axvspan(grid[win.starts[w]], grid[win.stops[w] - 1],
                        color=C.METHOD_COLORS["gradcampp"], alpha=0.20, lw=0, zorder=1)
    for hi, lo in C.FUNCTIONAL_GROUP_REGIONS[group]:
        for v in (lo, hi):
            ax_spec.axvline(v, color="#229954", lw=0.7, ls="--", alpha=0.8, zorder=4)
    ax_spec.set_xlim(grid[0], grid[-1])
    ax_spec.set_xticklabels([])
    ax_spec.set_ylabel("z-score", labelpad=1, fontsize=7.6)
    state = "correct" if bool(sam["correct"]) else "ERROR"
    ax_spec.set_title(f"{C.GROUPS[group]['short']}, count={int(sam['count'])} ({state})\n"
                      f"$D_{{KAN}}$={cond_d('kan'):.2f}   $D_{{CAM}}$={cond_d('gradcampp'):.2f}",
                      loc="left", fontsize=7.6)

    for key, color, label in (("kan", C.METHOD_COLORS["kan"], "Contribution-KAN"),
                              ("gradcampp", C.METHOD_COLORS["gradcampp"], "Grad-CAM")):
        mp = maps[key][sample]
        ax_map.plot(grid, mp / (np.abs(mp).max() + 1e-12), color=color, lw=0.8,
                    label=label, alpha=0.95)
    for w in sel_kan:
        ax_map.axvspan(grid[win.starts[w]], grid[win.stops[w] - 1],
                       color=C.METHOD_COLORS["kan"], alpha=0.12, lw=0)
    ax_map.axhline(0, color="0.6", lw=0.7)
    ax_map.set_xlim(grid[0], grid[-1])
    ax_map.set_yticks([])
    ax_map.set_ylabel("attribution", fontsize=7.2, labelpad=1)
    ax_map.set_xlabel(r"Wavenumber (cm$^{-1}$)", labelpad=1, fontsize=7.6)
    ax_map.tick_params(axis="x", labelsize=6.6)
    return {"group": group, "sample": int(sample), "count": int(sam["count"]),
            "correct": bool(sam["correct"]), "D_kan": cond_d("kan"),
            "D_gradcampp": cond_d("gradcampp")}


# --------------------------------------------------------------------------- #
def pure_chem_point(exp1: dict, subset: str = "all") -> tuple[float, float]:
    """Largest coverage whose selection is still 100 % inside the literature bands."""
    cond = exp1["cond"]
    ch = cond[(cond.method == "chem") & (cond.width_cm == 30.0)
              & (cond.perturb == "interp") & (cond.renorm == False)  # noqa: E712
              & (cond.sel_inband_frac >= 0.999)]                     # noqa: E712
    if ch.empty:
        return float("nan"), float("nan")
    pure = ch.sort_values("coverage_pct").iloc[-1]
    s = primary(exp1, subset=subset)
    row = s[(s.method == "chem") & (s.coverage_pct == pure.coverage_pct)
            & (s["metric"] == "D_mean")]
    if row.empty:
        return float(pure.frac_covered_pct), float("nan")
    return float(pure.frac_covered_pct), float(row.value.iloc[0])


def figure_x(exp1_by_group: dict) -> None:
    fig = plt.figure(figsize=(7.2, 9.6))
    outer = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.78, 1.20], hspace=0.46, wspace=0.30,
                             top=0.885, bottom=0.135, left=0.105, right=0.98)

    handles = None
    for j, group in enumerate(GROUPS):
        ax = fig.add_subplot(outer[0, j])
        panel_curves(ax, exp1_by_group[group], group)
        handles = handles or ax.get_legend_handles_labels()
        frac, val = pure_chem_point(exp1_by_group[group])
        if np.isfinite(val):
            ax.plot([frac], [val], marker="*", ms=12, color="0.1", ls="none", zorder=9,
                    label="literature bands only")
            tx, ty = {"nitro": (7.3, -0.30), "nitriles": (4.3, -0.68)}[group]
            ax.annotate(f"literature bands only\n{frac:.1f} % of axis, D={val:.2f}",
                        xy=(frac, val), xytext=(tx, ty), fontsize=6.5, color="0.12",
                        arrowprops=dict(arrowstyle="->", color="0.45", lw=0.6))
        _panel_label(ax, "ab"[j], dx=-0.17)

    # one shared legend for the selection methods
    fig.legend(*handles, loc="upper center", bbox_to_anchor=(0.53, 0.945), ncol=6,
               frameon=False, fontsize=7.2, handlelength=1.5, columnspacing=1.2)

    flip_gs = outer[1, :].subgridspec(1, 2, wspace=0.30)
    for j, group in enumerate(GROUPS):
        ax = fig.add_subplot(flip_gs[0, j])
        panel_flip_decomposition(ax, exp1_by_group[group], group)
        if j == 0:
            if ax.legend_ is not None:
                ax.legend_.remove()
            ax.legend(frameon=False, fontsize=6.8, loc="upper left", handlelength=1.2)
            _panel_label(ax, "c", dx=-0.29)
        elif ax.legend_ is not None:
            ax.legend_.remove()

    ex_gs = outer[2, :].subgridspec(2, 4, height_ratios=[1.0, 0.66], hspace=0.14, wspace=0.18)
    for j, group in enumerate(GROUPS):
        exp1 = exp1_by_group[group]
        for i, cls in enumerate((1, 2)):
            col = j * 2 + i
            sample = pick_representative(exp1, cls)
            ax_spec = fig.add_subplot(ex_gs[0, col])
            ax_map = fig.add_subplot(ex_gs[1, col], sharex=ax_spec)
            panel_example(ax_spec, ax_map, exp1, group, sample)
            if col == 0:
                _panel_label(ax_spec, "d", dx=-0.42, dy=1.20)
            if col > 0:
                ax_spec.set_ylabel("")

    fig.text(0.5, 0.012,
             "Window width 30 cm$^{-1}$, k = 4 windows (4 % of the 552-3842 cm$^{-1}$ axis). "
             "Shaded windows: regions perturbed by each method (red = Contribution-KAN top-k,\n"
             "blue = Grad-CAM++ top-k); green dashed lines: literature IR bands of the group; "
             "maps are normalised per spectrum; perturbation = local interpolation (method A).",
             ha="center", va="bottom", fontsize=6.9, color="0.3", linespacing=1.5)
    fig.savefig(FIG_DIR / "figX_attribution_fidelity.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figX_attribution_fidelity.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def figure_x_supplement(exp1_by_group: dict) -> None:
    """SI: flip decomposition over all coverages + operator / window-width sensitivity."""
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0))
    fig.subplots_adjust(hspace=0.62, wspace=0.34, left=0.09, right=0.985, top=0.90,
                        bottom=0.13)
    for j, group in enumerate(GROUPS):
        exp1 = exp1_by_group[group]
        # (a,b) flips vs coverage
        ax = axes[j, 0]
        s = primary(exp1, subset="correct")
        for method in METHOD_ORDER:
            blk = s[(s.method == method) & (s["metric"] == "flip_to_other_count_rate")
                    ].sort_values("coverage_pct")
            ax.plot(blk.frac_covered_pct, 100 * blk.value, marker=C.METHOD_MARKERS[method],
                    ms=3, lw=1.3, color=C.METHOD_COLORS[method], label=C.METHOD_LABELS[method])
        ax.set_xlabel("Perturbed fraction (%)")
        ax.set_ylabel("flips to another\ncount class (%)")
        ax.set_ylim(0, 6.0)
        ax.set_title(f"{C.GROUPS[group]['short']}: count flips", loc="left", fontsize=8.4, pad=5)
        ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="upper left",
                  handlelength=1.2, columnspacing=1.0)

        # (c,d) window width sensitivity (AUC of the curve)
        ax = axes[j, 1]
        a = exp1["auc"]
        a = a[(a.subset == "all") & (a.perturb == "interp") & (a.renorm == False)
              & (a.coverage_max_pct == 10.0)]  # noqa: E712
        for method in METHOD_ORDER:
            blk = a[a.method == method].sort_values("width_cm")
            ax.plot(blk.width_cm, blk.D_auc, marker="o", ms=3.4, lw=1.3,
                    color=C.METHOD_COLORS[method], label=C.METHOD_LABELS[method])
        ax.set_xlabel(r"window width (cm$^{-1}$)")
        ax.set_ylabel("AUC of $D(k)$")
        ax.set_ylim(0, 3.0)
        ax.set_title(f"{C.GROUPS[group]['short']}: window width", loc="left", fontsize=8.4, pad=5)

        # (e,f) perturbation operator sensitivity
        ax = axes[j, 2]
        a = exp1["auc"]
        a = a[(a.subset == "all") & (a.width_cm == 30.0) & (a.renorm == False)
              & (a.perturb.isin(["interp", "baseline"]))]  # noqa: E712
        order = [m for m in METHOD_ORDER]
        x = np.arange(len(order))
        for i, (perturb, color, label) in enumerate((("interp", "#1F618D", "A: interpolation"),
                                                    ("baseline", "#B9770E", "B: baseline"))):
            vals = [float(a[(a.method == m) & (a.perturb == perturb)].D_auc.iloc[0])
                    if not a[(a.method == m) & (a.perturb == perturb)].empty else np.nan
                    for m in order]
            ax.bar(x + (i - 0.5) * 0.38, vals, width=0.38, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([C.METHOD_LABELS[m].replace(" ", "\n") for m in order],
                           fontsize=5.6, rotation=35, ha="right")
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_ylabel("AUC of $D(k)$")
        ax.set_ylim(-1.9 if group == "nitro" else -0.35, 3.4)
        ax.set_title(f"{C.GROUPS[group]['short']}: perturbation operator", loc="left",
                     fontsize=8.4, pad=5)
        ax.legend(frameon=False, fontsize=6.2, loc="upper center", ncol=2, handlelength=1.2,
                  columnspacing=1.0)

    for i, ax in enumerate(fig.axes):
        ax.text(-0.30, 1.10, f"({'abcdef'[i]})", transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top")
    fig.savefig(FIG_DIR / "figXS_flip_and_robustness.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figXS_flip_and_robustness.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def figure_x2_presence_vs_count() -> None:
    """SI: one chemical region, two questions (exp1b)."""
    path = RES / "exp1" / "exp1b_presence_vs_count.csv"
    if not path.is_file():
        return
    d = pd.read_csv(path)
    lit = d[(d.region == "literature_band") & (d.perturb == "interp")].set_index(
        ["group", "metric"])
    rnd = (d[(d.region.str.startswith("random")) & (d.perturb == "interp")]
           .groupby(["group", "metric"]).value.agg(["mean", "std"]))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    fig.subplots_adjust(wspace=0.32, left=0.095, right=0.985, top=0.84, bottom=0.20)
    metrics = [("presence_margin_drop", "presence margin\n$\log\sum e^{o_{1,2}}-o_0$", "#2E86C1"),
               ("count_margin_drop", "count margin\n$s\,(o_2-o_1)$", "#C0392B")]
    for j, group in enumerate(GROUPS):
        ax = axes[j]
        n_win = int(d[d.group == group].n_windows.iloc[0])
        cov = float(d[d.group == group].axis_coverage_pct.iloc[0])
        for i, (metric, label, color) in enumerate(metrics):
            v = float(lit.loc[(group, metric), "value"])
            lo = float(lit.loc[(group, metric), "ci_lo"])
            hi = float(lit.loc[(group, metric), "ci_hi"])
            rv = float(rnd.loc[(group, metric), "mean"])
            x = i + 0.17
            ax.bar(x, v, width=0.30, color=color, label=label)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor="0.25",
                        elinewidth=0.8, capsize=2.5)
            ax.bar(i - 0.17, rv, width=0.30, color="0.75")
            ax.text(x, hi + 0.6, f"{v:+.2f}", ha="center", fontsize=7.4, color="0.15")
            ax.text(i - 0.17, max(rv, 0) + 0.6, f"{rv:+.2f}", ha="center", fontsize=6.6,
                    color="0.4")
        ax.axhline(0, color="0.55", lw=0.8)
        ax.set_xticks(range(2))
        ax.set_xticklabels(["presence margin", "count margin"], fontsize=7.8)
        ax.set_ylabel("margin drop (logits)")
        ax.set_title(f"{C.GROUPS[group]['short']}: {n_win} literature windows "
                     f"({cov:.1f} % of axis)", loc="left", fontsize=8.6)
        top = float(lit.loc[group, "value"].max())
        ax.set_ylim(-1.6, top * 1.42)
        axes[0].legend(frameon=False, fontsize=6.4, loc="upper right", handlelength=1.2)
    fig.text(0.5, 0.035, "coloured: remove the complete literature IR region;  grey: a random "
             "region of the same width (20 repeats, mean)", ha="center", fontsize=6.8,
             color="0.3")
    for i, ax in enumerate(axes):
        ax.text(-0.13, 1.10, f"({'ab'[i]})", transform=ax.transAxes, fontsize=10.5,
                fontweight="bold", va="top")
    fig.savefig(FIG_DIR / "figXS2_presence_vs_count.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figXS2_presence_vs_count.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def figure_y(exp2: pd.DataFrame, enr: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.4, 5.4))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.40,
                          top=0.90, bottom=0.11, left=0.085, right=0.985)
    combos = [("nitro", "binary"), ("nitro", "multi"),
              ("nitriles", "binary"), ("nitriles", "multi")]
    labels = [f"{C.GROUPS[g]['short']}\n{'binary' if t == 'binary' else 'multiplicity'}"
              for g, t in combos]
    colors = ["#A6ACAF", "#C0392B", "#A6ACAF", "#C0392B"]
    per = pd.read_parquet(RES / "exp2" / "exp2_per_sample.parquet")

    def strata(g, t):
        return "all_present" if t == "multi" else "present"

    def get(g, t, stratum, metric):
        r = exp2[(exp2.group == g) & (exp2.task_type == t) & (exp2.stratum == stratum)
                 & (exp2.metric == metric)]
        return (float(r.value.iloc[0]), float(r.ci_lo.iloc[0]), float(r.ci_hi.iloc[0])) \
            if not r.empty else (np.nan, np.nan, np.nan)

    def errs(v):
        return [[a[0] - a[1] for a in v], [a[2] - a[0] for a in v]]

    # (a) branch ratio ----------------------------------------------------- #
    ax = fig.add_subplot(gs[0, 0])
    vals = [get(g, t, strata(g, t), "branch_ratio_median") for g, t in combos]
    ax.bar(range(4), [v[0] for v in vals], color=colors, width=0.62)
    ax.errorbar(range(4), [v[0] for v in vals], yerr=errs(vals), fmt="none",
                ecolor="0.25", elinewidth=0.8, capsize=2)
    for i, v in enumerate(vals):
        ax.text(i, v[2] + 0.02, f"{v[0]:.2f}", ha="center", fontsize=6.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=6.3)
    ax.set_ylabel("median branch ratio\n" + r"$|m_{\rm KAN}|/(|m_{\rm base}|+|m_{\rm KAN}|)$",
                  fontsize=8)
    ax.set_ylim(0, 0.62)
    ax.set_title("Branch ratio", loc="left", pad=6)

    # (b) direction / correction ------------------------------------------- #
    ax = fig.add_subplot(gs[0, 1])
    width = 0.26
    for i, (metric, label, color) in enumerate((
            ("sign_agreement", "sign agreement", "#2E86C1"),
            ("error_correction_rate", "fixed", "#229954"),
            ("error_introduction_rate", "broken", "#CB4335"))):
        v = [get(g, t, strata(g, t), metric) for g, t in combos]
        x = np.arange(4) + (i - 1) * width
        ax.bar(x, [a[0] for a in v], width=width, color=color, label=label, alpha=0.95)
        ax.errorbar(x, [a[0] for a in v], yerr=errs(v), fmt="none", ecolor="0.25",
                    elinewidth=0.7, capsize=1.5)
        for xx, a in zip(x, v):
            ax.text(xx, a[2] + 0.03, f"{100*a[0]:.0f}", ha="center", fontsize=5.6,
                    color="0.25", va="bottom")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=6.3)
    ax.set_ylabel("rate (%)")
    ax.set_ylim(0, 1.36)
    ax.legend(frameon=False, fontsize=6.0, ncol=3, loc="upper center",
              handlelength=1.1, columnspacing=0.9)
    ax.set_title("Direction and effect", loc="left", pad=6)

    # (c) AUC -------------------------------------------------------------- #
    ax = fig.add_subplot(gs[0, 2])
    width = 0.26
    for i, (metric, label, color) in enumerate((("auc_full", "full", "#C0392B"),
                                                ("auc_base", "base", "#2E86C1"),
                                                ("auc_kan", "KAN", "#229954"))):
        v = [get(g, t, "population", metric) for g, t in combos]
        x = np.arange(4) + (i - 1) * width
        ax.bar(x, [a[0] for a in v], width=width, color=color, label=label)
        ax.errorbar(x, [a[0] for a in v], yerr=errs(v), fmt="none", ecolor="0.25",
                    elinewidth=0.7, capsize=1.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=6.3)
    ax.set_ylim(0.78, 1.10)
    ax.set_ylabel("AUC of the logit margin")
    ax.legend(frameon=False, fontsize=6.0, ncol=3, loc="upper center",
              handlelength=1.1, columnspacing=0.9)
    ax.set_title("Discrimination AUC", loc="left", pad=6)

    # (d) m_base vs m_KAN -------------------------------------------------- #
    ax = fig.add_subplot(gs[1, 0])
    lim = 16
    for g, color, marker in (("nitro", "#C0392B", "o"), ("nitriles", "#2E86C1", "s")):
        blk = per[(per.group == g) & (per.task_type == "multi")]
        blk = blk[blk["m_base"].abs() < blk["m_base"].abs().quantile(0.98)]
        ax.scatter(blk["m_base"], blk["m_kan"], s=6, alpha=0.55, color=color,
                   marker=marker, lw=0, label=C.GROUPS[g]["short"])
    ax.axhline(0, color="0.6", lw=0.7)
    ax.axvline(0, color="0.6", lw=0.7)
    ax.plot([-lim, lim], [-lim, lim], color="0.8", lw=0.8, ls=":")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$m_{2,1}$ base branch")
    ax.set_ylabel(r"$m_{2,1}$ Contribution-KAN")
    ax.legend(frameon=False, fontsize=6.4, loc="upper left", handletextpad=0.2,
              borderaxespad=0.2)
    ax.set_title("Multiplicity margin per spectrum", loc="left", pad=6, fontsize=9)

    # (e) evidence inside the literature bands ----------------------------- #
    ax = fig.add_subplot(gs[1, 1])
    if not enr.empty:
        width = 0.34
        for i, (view, label, color) in enumerate((
                ("binary_present_vs_absent", "binary", "#2E86C1"),
                ("multiplicity_c2plus_vs_c1", "multiplicity", "#C0392B"))):
            v = []
            for g in GROUPS:
                r = enr[(enr.group == g) & (enr.view == view)]
                v.append((100 * float(r.value.iloc[0]), 100 * float(r.ci_lo.iloc[0]),
                          100 * float(r.ci_hi.iloc[0])) if not r.empty else (np.nan,) * 3)
            x = np.arange(2) + (i - 0.5) * width
            ax.bar(x, [a[0] for a in v], width=width, color=color, label=label)
            ax.errorbar(x, [a[0] for a in v], yerr=errs(v), fmt="none", ecolor="0.25",
                        elinewidth=0.8, capsize=2)
        for j, g in enumerate(GROUPS):
            cov = C.characteristic_coverage(g)
            ax.hlines(cov, j - 0.44, j + 0.44, color="0.2", lw=1.0, ls="--")
            ax.text(j - 0.47, cov + 0.8, f"bands {cov:.1f}%", fontsize=5.6, color="0.2",
                    ha="left")
        ax.set_xticks(range(2))
        ax.set_xticklabels([C.GROUPS[g]["short"] for g in GROUPS])
        ax.set_ylabel("KAN evidence inside\nliterature bands (%)")
        ax.set_ylim(0, 26)
        ax.legend(frameon=False, fontsize=6.2, loc="upper right", handlelength=1.1)
        ax.set_title("Binary vs multiplicity evidence", loc="left", pad=6, fontsize=9)

    # (f) KAN vs peak similarity ------------------------------------------- #
    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(2)
    width = 0.34
    vals = {}
    for g in GROUPS:
        d = exp1_cache[g]["diag"]
        d = d[d.width_cm == 30.0]
        for c in ("kan_vs_peak", "kan_vs_gradcampp"):
            r = d[d.comparison == c]
            vals[(g, c)] = float(r.spearman_mean.iloc[0]) if not r.empty else np.nan
    for i, (c, label, color) in enumerate((("kan_vs_peak", "vs peak", "#7D3C98"),
                                          ("kan_vs_gradcampp", "vs Grad-CAM", "#2E86C1"))):
        ax.bar(x + (i - 0.5) * width, [vals[(g, c)] for g in GROUPS], width=width,
               color=color, label=label)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([C.GROUPS[g]["short"] for g in GROUPS])
    ax.set_ylabel(r"window-ranking $\rho$ per spectrum")
    ax.set_ylim(-0.16, 0.46)
    ax.legend(frameon=False, fontsize=6.2, loc="upper right", handlelength=1.1)
    ax.set_title("Is KAN just the strongest peaks?", loc="left", pad=6, fontsize=9)

    for i, ax in enumerate(fig.axes):
        ax.text(-0.30 if i % 3 == 0 else -0.20, 1.155, f"({'abcdef'[i]})",
                transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top")
    fig.savefig(FIG_DIR / "figY_branch_contribution.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figY_branch_contribution.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def table_x(exp1_by_group: dict, exp2: pd.DataFrame, enr: pd.DataFrame) -> pd.DataFrame:
    support = pd.read_csv(RES / "exp0" / "support_main_test.csv")
    released = released_per_class()
    rows = []

    for group in GROUPS:
        spec = C.GROUPS[group]
        exp1 = exp1_by_group[group]
        sup = support[support.group == group].iloc[0]
        task_stats = exp1["task"].set_index("scope")

        for task_type, task in (("binary", spec["binary"]), ("multi", spec["multi"])):
            pc = (released[released.task == task].set_index("class_id")
                  if released is not None else None)
            row = {
                "group": group, "task": task, "task_type": task_type,
                "heldout_support_c1": int(sup.n_c1),
                "heldout_support_c2plus": int(sup.n_c2plus),
                "heldout_compounds_c1": int(sup.n_compounds_c1),
                "heldout_compounds_c2plus": int(sup.n_compounds_c2plus),
                "released_f1_class1": (float(pc.loc[1, "f1"]) / 100.0
                                       if pc is not None else np.nan),
                "released_f1_class2": (float(pc.loc[2, "f1"]) / 100.0
                                       if pc is not None and 2 in pc.index else np.nan),
            }
            if task_type == "multi":
                row["analysis_set_n"] = int(task_stats.loc["analysis_set", "n"])
                row["analysis_set_acc_full"] = float(task_stats.loc["analysis_set", "acc_full"])
                row["analysis_set_acc_base"] = float(task_stats.loc["analysis_set", "acc_base"])
            else:
                row["analysis_set_n"] = int(sup.n_total)
                row["analysis_set_acc_full"] = float(task_stats.loc["main_test_all", "acc_full"])
                row["analysis_set_acc_base"] = float(task_stats.loc["main_test_all", "acc_base"])

            def br(stratum, metric):
                r = exp2[(exp2.group == group) & (exp2.task_type == task_type)
                         & (exp2.stratum == stratum) & (exp2.metric == metric)]
                return float(r.value.iloc[0]) if not r.empty else np.nan

            stratum = "all_present" if task_type == "multi" else "present"
            for k in ("auc_full", "auc_base", "auc_kan"):
                row[k] = br("population", k)
            for k in ("branch_ratio_median", "sign_agreement", "error_correction_rate",
                      "error_introduction_rate", "bias_share_of_m_kan_median"):
                row[k] = br(stratum, k)
            row["acc_kan_alone"] = br(stratum, "acc_kan_alone")

            dg = exp1["diag"]
            dg30 = dg[dg.width_cm == 30.0]
            def diag_val(name):
                r = dg30[dg30.comparison == name]
                return float(r.spearman_mean.iloc[0]) if not r.empty else np.nan
            row["kan_evidence_in_chem_bands_pct"] = (100 * diag_val("kan_mass_in_chem_bands")
                                                     if task_type == "multi" else np.nan)
            row["kan_vs_peak_window_spearman"] = diag_val("kan_vs_peak")
            row["kan_vs_gradcampp_window_spearman"] = diag_val("kan_vs_gradcampp")
            row["kan_vs_peak_topk_jaccard_4pct"] = diag_val("kan_vs_peak_topk_jaccard@4pct")

            if task_type == "multi":
                s = primary(exp1)
                for method in METHOD_ORDER:
                    blk = s[(s.method == method) & (s["metric"] == "D_mean")
                            & (s.coverage_pct == 4.0)]
                    row[f"D_4pct_{method}"] = float(blk.value.iloc[0]) if not blk.empty else np.nan
                a = exp1["auc"]
                a = a[(a.subset == "all") & (a.width_cm == 30.0) & (a.perturb == "interp")
                      & (a.renorm == False)]  # noqa: E712
                for method in METHOD_ORDER:
                    m = a[a.method == method]
                    row[f"D_auc_{method}"] = float(m.D_auc.iloc[0]) if not m.empty else np.nan
                for method in ("kan", "peak", "chem", "gradcampp"):
                    row[f"D_4pct_{method}_minus_random"] = (row[f"D_4pct_{method}"]
                                                            - row["D_4pct_random"])
                    row[f"D_auc_{method}_minus_random"] = (row[f"D_auc_{method}"]
                                                          - row["D_auc_random"])
                if row["D_4pct_random"] > 0:
                    row["D_4pct_kan_over_random"] = row["D_4pct_kan"] / row["D_4pct_random"]
                    row["D_4pct_peak_over_random"] = row["D_4pct_peak"] / row["D_4pct_random"]

                # operator B (baseline replacement) reference numbers
                auc_b = exp1["auc"]
                auc_b = auc_b[(auc_b.subset == "all") & (auc_b.width_cm == 30.0)
                              & (auc_b.perturb == "baseline") & (auc_b.renorm == False)]  # noqa: E712
                for method in ("kan", "peak", "random"):
                    m = auc_b[auc_b.method == method]
                    row[f"D_auc_{method}_operatorB"] = (float(m.D_auc.iloc[0])
                                                        if not m.empty else np.nan)

                # pure literature-band perturbation: largest coverage that is
                # still 100 % inside the characteristic bands
                cond = exp1["cond"]
                ch = cond[(cond.method == "chem") & (cond.width_cm == 30.0)
                          & (cond.perturb == "interp") & (cond.renorm == False)
                          & (cond.sel_inband_frac >= 0.999)]  # noqa: E712
                if not ch.empty:
                    pure = ch.sort_values("coverage_pct").iloc[-1]
                    sc = primary(exp1, subset="correct")
                    d_row = sc[(sc.method == "chem") & (sc.coverage_pct == pure.coverage_pct)]
                    row["chem_band_pure_coverage_pct"] = float(pure.frac_covered_pct)
                    row["D_chem_band_pure"] = float(
                        d_row[d_row["metric"] == "D_mean"].value.iloc[0])
                    row["flip_to_absent_chem_band_pure"] = float(
                        d_row[d_row["metric"] == "flip_rate"].value.iloc[0]
                        - d_row[d_row["metric"] == "flip_to_other_count_rate"].value.iloc[0])
                    row["flip_to_count_chem_band_pure"] = float(
                        d_row[d_row["metric"] == "flip_to_other_count_rate"].value.iloc[0])
                    ka = sc[(sc.method == "kan") & (sc.coverage_pct == pure.coverage_pct)]
                    row["D_kan_at_same_coverage"] = float(
                        ka[ka["metric"] == "D_mean"].value.iloc[0])

                # exp1b: the same manipulation seen through two different margins
                path = RES / "exp1" / "exp1b_presence_vs_count.csv"
                if path.is_file():
                    b = pd.read_csv(path)
                    b = b[(b.group == group) & (b.region == "literature_band")
                          & (b.perturb == "interp")].set_index("metric")
                    br = pd.read_csv(path)
                    br = br[(br.group == group) & (br.region.str.startswith("random"))
                            & (br.perturb == "interp")].groupby("metric").value.mean()
                    row["lit_band_axis_coverage_pct"] = float(
                        b.loc["count_margin_drop", "axis_coverage_pct"])
                    row["lit_band_presence_margin_drop"] = float(
                        b.loc["presence_margin_drop", "value"])
                    row["lit_band_count_margin_drop"] = float(
                        b.loc["count_margin_drop", "value"])
                    row["lit_band_flip_to_absent"] = float(
                        b.loc["flip_to_absent_rate", "value"])
                    row["lit_band_flip_to_other_count"] = float(
                        b.loc["flip_to_other_count_rate", "value"])
                    row["random_presence_margin_drop"] = float(br["presence_margin_drop"])
                    row["random_count_margin_drop"] = float(br["count_margin_drop"])
            else:
                if not enr.empty:
                    e = enr[(enr.group == group) & (enr.view == "binary_present_vs_absent")]
                    row["kan_evidence_in_chem_bands_pct"] = (
                        100 * float(e.value.iloc[0]) if not e.empty else np.nan)
            rows.append(row)

    table = pd.DataFrame(rows)
    table = table.rename(columns={"D_chem_band_pure": "D_chem_band_pure_correct_subset"})
    collapsed = {}
    for group in GROUPS:
        import json
        meta = json.loads((RES / "exp1" / f"exp1_{group}_meta.json").read_text())
        collapsed[group] = meta.get("zero_map_rate", {})
    table["gradcam_zero_map_frac"] = [collapsed[r.group].get("gradcam", np.nan)
                                      for r in table.itertuples()]

    table.to_csv(TAB_DIR / "tableX_summary.csv", index=False)

    md = ["| group | task | support c1 | support ≥2 | F1 c1 | F1 c≥2 | AUC full | AUC base "
          "| AUC KAN | branch ratio | sign agree | corrected | introduced |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        f2 = "-" if pd.isna(r["released_f1_class2"]) else f"{r['released_f1_class2']:.3f}"
        md.append(f"| {r['group']} | {r['task']} | {r['heldout_support_c1']} | "
                  f"{r['heldout_support_c2plus']} | {r['released_f1_class1']:.3f} | {f2} | "
                  f"{r['auc_full']:.3f} | {r['auc_base']:.3f} | {r['auc_kan']:.3f} | "
                  f"{r['branch_ratio_median']:.3f} | {r['sign_agreement']:.3f} | "
                  f"{r['error_correction_rate']:.3f} | {r['error_introduction_rate']:.4f} |")
    (TAB_DIR / "tableX_summary.md").write_text("\n".join(md) + "\n")
    return table


# --------------------------------------------------------------------------- #
exp1_cache: dict = {}

if __name__ == "__main__":
    C.ensure_dir(FIG_DIR)
    C.ensure_dir(TAB_DIR)
    exp1_cache.update({g: load_exp1(g) for g in GROUPS})
    exp2_metrics, exp2_enrichment = load_exp2()

    figure_x(exp1_cache)
    figure_x_supplement(exp1_cache)
    figure_x2_presence_vs_count()
    figure_y(exp2_metrics, exp2_enrichment)
    table = table_x(exp1_cache, exp2_metrics, exp2_enrichment)
    print(table.to_string())
    print(f"\n[out] {FIG_DIR}\n[out] {TAB_DIR}")