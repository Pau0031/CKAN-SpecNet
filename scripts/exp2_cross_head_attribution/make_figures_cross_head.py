#!/usr/bin/env python
"""Figures for the two newly added checks (teacher's items 3 and 4).

    figZ_binary_head      (a) nitro    : $D_{bin}$ of the *independent binary head*
                          (b) nitriles :   (recognition) vs perturbed fraction
                          (c) present -> absent flip rate of the binary head at 4 %
                          (d,e) the same perturbations read on the count margin of
                                the three-class head (multiplicity)
                          (f) recognition-minus-multiplicity contrast at 4 %
    figW_predicted_class  margin drop when the attribution is anchored on the
                          model's *predicted* class instead of the true class
                          (dotted = true-class read-out on the same spectra)

Everything is read from ``results/exp3`` and ``results/exp4``; no model is run.

Usage
-----
    python make_figures_exp3_exp4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``common.py`` (ensemble loading, analysis windows, the two perturbation
# operators, the cluster bootstrap) lives in the exp1 folder and is shared here.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "exp1_attribution_fidelity"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402

RES = C.RESULT_DIR
FIG_DIR = C.ensure_dir(RES / "figs")

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.2,
    "legend.fontsize": 7.0,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ORDER = ["kan_bin", "kan_cnt", "cam_bin", "peak", "chem", "random"]
SHORT = {"kan_bin": "KAN\nbin", "kan_cnt": "KAN\ncnt", "cam_bin": "CAM\nbin",
         "peak": "Peak", "chem": "Chem.", "random": "Rand."}
GROUP_COLORS = {"nitro": "#C0392B", "nitriles": "#2E86C1"}
LABELS = {
    "kan_bin": "KAN (binary head)",
    "kan_cnt": "KAN (count head)",
    "cam_bin": "Grad-CAM (binary head)",
    "peak": "Peak intensity",
    "chem": "Chem. bands",
    "random": "Random",
}
COLORS = {
    "kan_bin": "#E67E22",
    "kan_cnt": "#C0392B",
    "cam_bin": "#2E86C1",
    "peak": "#7D3C98",
    "chem": "#229954",
    "random": "#808080",
}
MARKERS = {"kan_bin": "o", "kan_cnt": "s", "cam_bin": "v", "peak": "^", "chem": "D",
           "random": "x"}
GROUPS = ["nitro", "nitriles"]
TAG_SHORT = {"nitro": "Nitro", "nitriles": "Nitriles"}


def curve_panel(ax, group: str, readout: str, ylabel: str, *, legend: bool) -> None:
    d = pd.read_csv(RES / "exp3" / f"exp3_{group}_summary.csv")
    d = d[(d.width_cm == 30.0) & (d.perturb == "interp") & (d.readout == readout)]
    for method in ORDER:
        sub = d[d.method == method].sort_values("coverage_pct")
        if sub.empty:
            continue
        xs = sub.coverage_pct.to_numpy()
        ax.plot(xs, sub.value.to_numpy(), marker=MARKERS[method], ms=3.0, lw=1.2,
                color=COLORS[method], label=LABELS[method],
                ls="--" if method == "random" else "-")
        ax.fill_between(xs, sub.ci_lo.to_numpy(), sub.ci_hi.to_numpy(),
                        color=COLORS[method], alpha=0.12, lw=0)
    ax.set_xlabel("perturbed fraction of the spectral axis (%)")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    if legend:
        ax.legend(frameon=False, loc="upper left", handlelength=1.5, borderaxespad=0.2)


def bar_pair(ax, values_per_group: dict, yerr_per_group: dict, *, ylabel: str,
             title: str, legend_loc: str = "upper left") -> None:
    width = 0.36
    xs = np.arange(len(ORDER))
    for i, group in enumerate(GROUPS):
        err = yerr_per_group[group]
        ax.bar(xs + (i - 0.5) * width, values_per_group[group], width, yerr=err, capsize=2,
               hatch="" if i == 0 else "///", alpha=1.0 if i == 0 else 0.9,
               edgecolor="#333333", lw=0.4, color=GROUP_COLORS[group],
               label=TAG_SHORT[group], error_kw={"lw": 0.8})
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[m] for m in ORDER], fontsize=6.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc=legend_loc)


def figure_z() -> None:
    fig = plt.figure(figsize=(11.4, 6.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.02], hspace=0.62, wspace=0.46)

    for col, group in enumerate(GROUPS):
        ax = fig.add_subplot(gs[0, col])
        curve_panel(ax, group, "bin",
                    "binary-head margin drop $D_{bin}$", legend=(col == 0))
        ax.set_title(f"({'ab'[col]}) {TAG_SHORT[group]}: recognition\n"
                     f"(independent binary head)", fontsize=8.8)

    # (c) present -> absent flip rate of the binary head at 4 %
    ax = fig.add_subplot(gs[0, 2])
    vals, errs = {}, {}
    for group in GROUPS:
        f = pd.read_csv(RES / "exp3" / f"exp3_{group}_flips.csv")
        f = f[(f.width_cm == 30.0) & (f.perturb == "interp")
              & (f.coverage_pct == 4.0) & (f.readout == "bin_present_to_absent")]
        f = f.groupby("method")["rate"].mean()
        vals[group] = [100 * float(f.get(m, np.nan)) for m in ORDER]
        errs[group] = None
    bar_pair(ax, vals, errs, ylabel="present $\\rightarrow$ absent flip rate (%)",
             title="(c) binary-head flips\nat 4 % coverage")

    for col, group in enumerate(GROUPS):
        ax = fig.add_subplot(gs[1, col])
        curve_panel(ax, group, "cnt", "count-margin drop $D_{cnt}$", legend=False)
        ax.set_title(f"({'de'[col]}) {TAG_SHORT[group]}: multiplicity\n(count head)",
                     fontsize=8.8)

    # (f) recognition minus multiplicity, same condition, per selection method
    ax = fig.add_subplot(gs[1, 2])
    vals2, errs2 = {}, {}
    for group in GROUPS:
        p = pd.read_csv(RES / "exp3" / f"exp3_{group}_paired.csv")
        p = p[(p.width_cm == 30.0) & (p.perturb == "interp")
              & (p.coverage_pct == 4.0) & (p.contrast == "D_bin_minus_D_cnt")]
        p = p.set_index("method")
        vals2[group] = [float(p.loc[m, "diff"]) if m in p.index else np.nan for m in ORDER]
        lo = [float(p.loc[m, "diff"]) - float(p.loc[m, "ci_lo"]) if m in p.index else 0.0
              for m in ORDER]
        hi = [float(p.loc[m, "ci_hi"]) - float(p.loc[m, "diff"]) if m in p.index else 0.0
              for m in ORDER]
        errs2[group] = np.array([lo, hi])
    bar_pair(ax, vals2, errs2, ylabel="$D_{bin} - D_{cnt}$ at 4 %",
             title="(f) recognition$-$multiplicity\nasymmetry at 4 %", legend_loc="upper right")

    fig.suptitle("Same perturbation budget, three margins read from one forward pass: the "
                 "independent binary head answers “present?”, the count head answers "
                 "“how many?”", fontsize=9.4, y=0.99)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"figZ_binary_head.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {FIG_DIR / 'figZ_binary_head.png'}")



def _load3b(group: str, name: str) -> pd.DataFrame:
    return pd.read_csv(RES / "exp3b" / f"exp3b_{group}_{name}.csv")


def figure_z2() -> None:
    """Present vs absent stratum, drawn in exactly the style of the three-class figure.

    Each curve panel shows the *same six selection methods* as the main attribution
    figure (Contribution-KAN on the count head, Contribution-KAN on the binary head,
    Grad-CAM, peak intensity, chemistry bands, random) so that the two figures can be
    read side by side; the two columns are the two strata of the experiment
    (present = count >= 1, absent = count = 0).
    """
    if not all((RES / "exp3b" / f"exp3b_{g}_summary.csv").exists() for g in GROUPS):
        print("[fig] figZ2 skipped (exp3b not run)")
        return

    order = ["kan_cnt", "kan_bin", "cam_bin", "peak", "chem", "random"]
    labels = {
        "kan_cnt": "Contribution-KAN (count head)",
        "kan_bin": "Contribution-KAN (binary head)",
        "cam_bin": "Grad-CAM (binary head)",
        "peak": "Peak intensity",
        "chem": "Chem. bands",
        "random": "Random",
    }

    fig = plt.figure(figsize=(11.8, 6.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.02], hspace=0.55, wspace=0.34)

    for row, group in enumerate(GROUPS):
        s = _load3b(group, "summary")
        s = s[(s.width_cm == 30.0) & (s.perturb == "interp") & (s.readout == "bin")]

        for col, stratum in enumerate(("present", "absent")):
            ax = fig.add_subplot(gs[row, col])
            for method in order:
                sub = s[(s.method == method) & (s.stratum == stratum)].sort_values("coverage_pct")
                if sub.empty:
                    continue
                xs = sub.coverage_pct.to_numpy()
                ax.plot(xs, sub.value.to_numpy(), marker=MARKERS[method], ms=3.0, lw=1.3,
                        color=COLORS[method], label=labels[method],
                        ls="--" if method == "random" else "-")
                ax.fill_between(xs, sub.ci_lo.to_numpy(), sub.ci_hi.to_numpy(),
                                color=COLORS[method], alpha=0.12, lw=0)
            ax.axhline(0, color="k", lw=0.6, alpha=0.45)
            ax.set_xlabel("perturbed fraction of the spectral axis (%)")
            ax.set_ylabel("presence-margin degradation $D_{bin}$")
            head = "present (count≥1)" if stratum == "present" else "absent (count=0)"
            letter = "abcd"[row * 2 + col]
            ax.set_title(f"({letter}) {TAG_SHORT[group]} — {head}", fontsize=8.6)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=5.9, loc="upper left",
                          handlelength=1.4, borderaxespad=0.2)

        # ---- 4 % coverage: present vs absent, one bar pair per method -------
        ax = fig.add_subplot(gs[row, 2])
        data4 = s[s.coverage_pct == 4.0]
        xs = np.arange(len(order))
        width = 0.36
        for i, (stratum, hatch) in enumerate((("present", ""), ("absent", "///"))):
            vals, err = [], []
            for m in order:
                sub = data4[(data4.method == m) & (data4.stratum == stratum)]
                if len(sub):
                    v = float(sub["value"].iloc[0])
                    vals.append(v)
                    err.append([v - float(sub.ci_lo.iloc[0]), float(sub.ci_hi.iloc[0]) - v])
                else:
                    vals.append(np.nan)
                    err.append([0, 0])
            ax.bar(xs + (i - 0.5) * width, vals, width, yerr=np.array(err).T, capsize=2,
                   hatch=hatch, edgecolor="#333333", lw=0.4,
                   color=GROUP_COLORS[group], alpha=1.0 if i == 0 else 0.9,
                   label=("present (count≥1)" if stratum == "present" else "absent (count=0)"),
                   error_kw={"lw": 0.8})
        ax.set_xticks(xs)
        ax.set_xticklabels(["KAN\ncnt", "KAN\nbin", "Grad-\nCAM", "Peak", "Chem.", "Rand."],
                           fontsize=6.2)
        ax.set_ylabel("$D_{bin}$ at 4 %")
        ax.axhline(0, color="k", lw=0.7)
        ax.margins(y=0.18)
        ax.set_title(f"({'ef'[row]}) {TAG_SHORT[group]}: 4 % coverage", fontsize=8.6)
        ax.legend(frameon=False, loc="upper right", fontsize=6.2)

    fig.suptitle("The same perturbation, read on the network's presence margin "
                 "(independent binary head $o_1-o_0$), separately for the spectra that "
                 "contain the group and those that do not", fontsize=9.2, y=0.99)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"figZ2_present_vs_absent.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {FIG_DIR / 'figZ2_present_vs_absent.png'}")



if __name__ == "__main__":
    figure_z()
    figure_z2()
