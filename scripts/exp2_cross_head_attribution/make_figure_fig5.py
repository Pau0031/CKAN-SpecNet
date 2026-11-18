#!/usr/bin/env python
"""Fig. 5 - standardized effect sizes and flip destinations (for readme3.md).

Panel (a)  D_z of the same perturbation, in units of each head's own margin SD,
           so that the binary (recognition) and the count (multiplicity) heads
           become comparable: chemistry bands, Contribution-KAN and peak
           intensity, each shown on both heads.
Panel (b)  probability scale: among the present spectra whose prediction changes,
           how much of the change lands on "absent" and how much on the other
           count class.

Reads ``results/exp5`` only; nothing is recomputed here.

Usage
-----
    python make_figure_fig5.py
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
    "axes.titlesize": 9.4,
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

GROUPS = ["nitro", "nitriles"]
GROUP_LABEL = {"nitro": "Nitro", "nitriles": "Nitriles"}
# method -> (label, own map on the binary head, own map on the count head)
METHODS = [
    ("chem", "Chem. bands", "#229954"),
    ("kan", "Contribution-KAN", "#C0392B"),
    ("peak", "Peak intensity", "#7D3C98"),
]


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eff = pd.read_csv(RES / "exp5" / "exp5_effects.csv")
    con = pd.read_csv(RES / "exp5" / "exp5_head_contrast.csv")
    flp = pd.read_csv(RES / "exp5" / "exp5_flips.csv")
    eff = eff[(eff.coverage_pct == 4.0) & (eff.stratum == "present")]
    con = con[con.coverage_pct == 4.0]
    flp = flp[(flp.coverage_pct == 4.0) & (flp.stratum == "present")]
    return eff, con, flp


def dz(eff: pd.DataFrame, group: str, head: str, method: str, col: str) -> float:
    key = "kan_bin" if head == "bin" else "kan_cnt"
    m = method if method != "kan" else key
    row = eff[(eff["group"] == group) & (eff["head"] == head) & (eff["method"] == m)]
    return float(row[col].iloc[0])


def panel_a(ax, eff: pd.DataFrame, con: pd.DataFrame) -> None:
    bar_w = 0.46
    head_offsets = {"bin": [0.0, 0.56, 1.12], "cnt": [1.95, 2.51, 3.07]}
    group_base = {"nitro": 0.0, "nitriles": 5.2}

    for group in GROUPS:
        base = group_base[group]
        for head in ("bin", "cnt"):
            for (method, label, color), off in zip(METHODS, head_offsets[head]):
                x = base + off
                v = dz(eff, group, head, method, "D_z")
                lo = dz(eff, group, head, method, "D_z_lo")
                hi = dz(eff, group, head, method, "D_z_hi")
                ax.bar(x, v, bar_w, color=color, alpha=1.0 if head == "bin" else 0.55,
                       hatch="" if head == "bin" else "///", edgecolor="#333333", lw=0.5,
                       yerr=[[v - lo], [hi - v]], capsize=2.2,
                       error_kw={"lw": 0.8, "ecolor": "#333333"})
                ax.annotate(f"{v:.2f}", xy=(x, hi), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=5.8, color="#333333")
            # significance of the recognition-vs-multiplicity contrast of each method
            for (method, _, _), xb, xc in zip(METHODS, head_offsets["bin"], head_offsets["cnt"]):
                row = con[(con["group"] == group) & (con["head"] == "bin")
                          & (con["method"] == {"chem": "chem", "kan": "kan_bin",
                                               "peak": "peak"}[method])]
                if not len(row):
                    continue
                p = float(row["diff_p"].iloc[0])
                star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
                y = max(dz(eff, group, "bin", method, "D_z_hi"),
                        dz(eff, group, "cnt", method, "D_z_hi"))
                ax.annotate(star, xy=(base + (xb + xc) / 2, y), xytext=(0, 13),
                            textcoords="offset points", ha="center", fontsize=8.4,
                            color="#111111")

    # separators between the two heads of a group
    for group in GROUPS:
        base = group_base[group]
        ax.axvline(base + 1.55, color="#BBBBBB", lw=0.7)

    ticks = [0.56, 2.51, 5.76, 7.71]
    labels = ["recognition\n(binary head)", "multiplicity\n(count head)"] * 2
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=6.8)
    for group, centre in (("nitro", 1.55), ("nitriles", 6.75)):
        ax.annotate(GROUP_LABEL[group], xy=(centre, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -34), textcoords="offset points", ha="center",
                    fontsize=8.8, fontweight="bold")
    ax.set_xlim(-0.6, group_base["nitriles"] + 3.6)
    ax.set_ylim(0, 3.25)
    ax.set_ylabel("effect size  $D_z$  (units of the head's own margin SD)")
    ax.set_title("(a) one perturbation, two heads, each on its own scale", pad=8)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#333333", lw=0.5, label=l)
               for _, l, c in METHODS]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="#DDDDDD", edgecolor="#333333", lw=0.5,
                              label="solid: recognition head"),
                plt.Rectangle((0, 0), 1, 1, facecolor="#DDDDDD", edgecolor="#333333",
                              hatch="///", lw=0.5, label="hatched: count head")]
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=1, handlelength=1.3,
              borderaxespad=0.4, labelspacing=0.35)


def panel_b(ax, flp: pd.DataFrame) -> None:
    bar_w = 0.62
    xs = {"nitro": [0.0, 1.0], "nitriles": [2.6, 3.6]}
    to_absent, to_count = "#4D4D4D", "#E67E22"
    note_x = {"nitro": 0.5, "nitriles": 3.1}

    for group in GROUPS:
        for i, head in enumerate(("bin", "multi")):
            x = xs[group][i]
            if head == "bin":
                v = float(flp[(flp["group"] == group) & (flp["head"] == "bin")
                              & (flp["method"] == "chem")]["flip_rate"].iloc[0]) * 100
                ax.bar(x, v, bar_w, color=to_absent, edgecolor="#222222", lw=0.5)
                ax.annotate(f"{v:.1f} %", xy=(x, v), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=6.6)
                ax.annotate("all → absent", xy=(x, v / 2), ha="center", fontsize=5.8,
                            color="white", rotation=90, va="center")
            else:
                sub = flp[(flp["group"] == group) & (flp["head"] == "multi")
                          & (flp["method"] == "chem")].set_index("metric")
                a = float(sub.loc["flip_to_absent", "flip_rate"]) * 100
                b = float(sub.loc["flip_to_other_count", "flip_rate"]) * 100
                ax.bar(x, a, bar_w, color=to_absent, edgecolor="#222222", lw=0.5)
                ax.bar(x, b, bar_w, bottom=a, color=to_count, edgecolor="#222222", lw=0.5,
                       hatch="//")
                tot = a + b
                ax.annotate(f"{tot:.1f} %", xy=(x, tot), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=6.6)
                ax.annotate(f"{a:.1f}", xy=(x, a / 2), ha="center", fontsize=6.4,
                            color="white", va="center")
                ax.annotate(f"{b:.1f}", xy=(x + bar_w / 2 + 0.05, a + b / 2), ha="left",
                            fontsize=6.4, color="#B9770E", va="center")
                ax.annotate(f"{100 * b / tot:.1f} % of all changes", xy=(note_x[group], tot),
                            xytext=(0, 15), textcoords="offset points", ha="center",
                            fontsize=6.0, color="#B9770E")
        r_bin = float(flp[(flp["group"] == group) & (flp["head"] == "bin")
                          & (flp["method"] == "random")]["flip_rate"].iloc[0]) * 100
        r_cnt = float(flp[(flp["group"] == group) & (flp["head"] == "multi")
                          & (flp["method"] == "random")
                          & (flp["metric"] == "flip_any")]["flip_rate"].iloc[0]) * 100
        for x, v in zip(xs[group], (r_bin, r_cnt)):
            ax.hlines(v, x - bar_w / 2 - 0.08, x + bar_w / 2 + 0.08, color="#7F7F7F",
                      lw=1.3, ls="--")
        ax.annotate(f"random control {r_bin:.1f} / {r_cnt:.1f} %",
                    xy=(note_x[group], 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -46), textcoords="offset points", ha="center",
                    fontsize=5.8, color="#555555")

    ax.set_xticks([0.5, 3.1])
    ax.set_xticklabels(["Nitro", "Nitriles"], fontsize=8.6, fontweight="bold")
    for x, lab in ((0.0, "binary\nhead"), (1.0, "count\nhead"),
                   (2.6, "binary\nhead"), (3.6, "count\nhead")):
        ax.annotate(lab, xy=(x, 0), xycoords=("data", "axes fraction"), xytext=(0, -32),
                    textcoords="offset points", ha="center", fontsize=6.4, color="#333333")
    ax.set_ylabel("changed predictions\n(% of the present spectra)")
    ax.set_title("(b) where those changed predictions go", pad=8)
    ax.set_ylim(0, 70)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=to_absent, edgecolor="#222222",
                                     label="→ absent (loses the group)"),
                       plt.Rectangle((0, 0), 1, 1, facecolor=to_count, edgecolor="#222222",
                                     hatch="//", label="→ other count class")],
              frameon=False, loc="upper left", handlelength=1.3, borderaxespad=0.4)


def main() -> None:
    eff, con, flp = load()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0),
                             gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.24})
    panel_a(axes[0], eff, con)
    panel_b(axes[1], flp)
    fig.subplots_adjust(bottom=0.22, top=0.9)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig5_standardized_effects_and_flips.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {FIG_DIR / 'fig5_standardized_effects_and_flips.png'}")


if __name__ == "__main__":
    main()
