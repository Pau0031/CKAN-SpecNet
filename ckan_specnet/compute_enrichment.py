#!/usr/bin/env python
"""Compute enrichment factors for 21 functional groups across class transitions.

Enrichment = (evidence fraction within chemically relevant IR regions)
           / (spectral-axis coverage of those regions)

Output: data/enrichment_result.csv
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from ckan_specnet.core import COMMON_X, Config, EVAL_NAME_COL, configure_torch_runtime, device_of
from ckan_specnet.data import frame_to_xy
from ckan_specnet.eval import load_manifest, tasks_from_manifest
from ckan_specnet.plot import (
    configure_matplotlib,
    get_kan_contribution_sample,
    get_transition_evidence,
    load_plot_ensemble,
    predict_plot_ensemble,
)

# ── IR regions per functional group ──────────────────────────────────────────
# Parsed from data/functional_groups.md
# Rules: explicit ranges as-is; ~X → (X+5, X-5); X ± 20 → (X+20, X-20)
# All clipped to COMMON_X range [552, 3842]; overlapping/adjacent (gap≤30) merged.
# Each tuple: (high_wavenumber, low_wavenumber)

FUNCTIONAL_GROUP_REGIONS: dict[str, list[tuple[float, float]]] = {
    "alkane": [
        (2975, 2915),
        (2885, 2840),
        (1475, 1435),
        (1385, 1370),
        (1260, 700),
    ],
    "alkene": [
        (3095, 3000),
        (1680, 1580),
        (995, 960),
        (915, 905),
        (850, 790),
        (730, 650),
    ],
    "alkyne": [
        (3340, 3300),
        (2260, 2190),
        (2150, 2100),
        (730, 575),
    ],
    "aromatics": [
        (3100, 3000),
        (2000, 1660),
        (1620, 1560),
        (1520, 1430),
        (1300, 1000),
        (900, 650),
    ],
    "esters": [
        (1750, 1700),
        (1310, 1210),
        (1120, 1020),
    ],
    "ketones": [
        (1745, 1660),
        (1300, 1000),
    ],
    "ortho": [
        (2000, 1660),
        (1605, 1575),
        (1505, 1495),
        (1455, 1445),
        (790, 720),
    ],
    "meta": [
        (2000, 1660),
        (1605, 1575),
        (1505, 1495),
        (1455, 1445),
        (880, 830),
        (725, 680),
    ],
    "para": [
        (2000, 1660),
        (1605, 1575),
        (1505, 1495),
        (1455, 1445),
        (860, 780),
    ],
    "alkyl_halides": [
        (1400, 1000),
        (800, 552),
    ],
    "alcohols": [
        (3670, 2500),
        (1440, 1260),
        (1205, 1000),
    ],
    "ether": [
        (1310, 820),
    ],
    "amines": [
        (3550, 3250),
        (1650, 1580),
        (1360, 1020),
        (895, 650),
    ],
    "carbonyl_oxygen": [
        (1850, 1650),
    ],
    "aldehydes": [
        (2900, 2800),
        (2745, 2650),
        (1740, 1680),
        (1440, 1320),
    ],
    "acyl_halides": [
        (1900, 1765),
        (900, 800),
    ],
    "amides": [
        (3540, 3480),
        (3420, 3270),
        (1700, 1620),
        (1570, 1515),
        (1305, 1200),
        (770, 620),
    ],
    "nitriles": [
        (2260, 2200),
    ],
    "nitro": [
        (1560, 1500),
        (1380, 1290),
        (920, 800),
    ],
    "isocyanate": [
        (2300, 2250),
        (1420, 1340),
        (650, 580),
    ],
    "isothiocyanate": [
        (2150, 1990),
        (1000, 900),
    ],
}

# ── Task priority mapping ────────────────────────────────────────────────────
# Priority: 4-class > 3-class > binary
# Each entry: (task_name, num_classes, transitions)
# transitions: list of (reference_class, target_class)

QUATERNARY_GROUPS = ["alkyl_halides", "alcohols", "ether", "amines", "carbonyl_oxygen"]
TERNARY_GROUPS = [
    "aldehydes", "acyl_halides", "amides", "nitriles", "nitro",
    "isocyanate", "isothiocyanate",
]
# Binary groups = all 21 - quaternary - ternary
BINARY_GROUPS = [
    "alkane", "alkene", "alkyne", "aromatics", "esters", "ketones",
    "ortho", "meta", "para",
]


def _task_name(group: str) -> str:
    """Return the priority task name for a functional group."""
    if group in QUATERNARY_GROUPS:
        return f"{group}_4class"
    if group in TERNARY_GROUPS:
        return f"{group}_3class"
    return group  # binary


def _transitions(group: str) -> list[tuple[int, int]]:
    """Return list of (ref_class, target_class) transitions for a group."""
    if group in QUATERNARY_GROUPS:
        return [(0, 1), (1, 2), (2, 3)]
    if group in TERNARY_GROUPS:
        return [(0, 1), (1, 2)]
    return [(0, 1)]


def _display_name(group: str) -> str:
    """Human-readable functional group name."""
    names: dict[str, str] = {
        "alkane": "Alkane",
        "alkene": "Alkene",
        "alkyne": "Alkyne",
        "aromatics": "Aromatics",
        "esters": "Esters",
        "ketones": "Ketones",
        "ortho": "Ortho",
        "meta": "Meta",
        "para": "Para",
        "alkyl_halides": "Alkyl halides",
        "alcohols": "Alcohols",
        "ether": "Ether",
        "amines": "Amines",
        "carbonyl_oxygen": "Carbonyl oxygen",
        "aldehydes": "Aldehydes",
        "acyl_halides": "Acyl halides",
        "amides": "Amides",
        "nitriles": "Nitriles",
        "nitro": "Nitro",
        "isocyanate": "Isocyanate",
        "isothiocyanate": "Isothiocyanate",
    }
    return names.get(group, group)


def _regions_description(group: str) -> str:
    """Human-readable description of chemically relevant regions."""
    regions = FUNCTIONAL_GROUP_REGIONS.get(group, [])
    parts = [f"{s}-{e} cm⁻¹" for s, e in regions]
    return "; ".join(parts)


def build_region_mask(regions: list[tuple[float, float]]) -> np.ndarray:
    """Build a boolean mask over COMMON_X indicating which points fall in any region."""
    mask = np.zeros(len(COMMON_X), dtype=bool)
    for start, end in regions:
        # start > end (high wavenumber first)
        in_region = (COMMON_X <= start) & (COMMON_X >= end)
        mask |= in_region
    return mask


def compute_spectral_coverage(regions: list[tuple[float, float]]) -> float:
    """Compute fraction of the spectral axis covered by (merged) regions.

    Returns a value in [0, 1].
    """
    if not regions:
        return 0.0

    total_points = len(COMMON_X)
    mask = build_region_mask(regions)
    return float(mask.sum()) / total_points


def compute_evidence_fraction(evidence: np.ndarray, region_mask: np.ndarray) -> float:
    """Compute fraction of total evidence that falls within the region mask.

    evidence: 1-D array of transition evidence over COMMON_X.
    Returns a value in [0, 1].
    """
    total = float(np.sum(evidence))
    if total < 1e-12:
        return 0.0
    in_regions = float(np.sum(evidence[region_mask]))
    return in_regions / total


def select_correct_samples(
    y_true: dict[str, np.ndarray],
    pred_result: dict,
    task: str,
    cls: int,
    max_samples: int,
) -> np.ndarray:
    """Return indices of correctly predicted samples for a given class, ranked by confidence."""
    true = np.asarray(y_true[task])
    pred = np.asarray(pred_result["y_pred"][task])
    correct_idx = np.where((true == cls) & (pred == cls))[0]

    if len(correct_idx) == 0:
        return np.array([], dtype=np.int64)

    # Rank by prediction confidence (probability of the correct class)
    confidence = pred_result["prob"][task][correct_idx, cls]
    order = np.argsort(-confidence)
    n = min(max_samples, len(order))
    return correct_idx[order[:n]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute enrichment factors for 21 functional groups."
    )
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-name", type=str, default="main_test")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Max samples per class per functional group.")
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure output directory exists
    args.out.parent.mkdir(parents=True, exist_ok=True)

    configure_torch_runtime()
    configure_matplotlib()

    print("Loading manifest and tasks...")
    manifest = load_manifest(args.run_dir)
    tasks = tasks_from_manifest(manifest)

    print(f"Loading test data: {args.test}")
    test_df = pl.read_parquet(args.test)
    subset = test_df.filter(pl.col(EVAL_NAME_COL) == args.eval_name)
    if subset.height == 0:
        raise ValueError(f"Empty eval subset: {args.eval_name}")

    x, y, _ = frame_to_xy(subset, tasks)
    print(f"Eval subset '{args.eval_name}': {len(x)} samples")

    cfg = replace(
        Config(),
        test_parquet=args.test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    device = device_of()
    print(f"Device: {device}")

    print("Loading 5-fold ensemble...")
    loaded = load_plot_ensemble(run_dir=args.run_dir, tasks=tasks)

    print("Running ensemble prediction...")
    pred_result = predict_plot_ensemble(
        models=loaded["models"],
        preprocessors=loaded["preprocessors"],
        x=x,
        tasks=tasks,
        cfg=cfg,
        device=device,
    )

    # ── Compute enrichment for each functional group ─────────────────────────
    all_groups = QUATERNARY_GROUPS + TERNARY_GROUPS + BINARY_GROUPS
    rows: list[dict] = []

    for group in all_groups:
        task = _task_name(group)
        transitions = _transitions(group)
        regions = FUNCTIONAL_GROUP_REGIONS.get(group, [])
        region_mask = build_region_mask(regions)
        spectral_coverage = compute_spectral_coverage(regions)

        print(f"\n{'='*60}")
        print(f"Group: {_display_name(group)}  |  Task: {task}  |  "
              f"Regions: {len(regions)}  |  Coverage: {spectral_coverage*100:.1f}%")
        print(f"{'='*60}")

        for ref_cls, tgt_cls in transitions:
            # Select correctly predicted samples at the target class level
            sample_indices = select_correct_samples(
                y_true=y,
                pred_result=pred_result,
                task=task,
                cls=tgt_cls,
                max_samples=args.n_samples,
            )

            if len(sample_indices) == 0:
                print(f"  C{ref_cls}→C{tgt_cls}: NO correct samples found for class {tgt_cls}!")
                rows.append({
                    "Functional group": _display_name(group),
                    "Transition": f"C{ref_cls}→C{tgt_cls}",
                    "Chemically relevant regions": _regions_description(group),
                    "Spectral-axis coverage (%)": round(spectral_coverage * 100, 1),
                    "Evidence within regions (%)": "N/A",
                    "Enrichment": "N/A",
                    "N samples": 0,
                })
                continue

            enrichments = []
            evidence_fractions = []

            for idx in sample_indices:
                try:
                    sample = get_kan_contribution_sample(
                        models=loaded["models"],
                        preprocessors=loaded["preprocessors"],
                        x=x,
                        y=y,
                        pred_result=pred_result,
                        task=task,
                        sample_idx=int(idx),
                        device=device,
                    )
                except Exception as exc:
                    print(f"  [WARN] sample {idx}: KAN contribution failed: {exc}")
                    continue

                # Compute transition evidence C_{ref} → C_{tgt}
                try:
                    evidence = get_transition_evidence(
                        sample=sample,
                        target_class=tgt_cls,
                        reference_class=ref_cls,
                        smooth_window=args.smooth_window,
                    )
                except Exception as exc:
                    print(f"  [WARN] sample {idx}: transition evidence failed: {exc}")
                    continue

                ef = compute_evidence_fraction(evidence, region_mask)
                evidence_fractions.append(ef)

                if spectral_coverage > 0 and ef > 0:
                    enrichments.append(ef / spectral_coverage)

            n_valid = len(evidence_fractions)
            if n_valid == 0:
                print(f"  C{ref_cls}→C{tgt_cls}: all {len(sample_indices)} samples failed!")
                rows.append({
                    "Functional group": _display_name(group),
                    "Transition": f"C{ref_cls}→C{tgt_cls}",
                    "Chemically relevant regions": _regions_description(group),
                    "Spectral-axis coverage (%)": round(spectral_coverage * 100, 1),
                    "Evidence within regions (%)": "N/A",
                    "Enrichment": "N/A",
                    "N samples": 0,
                })
                continue

            mean_ef = float(np.mean(evidence_fractions))
            mean_enrich = float(np.mean(enrichments)) if enrichments else float("nan")

            print(f"  C{ref_cls}→C{tgt_cls}: {n_valid} samples | "
                  f"evidence_in_regions={mean_ef*100:.1f}% | "
                  f"enrichment={mean_enrich:.2f}")

            rows.append({
                "Functional group": _display_name(group),
                "Transition": f"C{ref_cls}→C{tgt_cls}",
                "Chemically relevant regions": _regions_description(group),
                "Spectral-axis coverage (%)": round(spectral_coverage * 100, 1),
                "Evidence within regions (%)": round(mean_ef * 100, 1),
                "Enrichment": round(mean_enrich, 2),
                "N samples": n_valid,
            })

    # ── Write CSV ────────────────────────────────────────────────────────────
    result_df = pl.DataFrame(rows)
    result_df.write_csv(args.out,encoding="utf-8-sig")
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.out}")
    print(f"Total rows: {result_df.height}")
    print(result_df)


if __name__ == "__main__":
    main() 
