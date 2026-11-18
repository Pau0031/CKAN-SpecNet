"""Infrared band definitions and helpers shared by the interpretability scripts.

Extracted from ``scripts/compute_enrichment.py`` so that the enrichment
analysis and the KAN-vs-Grad-CAM comparison use one single source of truth.

Region table rules (as in the paper): explicit ranges as-is; ``~X`` -> (X+5, X-5);
``X +- 20`` -> (X+20, X-20); all clipped to COMMON_X ``[552, 3842]``; overlapping
or adjacent regions (gap <= 30 cm^-1) merged.  Each tuple is
``(high_wavenumber, low_wavenumber)``.
"""

from __future__ import annotations

import numpy as np

from ckan_specnet.core import COMMON_X

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


# public aliases (the comparison scripts should not import underscored names)
task_name = _task_name
transitions = _transitions
display_name = _display_name
regions_description = _regions_description

ALL_GROUPS: list[str] = QUATERNARY_GROUPS + TERNARY_GROUPS + BINARY_GROUPS
