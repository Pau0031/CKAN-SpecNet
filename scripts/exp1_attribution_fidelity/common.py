"""Shared machinery for the revision experiments (exp1 + exp2).

exp1  Spectral Attribution Fidelity
      Band-perturbation faithfulness of Contribution-KAN vs Grad-CAM vs
      conventional peak intensity vs chemistry-anchored bands vs random bands,
      for the count = 1  vs  count >= 2 discrimination of nitro / nitriles.

exp2  Contribution-KAN branch contribution
      Exact logit decomposition  m_full = m_base + m_KAN , direction agreement,
      error correction / introduction, class dependence, for both the binary
      (present vs absent) and the multiplicity (count>=2 vs count=1) task.

Everything here is read-only with respect to the trained ensemble in
``models/`` (five folds, ``normalize=sample_zscore``); no model is trained
and no hyper-parameter is re-selected.

Conventions used throughout
---------------------------
GRID            1646 points, 552-3842 cm^-1, 2 cm^-1 step (``COMMON_X``).
count_key       "c1"  -> true count == 1        (label class 1)
                "c2+" -> true count >= 2        (label class 2, clipped)
support sign s  +1 for c2+ samples, -1 for c1 samples, so that

                    m_y(x) = s * (o_2(x) - o_1(x))

                is the *true-class-supported* count margin (positive = the model
                supports the true count class).  D(k) = m_y(x) - m_y(x_pert) is
                therefore the margin degradation we want to be large.

Attribution maps are always aligned to the same convention: a map value is
"evidence supporting the true count class relative to the other count class".
For the KAN branch that is  s * (M_2(i) - M_1(i)); for Grad-CAM it is the CAM of
the score  s * (o_2 - o_1)  taken on the fused (base + KAN) forward pass.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ``ikan`` (imported through ckan_specnet.model) changes the process cwd at
# import time, so every path below is absolute before any heavy import.
EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
RESULT_DIR = ROOT / "results"
RUN_DIR = ROOT / "models"
TEST_PARQUET = ROOT / "data" / "test.parquet"
TRAIN_PARQUET = ROOT / "data" / "train.parquet"
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from ckan_specnet.core import COMMON_X, EVAL_NAME_COL, configure_torch_runtime  # noqa: E402
from ckan_specnet.data import Preprocessor, frame_to_xy  # noqa: E402
from ckan_specnet.eval import (  # noqa: E402
    load_fold_model,
    load_fold_preprocessors,
    load_manifest,
    tasks_from_manifest,
)
from ckan_specnet.gradcam1d import GradCAM1D  # noqa: E402
from ckan_specnet.regions import FUNCTIONAL_GROUP_REGIONS  # noqa: E402

GRID = np.asarray(COMMON_X, dtype=np.float32)
EVAL_NAME = "main_test"
STEP_CM = float(np.median(np.diff(GRID)))          # 2.0 cm^-1
AXIS_SPAN = float(GRID[-1] - GRID[0])              # 3290 cm^-1

#: analysis groups: structural name -> task names + display label
GROUPS: dict[str, dict] = {
    "nitro": {
        "binary": "nitro",
        "multi": "nitro_3class",
        "column": "nitro",
        "display": "Nitro (NO$_2$)",
        "short": "Nitro",
    },
    "nitriles": {
        "binary": "nitriles",
        "multi": "nitriles_3class",
        "column": "nitriles",
        "display": "Nitriles (C$\\equiv$N)",
        "short": "Nitriles",
    },
}

METHOD_LABELS = {
    "kan": "Contribution-KAN",
    # The CAM baseline is labelled "Grad-CAM" in the figures; its channel weights
    # follow Grad-CAM++ (the plain Grad-CAM weighting collapses on this backbone).
    "gradcampp": "Grad-CAM",
    "gradcam": "Grad-CAM (plain weighting)",
    "peak": "Peak intensity",
    "chem": "Chem. bands",
    "random": "Random",
}

METHOD_COLORS = {
    "kan": "#C0392B",
    "gradcampp": "#2E86C1",
    "gradcam": "#7FB3D5",
    "peak": "#7D3C98",
    "chem": "#229954",
    "random": "#808080",
}

METHOD_MARKERS = {
    "kan": "o",
    "gradcampp": "s",
    "gradcam": "v",
    "peak": "^",
    "chem": "D",
    "random": "x",
}


# --------------------------------------------------------------------------- #
# ensemble loading / forward helpers
# --------------------------------------------------------------------------- #
@dataclass
class Ensemble:
    """Five-fold ensemble kept resident on one GPU plus its metadata."""

    run_dir: Path
    models: list[nn.Module]
    preprocessors: list[Preprocessor]
    manifest: dict
    tasks: object
    device: torch.device
    fold_files: list[str] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.models)

    def alpha(self, task: str) -> np.ndarray:
        return np.array([float(m.contrib.alpha[task].detach().cpu()) for m in self.models])

    def preprocess(self, raw: np.ndarray, fold: int = 0) -> np.ndarray:
        pre = self.preprocessors[fold]
        return np.stack([pre(row) for row in raw]).astype(np.float32)


def load_ensemble(device: torch.device, run_dir: Path = RUN_DIR, folds: list[int] | None = None) -> Ensemble:
    configure_torch_runtime()
    manifest = load_manifest(run_dir)
    tasks = tasks_from_manifest(manifest)
    fold_files = list(manifest["ensemble"]["fold_files"])
    if folds:
        fold_files = [fold_files[i] for i in folds]

    preprocessors = load_fold_preprocessors(run_dir, manifest)
    if folds:
        preprocessors = [preprocessors[i] for i in folds]

    modes = {p.mode for p in preprocessors}
    if not modes <= {"sample_zscore", "none"}:
        raise RuntimeError(f"perturbation engine assumes a stateless preprocessor, got {modes}")

    models = []
    for name in fold_files:
        model, _ = load_fold_model(run_dir / name, tasks)
        models.append(model.to(device).eval())

    return Ensemble(
        run_dir=run_dir,
        models=models,
        preprocessors=preprocessors,
        manifest=manifest,
        tasks=tasks,
        device=device,
        fold_files=fold_files,
    )


def fold_logits(ens: Ensemble, X: np.ndarray, task: str, logit_mode: str = "full",
                batch_size: int = 512) -> np.ndarray:
    """Per-fold logits ``(n_folds, N, n_classes)`` for raw (preprocessed) input ``X``."""
    out = None
    for f, model in enumerate(ens.models):
        chunks = []
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=ens.device)
            with torch.inference_mode():
                logits = model(xb, logit_mode=logit_mode)[task]
            chunks.append(logits.detach().float().cpu().numpy())
        fold = np.concatenate(chunks, axis=0)
        if out is None:
            out = np.zeros((ens.n_folds,) + fold.shape, dtype=np.float32)
        out[f] = fold
    return out


def fold_logits_tasks(ens: Ensemble, X: np.ndarray, tasks: list[str],
                      logit_mode: str = "full",
                      batch_size: int = 512) -> dict[str, np.ndarray]:
    """Per-fold logits for *several* tasks from a **single** forward pass.

    Returns ``{task: (n_folds, N, n_classes_task)}``.  The multi-task CKAN-SpecNet
    computes every head in one pass, so reading the independent binary head
    (``nitro`` / ``nitriles``) together with the multiplicity head
    (``*_3class``) costs exactly one forward per fold.
    """
    chunks: dict[str, list[list[np.ndarray]]] = {t: [] for t in tasks}
    for model in ens.models:
        per_task: dict[str, list[np.ndarray]] = {t: [] for t in tasks}
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=ens.device)
            with torch.inference_mode():
                logits = model(xb, logit_mode=logit_mode)
            for t in tasks:
                per_task[t].append(logits[t].detach().float().cpu().numpy())
        for t in tasks:
            chunks[t].append(np.concatenate(per_task[t], axis=0))
    return {t: np.stack(v, axis=0) for t, v in chunks.items()}


def presence_margin(logits: np.ndarray) -> np.ndarray:
    """``logsumexp(o_present) - o_absent`` on a three-class head."""
    present = logits[..., 1:]
    m = present.max(axis=-1, keepdims=True)
    lse = (m + np.log(np.exp(present - m).sum(axis=-1, keepdims=True)))[..., 0]
    return lse - logits[..., 0]


def fold_logits_and_bias(ens: Ensemble, X: np.ndarray, task: str,
                         batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Per-fold ``full`` logits and per-fold KAN bias term ``b(z)`` (both ``(F,N,K)``).

    ``m_KAN = sum_l dM(l) + alpha * (b_hi - b_lo)`` -- the second term carries no
    wavenumber meaning and must not be attributed to any spectral region.
    """
    full, bias = None, None
    for f, model in enumerate(ens.models):
        full_chunks, bias_chunks = [], []
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=ens.device)
            with torch.inference_mode():
                z = model.global_head(model.backbone(xb))
                base = model.base_heads[task](z)
                delta = model.contrib(xb, z)[task]
                b = model.contrib.bias[task](z)
            full_chunks.append((base + delta).detach().float().cpu().numpy())
            bias_chunks.append(b.detach().float().cpu().numpy())
        fold_full = np.concatenate(full_chunks, axis=0)
        fold_bias = np.concatenate(bias_chunks, axis=0)
        if full is None:
            full = np.zeros((ens.n_folds,) + fold_full.shape, dtype=np.float32)
            bias = np.zeros((ens.n_folds,) + fold_bias.shape, dtype=np.float32)
        full[f] = fold_full
        bias[f] = fold_bias
    return full, bias


def fold_contrib(ens: Ensemble, X: np.ndarray, task: str,
                 batch_size: int = 512) -> np.ndarray:
    """Per-fold KAN contribution maps ``(F, N, K, L)``."""
    out = None
    for f, model in enumerate(ens.models):
        chunks = []
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=ens.device)
            with torch.inference_mode():
                _, contrib = model(xb, return_contrib=True, logit_mode="full")
            chunks.append(contrib[task].detach().float().cpu().numpy())
        fold = np.concatenate(chunks, axis=0)
        if out is None:
            out = np.zeros((ens.n_folds,) + fold.shape, dtype=np.float32)
        out[f] = fold
    return out


# --------------------------------------------------------------------------- #
# margins
# --------------------------------------------------------------------------- #
def multi_margin(logits: np.ndarray) -> np.ndarray:
    """``o_2 - o_1`` (the count>=2 vs count=1 logit margin)."""
    return logits[..., 2] - logits[..., 1]


def binary_margin(logits: np.ndarray) -> np.ndarray:
    """``o_1 - o_0`` (present vs absent logit margin)."""
    return logits[..., 1] - logits[..., 0]


def support_sign(count: np.ndarray) -> np.ndarray:
    """+1 for count >= 2, -1 for count == 1."""
    return np.where(np.asarray(count) >= 2, 1.0, -1.0).astype(np.float32)


def true_class_margin(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """3-class margin ``o_y - max_{c != y} o_c`` (auxiliary metric)."""
    n, k = logits.shape
    mask = np.ones_like(logits, dtype=bool)
    mask[np.arange(n), y] = False
    other = logits[mask].reshape(n, k - 1).max(axis=1)
    return logits[np.arange(n), y] - other


# --------------------------------------------------------------------------- #
# spectral windows
# --------------------------------------------------------------------------- #
@dataclass
class Windows:
    """Equal-width, non-overlapping analysis windows on the common grid."""

    width_cm: float
    width_pts: int
    starts: np.ndarray           # (W,) inclusive index
    stops: np.ndarray            # (W,) exclusive index
    centers: np.ndarray          # (W,) centre wavenumber

    @property
    def n(self) -> int:
        return len(self.starts)

    @property
    def covered_cm(self) -> float:
        return self.width_pts * STEP_CM

    def mask_of(self, window_idx) -> np.ndarray:
        mask = np.zeros(len(GRID), dtype=bool)
        for w in np.atleast_1d(window_idx):
            mask[self.starts[w]:self.stops[w]] = True
        return mask


def make_windows(width_cm: float) -> Windows:
    width_pts = int(round(width_cm / STEP_CM))
    n = len(GRID) // width_pts
    starts = np.arange(n) * width_pts
    stops = starts + width_pts
    centers = GRID[(starts + stops - 1) // 2]
    return Windows(float(width_cm), width_pts, starts, stops, centers)


def window_sum(values: np.ndarray, windows: Windows) -> np.ndarray:
    """Integrate a ``(N, L)`` map over each window -> ``(N, W)`` (mean value x width)."""
    values = np.atleast_2d(values)
    cum = np.concatenate([np.zeros((len(values), 1), dtype=np.float64), np.cumsum(values, axis=1)], axis=1)
    return (cum[:, windows.stops] - cum[:, windows.starts]).astype(np.float32)


def coverage_to_k(windows: Windows, fraction: float) -> int:
    """Number of windows that cover ``fraction`` of the full spectral axis."""
    k = int(round(fraction / 100.0 * AXIS_SPAN / windows.width_cm))
    return max(1, min(k, windows.n))


def actual_fraction(windows: Windows, k: int) -> float:
    return 100.0 * k * windows.width_cm / AXIS_SPAN


def top_k(score: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest scores (negative preference), stable ties."""
    return np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")[:k]


def bottom_k(score: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(np.asarray(score, dtype=np.float64), kind="stable")[:k]


def characteristic_windows(group: str, windows: Windows) -> np.ndarray:
    """Boolean per-window flag: does the window centre fall inside a literature band?"""
    bands = FUNCTIONAL_GROUP_REGIONS[group]
    in_band = np.zeros(windows.n, dtype=bool)
    for hi, lo in bands:
        lo, hi = min(lo, hi), max(lo, hi)
        in_band |= (windows.centers >= lo) & (windows.centers <= hi)
    return in_band


def characteristic_coverage(group: str) -> float:
    """Fraction of the spectral axis covered by the literature bands of a group (%)."""
    bands = FUNCTIONAL_GROUP_REGIONS[group]
    mask = np.zeros(len(GRID), dtype=bool)
    for hi, lo in bands:
        lo, hi = min(lo, hi), max(lo, hi)
        mask |= (GRID >= lo) & (GRID <= hi)
    return 100.0 * float(mask.mean())


# --------------------------------------------------------------------------- #
# perturbation engine
# --------------------------------------------------------------------------- #
def mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[a, b)`` runs of ``True`` in a boolean mask."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


# --- reference (scalar) implementations, used by selftest_perturbation ------ #
def _interp_band(x: np.ndarray, a: int, b: int, out: np.ndarray) -> None:
    """Replace ``[a, b)`` by a straight line between the neighbouring samples."""
    n = len(x)
    left = x[a - 1] if a > 0 else None
    right = x[b] if b < n else None
    if left is None and right is None:
        out[a:b] = float(x.mean())
    elif left is None:
        out[a:b] = right
    elif right is None:
        out[a:b] = left
    else:
        out[a:b] = np.linspace(left, right, b - a + 2)[1:-1]


def _baseline_band(x: np.ndarray, a: int, b: int, excluded: np.ndarray, out: np.ndarray) -> None:
    """Replace ``[a, b)`` by a line between the flank baseline levels (median)."""
    n = len(x)
    width = b - a
    left = np.arange(max(0, a - width), a)
    right = np.arange(b, min(n, b + width))
    left = left[~excluded[left]]
    right = right[~excluded[right]]
    left_level = float(np.median(x[left])) if left.size else None
    right_level = float(np.median(x[right])) if right.size else None
    if left_level is None and right_level is None:
        keep = x[~excluded]
        out[a:b] = float(np.median(keep)) if keep.size else float(x.mean())
        return
    if left_level is None:
        left_level = right_level
    if right_level is None:
        right_level = left_level
    out[a:b] = np.linspace(left_level, right_level, width + 2)[1:-1]


def perturb_reference(X: np.ndarray, masks: np.ndarray, kind: str = "interp") -> np.ndarray:
    """Straightforward per-sample implementation (kept for the equivalence test)."""
    X = np.asarray(X, dtype=np.float32)
    out = X.copy()
    for i in range(len(X)):
        mask = masks[i]
        if not mask.any():
            continue
        for a, b in mask_runs(mask):
            if kind == "interp":
                _interp_band(X[i], a, b, out[i])
            else:
                _baseline_band(X[i], a, b, mask, out[i])
    return out


# --- vectorised engine used by the experiments ------------------------------ #
def _flank_mean(values: np.ndarray, counts: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Mean of ``values`` over ``[lo, hi)`` per element (``values`` are masked-out)."""
    top = values.shape[1] - 1            # cumulative arrays are indexed 0..L
    lo = np.clip(lo, 0, top)
    hi = np.clip(hi, 0, top)
    total = (np.take_along_axis(values, hi, axis=1)
             - np.take_along_axis(values, lo, axis=1))
    n = (np.take_along_axis(counts, hi, axis=1)
         - np.take_along_axis(counts, lo, axis=1))
    return total / np.maximum(n, 1.0), n


def perturb(X: np.ndarray, masks: np.ndarray, kind: str = "interp",
            renormalize: bool = False) -> np.ndarray:
    """Apply one perturbation to every row of ``X`` with its own boolean mask.

    kind="interp"    local smooth interpolation between the band edges (method A)
    kind="baseline"  local baseline replacement: the band is replaced by a linear
                     ramp between the mean *lower-envelope* levels of the two
                     flanking regions (masked points excluded).  The envelope is a
                     running minimum, so this operator can only remove absorption,
                     never create it -- which a plain flank mean would do.

    The operation acts on the *preprocessed* spectrum, i.e. exactly what the
    network consumes.  ``renormalize=True`` re-applies the per-sample z-score
    afterwards (robustness check: keeps the input on the training manifold).
    """
    X = np.asarray(X, dtype=np.float32)
    if masks.shape != X.shape:
        raise ValueError(f"mask shape {masks.shape} != input shape {X.shape}")

    out = X.copy()
    active = masks.any(axis=1)
    if not active.any():
        return out

    n_pts = X.shape[1]
    idx = np.arange(n_pts, dtype=np.float32)
    rows = np.flatnonzero(active)
    x, m = X[rows], masks[rows]

    # nearest unmasked anchor on each side (== the band edge we interpolate from)
    left = np.maximum.accumulate(np.where(~m, np.arange(n_pts)[None, :], -1), axis=1)
    right = np.minimum.accumulate(
        np.where(~m, np.arange(n_pts)[None, :], n_pts)[:, ::-1], axis=1)[:, ::-1]
    li = np.clip(left, 0, n_pts - 1)
    ri = np.clip(right, 0, n_pts - 1)
    left_val = np.take_along_axis(x, li, axis=1)
    right_val = np.take_along_axis(x, ri, axis=1)
    left_val = np.where(left < 0, right_val, left_val)
    right_val = np.where(right >= n_pts, left_val, right_val)

    if kind == "interp":
        denom = np.maximum(ri - li, 1).astype(np.float32)
        t = (idx[None, :] - li) / denom
        filled = left_val + (right_val - left_val) * t
    elif kind == "baseline":
        # Local *lower-envelope* baseline.  Using the mean of the flanks would also
        # raise regions that are already at baseline level, i.e. it would *create*
        # absorption; the running minimum of the neighbourhood is by construction
        # below the local absorption level, so replacing a band with it only ever
        # removes absorption.
        from scipy.ndimage import minimum_filter1d

        run_start = np.minimum.accumulate(
            np.where(m, np.arange(n_pts)[None, :], n_pts)[:, ::-1], axis=1)[:, ::-1]
        run_stop = np.maximum.accumulate(
            np.where(m, np.arange(n_pts)[None, :], -1), axis=1) + 1
        width = (run_stop - run_start).astype(np.int64)
        radius = 15          # running minimum over +/-30 cm^-1 (condition independent)

        envelope = minimum_filter1d(
            np.where(m, np.inf, x).astype(np.float64), size=2 * radius + 1, mode="nearest")
        valid = np.isfinite(envelope) & ~m
        values = np.where(valid, envelope, 0.0)
        counts = valid.astype(np.float64)
        cs = np.concatenate([np.zeros((len(x), 1)), np.cumsum(values, axis=1)], axis=1)
        cc = np.concatenate([np.zeros((len(x), 1)), np.cumsum(counts, axis=1)], axis=1)

        left_mean, left_n = _flank_mean(cs, cc, run_start - width, run_start)
        right_mean, right_n = _flank_mean(cs, cc, run_stop, run_stop + width)
        global_mean, _ = _flank_mean(cs, cc, np.zeros_like(width), np.full_like(width, n_pts))
        global_fallback = np.median(np.where(valid, envelope, np.nan), axis=1, keepdims=True)
        global_fallback = np.where(np.isfinite(global_fallback), global_fallback, global_mean)

        left_level = np.where(left_n > 0, left_mean,
                              np.where(right_n > 0, right_mean, global_fallback))
        right_level = np.where(right_n > 0, right_mean,
                               np.where(left_n > 0, left_mean, global_fallback))
        t = (idx[None, :] - run_start + 1.0) / (width + 1.0)
        filled = left_level + (right_level - left_level) * t
    else:
        raise ValueError(f"unknown perturbation kind: {kind}")

    out[rows] = np.where(m, filled.astype(np.float32), x)

    if renormalize:
        mu = out.mean(axis=1, keepdims=True)
        sd = out.std(axis=1, keepdims=True) + 1e-6
        out = ((out - mu) / sd).astype(np.float32)

    return out


def masks_from_selection(selected: np.ndarray, windows: "Windows") -> np.ndarray:
    """Turn per-sample window indices ``(N, k)`` into per-sample boolean masks ``(N, L)``."""
    sel = np.atleast_2d(selected)
    pts = (windows.starts[sel][:, :, None]
           + np.arange(windows.width_pts)[None, None, :]).reshape(len(sel), -1)
    masks = np.zeros((len(sel), len(GRID)), dtype=bool)
    np.put_along_axis(masks, pts, True, axis=1)
    return masks


def selftest_perturbation(seed: int = 0, n: int = 24) -> dict:
    """Compare the vectorised engine with the scalar reference."""
    rng = np.random.default_rng(seed)
    length = len(GRID)
    X = rng.normal(size=(n, length)).astype(np.float32)
    X += 3.0 * np.exp(-0.5 * ((np.arange(length)[None, :] - 800) / 12.0) ** 2)
    win = make_windows(30.0)
    masks = np.zeros((n, length), dtype=bool)
    for i in range(n):
        k = int(rng.integers(1, 6))
        for w in rng.choice(win.n, size=k, replace=False):
            masks[i, win.starts[w]:win.stops[w]] = True
        # force edge-touching bands for a few rows
        if i % 7 == 0:
            masks[i, 0:15] = True
        if i % 11 == 0:
            masks[i, length - 15:] = True

    report = {}
    ref = perturb_reference(X, masks, "interp")
    got = perturb(X, masks, "interp")
    report["interp_max_abs_err"] = float(np.abs(ref - got).max())

    ref_b = perturb_reference(X, masks, "baseline")
    got_b = perturb(X, masks, "baseline")
    # the reference uses a flank median, the engine a flank mean
    report["baseline_mean_abs_diff"] = float(np.abs(ref_b - got_b).mean())
    report["baseline_max_abs_diff"] = float(np.abs(ref_b - got_b).max())
    report["n_masked_points"] = int(masks.sum())
    report["ok"] = bool(report["interp_max_abs_err"] < 1e-5
                        and report["baseline_mean_abs_diff"] < 0.5)
    return report


# --------------------------------------------------------------------------- #
# attribution: Grad-CAM on the fused model with a margin target
# --------------------------------------------------------------------------- #
class MarginScore(nn.Module):
    """Expose ``s * (o_hi - o_lo)`` as a single-logit "class" for the CAM engine.

    ``s`` is set per batch so that one engine call can explain samples whose
    true count class differs (count=1 vs count>=2).
    """

    def __init__(self, model: nn.Module, task: str, hi: int, lo: int, logit_mode: str = "full"):
        super().__init__()
        self.model = model
        self.task = task
        self.hi = hi
        self.lo = lo
        self.logit_mode = logit_mode
        self.backbone = model.backbone            # used by GradCAM1D to pick the conv layer
        self.register_buffer("sign", torch.ones(1, 1), persistent=False)

    def set_sign(self, sign: np.ndarray) -> None:
        self.sign = torch.as_tensor(np.asarray(sign, dtype=np.float32).reshape(-1, 1),
                                    device=next(self.model.parameters()).device)

    def forward(self, x: torch.Tensor, logit_mode: str | None = None) -> dict[str, torch.Tensor]:
        out = self.model(x, logit_mode=self.logit_mode)
        logits = out[self.task]
        margin = logits[:, self.hi:self.hi + 1] - logits[:, self.lo:self.lo + 1]
        sign = self.sign.to(margin.device, margin.dtype)
        if sign.shape[0] not in (1, margin.shape[0]):
            raise ValueError("MarginScore.sign has the wrong batch size")
        return {self.task: margin * sign}


def gradcam_maps(ens: Ensemble, X: np.ndarray, task: str, hi: int, lo: int,
                 sign: np.ndarray, method: str = "gradcam++",
                 batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Fold-mean margin CAM ``(N, L)`` and the per-sample "collapsed map" flag.

    The target is the *deployed* fused score, so gradients flow through both the
    CNN branch and the KAN branch, exactly like the real prediction path.
    ``collapsed[i]`` marks samples whose fold-mean CAM is identically zero (the
    ReLU kills every channel); selection from such a map carries no attribution
    and must be reported separately.
    """
    out = np.zeros((len(X), len(GRID)), dtype=np.float64)

    for f, model in enumerate(ens.models):
        wrapper = MarginScore(model, task, hi, lo)
        engine = GradCAM1D(wrapper, layer=None, method=method)
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            xb = torch.as_tensor(X[start:stop], dtype=torch.float32, device=ens.device)
            wrapper.set_sign(sign[start:stop])
            res = engine(xb, np.zeros(stop - start, dtype=np.int64), task)
            out[start:stop] += res.cam
        del engine, wrapper

    cam = (out / ens.n_folds).astype(np.float32)
    return cam, cam.max(axis=1) <= 1e-12


# --------------------------------------------------------------------------- #
# selection rules
# --------------------------------------------------------------------------- #
def select_windows(score_maps: np.ndarray, windows: Windows, k: int) -> np.ndarray:
    """Top-k windows per sample from a ``(N, L)`` support-aligned map."""
    win = window_sum(score_maps, windows)
    return np.stack([top_k(win[i], k) for i in range(len(win))])


def select_random(n: int, windows: Windows, k: int, rng: np.random.Generator) -> np.ndarray:
    return np.stack([rng.choice(windows.n, size=k, replace=False) for _ in range(n)])


def select_chemistry(n: int, windows: Windows, k: int, intensity: np.ndarray,
                     in_band: np.ndarray) -> np.ndarray:
    """Chemistry-first ranking: literature bands first, then strongest bands.

    Windows inside the functional-group characteristic regions are taken first
    (ordered by mean intensity), the remaining slots are filled by the strongest
    remaining windows.  When the literature bands cover fewer windows than ``k``
    (always the case for nitriles, one C==N band only) the fill is reported, so
    the curve can be read as "how far chemistry alone gets you".
    """
    win = window_sum(intensity, windows)
    order = np.argsort(-in_band.astype(np.int64) * 1e6 - win, axis=1, kind="stable")
    return order[:, :k]


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def cluster_index(clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map cluster labels to ``(codes, sizes)``."""
    uniq, codes = np.unique(clusters, return_inverse=True)
    return codes, np.bincount(codes)


def bootstrap_mean(values: np.ndarray, codes: np.ndarray, n_boot: int = 2000,
                   seed: int = 0) -> dict:
    """Cluster bootstrap CI of the mean of ``values`` (rows = samples)."""
    values = np.asarray(values, dtype=np.float64)
    n_clusters = codes.max() + 1
    sums = np.bincount(codes, weights=values, minlength=n_clusters)
    counts = np.bincount(codes, minlength=n_clusters).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "n": int(values.size),
        "n_clusters": int(n_clusters),
    }


def bootstrap_rate(flags: np.ndarray, codes: np.ndarray, n_boot: int = 2000,
                   seed: int = 0) -> dict:
    """Cluster bootstrap CI of a proportion."""
    return bootstrap_mean(np.asarray(flags, dtype=np.float64), codes, n_boot, seed)


def bootstrap_paired(a: np.ndarray, b: np.ndarray, codes: np.ndarray,
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """Cluster bootstrap CI of ``mean(a - b)`` plus a two-sided bootstrap p-value."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n_clusters = codes.max() + 1
    sums = np.bincount(codes, weights=diff, minlength=n_clusters)
    counts = np.bincount(codes, minlength=n_clusters).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    p_left = float((boot <= 0).mean())
    p_right = float((boot >= 0).mean())
    return {
        "diff": float(diff.mean()),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "p_two_sided": float(min(1.0, 2.0 * min(p_left, p_right))),
        "n_clusters": int(n_clusters),
    }


def bootstrap_stat(stat_fn, n_samples: int, codes: np.ndarray, n_boot: int = 2000,
                   seed: int = 0) -> dict:
    """Cluster bootstrap CI of an arbitrary statistic ``stat_fn(idx) -> float``."""
    n_clusters = codes.max() + 1
    members = [np.flatnonzero(codes == c) for c in range(n_clusters)]
    rng = np.random.default_rng(seed)
    base = float(stat_fn(np.arange(n_samples)))
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([members[c] for c in pick])
        boot[b] = stat_fn(idx)
    return {
        "value": base,
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "n_clusters": int(n_clusters),
    }


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank AUC (ties handled by average ranks); NaN when only one class is present."""
    from scipy.stats import rankdata

    labels = np.asarray(labels).astype(int)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row Spearman correlation between two ``(N, W)`` matrices."""
    from scipy.stats import rankdata

    ra = rankdata(a, axis=1).astype(np.float64)
    rb = rankdata(b, axis=1).astype(np.float64)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    denom = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    return (ra * rb).sum(axis=1) / np.maximum(denom, 1e-12)


def topk_overlap(sel_a: np.ndarray, sel_b: np.ndarray) -> float:
    """Mean Jaccard overlap between two ``(N, k)`` selections."""
    out = np.empty(len(sel_a), dtype=np.float64)
    for i in range(len(sel_a)):
        a, b = set(np.atleast_1d(sel_a[i]).tolist()), set(np.atleast_1d(sel_b[i]).tolist())
        out[i] = len(a & b) / max(1, len(a | b))
    return float(out.mean())


def mass_in_bands(maps: np.ndarray, windows: "Windows", in_band: np.ndarray) -> float:
    """Share of the absolute attribution mass falling inside the literature bands."""
    win = np.abs(window_sum(maps, windows))
    total = win.sum(axis=1)
    inside = win[:, in_band].sum(axis=1)
    ok = total > 1e-12
    return float((inside[ok] / total[ok]).mean()) if ok.any() else float("nan")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class AnalysisData:
    """Everything the two experiments need about the held-out evaluation set."""

    frame: pl.DataFrame          # main_test rows (row order defines the index space)
    raw: np.ndarray              # (N, L) raw spectra
    labels: dict[str, np.ndarray]
    compounds: np.ndarray        # str cluster label per sample (resampling unit)
    smiles: np.ndarray

    def __len__(self) -> int:
        return self.frame.height


def load_analysis_data(eval_name: str = EVAL_NAME) -> AnalysisData:
    frame = pl.read_parquet(TEST_PARQUET).filter(pl.col(EVAL_NAME_COL) == eval_name)
    tasks = tasks_from_manifest(load_manifest(RUN_DIR))
    raw, labels, _ = frame_to_xy(frame, tasks)

    names = frame["compound_name"].to_list()
    smiles = frame["smiles"].to_list()
    sample_ids = frame["sample_id"].to_list()

    # Resampling unit = molecular structure (SMILES).  Several records of the same
    # structure (different source / different acquisition) are *not* independent,
    # so they must move together in the bootstrap.
    compounds = np.array([
        (s or "").strip() or (n or "").strip().lower() or sid
        for s, n, sid in zip(smiles, names, sample_ids)
    ], dtype=object)

    return AnalysisData(
        frame=frame,
        raw=np.asarray(raw, dtype=np.float32),
        labels=labels,
        compounds=compounds,
        smiles=np.asarray(smiles, dtype=object),
    )


def group_samples(data: AnalysisData, group: str, min_count: int = 1) -> np.ndarray:
    """Row indices of the analysis set (count >= 1 by default: multiplicity is defined)."""
    col = GROUPS[group]["column"]
    counts = data.frame[col].to_numpy()
    return np.flatnonzero(counts >= min_count)


def support_table(data: AnalysisData, group: str) -> dict:
    col = GROUPS[group]["column"]
    counts = data.frame[col].to_numpy()
    return {
        "group": group,
        "n_total": int(len(counts)),
        "n_c0": int((counts == 0).sum()),
        "n_c1": int((counts == 1).sum()),
        "n_c2plus": int((counts >= 2).sum()),
        "n_c2": int((counts == 2).sum()),
        "n_c3": int((counts == 3).sum()),
        "n_c4plus": int((counts >= 4).sum()),
        "n_compounds_c1": int(len(set(data.compounds[counts == 1]))),
        "n_compounds_c2plus": int(len(set(data.compounds[counts >= 2]))),
    }


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def parse_device(text: str | None) -> torch.device:
    if text:
        return torch.device(text)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
