"""1-D Grad-CAM / Grad-CAM++ attribution for the CKAN-SpecNet **CNN branch**.

The deployed model is ``CNNContributionKAN``::

    x (B, 1646) -> backbone (4 x [Conv1d + BN + act (+pool) (+ECA)])  -> (B, 256, 250)
                -> global_head (adaptive avg-max pool + Linear)        -> (B, 1024) == z
                -> base_heads[task](z)                                 -> CNN logits
                -> contrib(x, z)                                       -> KAN logits (delta)

Two logit paths can be attributed:

``logit_mode="base"``
    Pure CNN branch (``base_heads``): the exact decision path of the CNN-only
    ablation model.  Nothing from the KAN module reaches the graph.
``logit_mode="full"``
    Deployed model (``base + delta``).  Gradients still enter the backbone
    through ``global_head``, so the resulting map explains the real prediction.

Why a dedicated 1-D implementation
----------------------------------
`pytorch-grad-cam` is a 2-D image library: ``BaseCAM.get_target_width_height``
only accepts (B,C,H,W) / (B,C,D,H,W) tensors and the CAM is resized with
``cv2.resize``.  This module implements the same equations directly on
(B, C, L) feature maps, so the CAM stays on the real wavenumber axis and its
**native resolution** (250 points for the released model, i.e. 13.2 cm^-1 per
CAM cell) can be reported honestly -- which is exactly what reviewer comment 3
asks about.  ``scripts/../gradCam/verify_gradcam_impl.py`` cross-checks the
numbers against ``pytorch_grad_cam`` (which works on 1-D spectra when the input
is reshaped to (B, 1, 1, L)).

Formulas (1-D, Z = length of the feature map)
---------------------------------------------
Grad-CAM (Selvaraju et al. 2017), channel weights::

    alpha_k = (1/Z) * sum_z  d y^c / d A_k(z)
    L_cam   = ReLU( sum_k alpha_k * A_k )        (upsampled to input length)

Grad-CAM++ (Chattopadhay et al. 2018), pixel-wise weights::

    a_ij^k  = g_ij^2 / (2 g_ij^2 + sum_z A_k(z) g_k(z)^3 + eps)
    alpha_k = sum_z ReLU(g_k(z)) * a_k(z)
    L_cam   = ReLU( sum_k alpha_k * A_k )
    (g = d y^c / d A)

XGrad-CAM (Fu et al. 2020)::

    alpha_k = sum_z A_k(z) * g_k(z) / (sum_z A_k(z) + eps)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CAM_METHODS = ("gradcam", "gradcam++", "xgradcam")


# --------------------------------------------------------------------------- #
# model helpers
# --------------------------------------------------------------------------- #
def backbone_of(model: nn.Module) -> nn.Sequential:
    """Return the conv stack of a CKAN-SpecNet model (plain ``CNN`` or ``CNNContributionKAN``)."""
    if hasattr(model, "backbone"):
        return model.backbone
    raise ValueError("model has no `.backbone` (expected CNN / CNNContributionKAN)")


def conv1d_layers(model: nn.Module) -> list[nn.Conv1d]:
    """All Conv1d layers of the backbone, in forward order.

    Falls back to every Conv1d of the module tree for non-CKAN models.
    """
    try:
        return [m for m in backbone_of(model).net if isinstance(m, nn.Conv1d)]
    except ValueError:
        return [m for m in model.modules() if isinstance(m, nn.Conv1d)]


def resolve_cam_layer(model: nn.Module, which: str | int | None = None) -> tuple[nn.Conv1d, str]:
    """Pick the target conv layer.

    ``None`` / ``"last"`` -> last Conv1d (default, the standard Grad-CAM layer).
    ``"first"``/``"c1"`` -> first conv block; ``"c2"``, ``"c3"``, ``"c4"`` are the
    successive conv blocks (c4 == last for the released 4-block backbone).
    An ``int`` is used as an index into :func:`conv1d_layers`.
    """
    convs = conv1d_layers(model)

    if isinstance(which, int):
        if not -len(convs) <= which < len(convs):
            raise IndexError(f"conv layer index {which} out of range ({len(convs)} layers)")
        layer = convs[which]
        return layer, f"conv[{which}]"

    key = "last" if which is None else str(which).lower()

    if key == "last":
        return convs[-1], f"conv[{len(convs) - 1}]"

    if key == "first":
        return convs[0], "conv[0]"

    if key.startswith("c") and key[1:].isdigit():
        idx = int(key[1:]) - 1
        if not 0 <= idx < len(convs):
            raise IndexError(f"layer {which} out of range ({len(convs)} conv blocks)")
        return convs[idx], f"conv[{idx}]"

    raise ValueError(f"Unknown cam layer selector: {which!r}")


class TaskLogitWrapper(nn.Module):
    """Expose one task's logits as a plain tensor so 3rd-party CAM libs can drive it."""

    def __init__(self, model: nn.Module, task: str, logit_mode: str = "full"):
        super().__init__()
        self.model = model
        self.task = task
        self.logit_mode = logit_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x, logit_mode=self.logit_mode)
        return out[self.task]


# --------------------------------------------------------------------------- #
# core CAM
# --------------------------------------------------------------------------- #
@dataclass
class CamResult:
    """Container returned by :class:`GradCAM1D`."""

    cam: np.ndarray          # (B, L) upsampled to input length
    cam_native: np.ndarray   # (B, Z) on the target layer's own grid
    weights: np.ndarray      # (B, K) channel weights
    native_length: int       # Z
    layer_name: str


class GradCAM1D:
    """Native 1-D Grad-CAM / Grad-CAM++ / XGrad-CAM for CKAN-SpecNet."""

    def __init__(
        self,
        model: nn.Module,
        layer: str | int | None = None,
        method: str = "gradcam",
        eps: float = 1e-6,   # matches pytorch-grad-cam's GradCAM++ denominator
    ):
        method = method.lower()
        if method not in CAM_METHODS:
            raise ValueError(f"method must be one of {CAM_METHODS}, got {method!r}")

        self.model = model
        self.method = method
        self.eps = eps

        if isinstance(layer, nn.Module):
            # any module works (useful for toy models / unit tests)
            self.layer, self.layer_name = layer, type(layer).__name__
        else:
            self.layer, self.layer_name = resolve_cam_layer(model, layer)

        self._act: torch.Tensor | None = None
        self._handle = None

    # -- hooks ------------------------------------------------------------- #
    def _fwd_hook(self, _module, _inp, out):
        self._act = out

    def _capture(self, x, task, logit_mode):
        self.model.eval()
        self._act = None
        self._handle = self.layer.register_forward_hook(self._fwd_hook)
        try:
            out = self.model(x, logit_mode=logit_mode)
        finally:
            self._handle.remove()
            self._handle = None

        if self._act is None:
            raise RuntimeError("target layer was never executed during forward()")

        return out[task]

    # -- per-method channel weights ---------------------------------------- #
    def _weights(self, act: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
        if self.method == "gradcam":
            return grads.mean(dim=2)

        if self.method == "gradcam++":
            g2 = grads.pow(2)
            g3 = g2 * grads
            sum_act = act.sum(dim=2, keepdim=True)          # (B,K,1)
            aij = g2 / (2.0 * g2 + sum_act * g3 + self.eps)
            aij = torch.where(grads != 0, aij, torch.zeros_like(aij))
            return (F.relu(grads) * aij).sum(dim=2)

        # xgradcam
        act_sum = act.sum(dim=2)
        return (act * grads).sum(dim=2) / (act_sum + self.eps)

    # -- public API -------------------------------------------------------- #
    def __call__(
        self,
        x: torch.Tensor,
        class_ids: Sequence[int] | torch.Tensor,
        task: str,
        logit_mode: str = "full",
        input_length: int | None = None,
    ) -> CamResult:
        """Compute CAM maps.

        Parameters
        ----------
        x : (B, L) preprocessed spectra (``Preprocessor`` applied).
        class_ids : per-sample class index to explain (len B).
        task : task key, e.g. ``"alcohols_4class"``.
        logit_mode : ``"full"`` (deployed model) or ``"base"`` (CNN branch only).
        input_length : output length of the CAM; defaults to ``x.shape[1]``.
        """
        if x.dim() != 2:
            raise ValueError(f"expected (B, L) input, got {tuple(x.shape)}")

        batch = x.shape[0]
        class_ids = torch.as_tensor(class_ids, dtype=torch.long, device=x.device)
        if class_ids.numel() != batch:
            raise ValueError("class_ids must have one entry per sample")

        with torch.enable_grad():
            x = x.detach().requires_grad_(False)
            logits = self._capture(x, task, logit_mode)          # (B, C)
            act = self._act                                       # (B, K, Z)

            # eval-mode BatchNorm -> samples are statistically independent, so a
            # single backward on the summed score yields exact per-sample grads.
            score = logits.gather(1, class_ids.view(-1, 1)).sum()
            (grads,) = torch.autograd.grad(score, act, retain_graph=False)

        weights = self._weights(act, grads)                       # (B, K)
        cam = F.relu((weights.unsqueeze(-1) * act).sum(dim=1))    # (B, Z)

        target_len = int(input_length or x.shape[1])
        if cam.shape[1] != target_len:
            cam_up = F.interpolate(
                cam.unsqueeze(1), size=target_len, mode="linear", align_corners=False
            ).squeeze(1)
        else:
            cam_up = cam

        return CamResult(
            cam=cam_up.detach().cpu().numpy(),
            cam_native=cam.detach().cpu().numpy(),
            weights=weights.detach().cpu().numpy(),
            native_length=int(cam.shape[1]),
            layer_name=self.layer_name,
        )


# --------------------------------------------------------------------------- #
# attribution post-processing
# --------------------------------------------------------------------------- #
def normalize_cam(cam: np.ndarray, mode: str = "max") -> np.ndarray:
    """Per-sample normalisation.

    ``max``   divide by the per-sample maximum (keeps peak ratios, standard).
    ``sum``   divide by the per-sample sum (turns CAM into a density/mass).
    ``none``  unchanged.
    """
    cam = np.asarray(cam, dtype=np.float32)

    if mode == "none":
        return cam

    if mode == "max":
        denom = cam.max(axis=1, keepdims=True)
    elif mode == "sum":
        denom = cam.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown normalisation: {mode}")

    return cam / np.where(denom > 1e-12, denom, 1.0)


def smooth_cam(cam: np.ndarray, window: int = 15) -> np.ndarray:
    """Box-filter a CAM along the wavenumber axis (same padding)."""
    cam = np.asarray(cam, dtype=np.float32)

    if window is None or window <= 1:
        return cam

    window = int(window) + int(int(window) % 2 == 0)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.stack([np.convolve(row, kernel, mode="same") for row in np.atleast_2d(cam)])


def top_bands(
    values: np.ndarray,
    grid: np.ndarray,
    top_percent: float = 5.0,
    min_width: float = 8.0,
    max_gap: float = 6.0,
    mode: str = "abs",
) -> list[tuple[float, float, float]]:
    """Longest-contiguous bands holding the top ``top_percent`` of ``values``.

    ``mode="abs"``   rank by ``|values|`` (a difference map).
    ``mode="value"`` rank by ``values`` (a CAM density).

    Returns ``[(start_cm, end_cm, mean_signed_value), ...]`` sorted by |mean value|.
    """
    values = np.asarray(values, dtype=np.float32).ravel()
    grid = np.asarray(grid, dtype=np.float32).ravel()

    if values.shape != grid.shape:
        raise ValueError("values and grid must have the same length")

    score = np.abs(values) if mode == "abs" else values
    n = values.size
    k = max(1, int(round(n * top_percent / 100.0)))

    # threshold at the k-th largest score -> mask of selected points
    cut = np.partition(score, -k)[-k]
    mask = score >= cut

    if not mask.any():
        return []

    # split into contiguous runs, bridging tiny holes
    idx = np.flatnonzero(mask)
    runs: list[tuple[int, int]] = []
    start = prev = idx[0]

    for i in idx[1:]:
        step = float(np.abs(grid[i] - grid[prev]))
        if step <= max_gap:
            prev = i
            continue
        runs.append((start, prev))
        start = prev = i
    runs.append((start, prev))

    step = float(np.median(np.abs(np.diff(grid)))) if n > 1 else 2.0
    bands = []

    for a, b in runs:
        lo, hi = (grid[a], grid[b]) if grid[a] <= grid[b] else (grid[b], grid[a])
        # widen by half a step so the band covers the selected samples fully
        lo, hi = lo - step / 2.0, hi + step / 2.0
        if hi - lo < min_width:
            hi = lo + min_width
        bands.append((float(lo), float(hi), float(values[a : b + 1].mean())))

    bands.sort(key=lambda item: -abs(item[2]))
    return bands


def attention_mask(cam: np.ndarray, quantile: float = 0.9) -> np.ndarray:
    """Boolean mask of the highest-attribution samples (``quantile`` of the CAM mass)."""
    cam = np.asarray(cam, dtype=np.float32).ravel()
    cut = float(np.quantile(cam, quantile))
    return cam > cut


# --------------------------------------------------------------------------- #
# ensemble CAM (five-fold mean, mirrors the soft-voting ensemble)
# --------------------------------------------------------------------------- #
def ensemble_cam(
    models: Sequence[nn.Module],
    preprocessors: Sequence,
    raw_spectra: np.ndarray,
    class_ids: Sequence[int] | torch.Tensor,
    task: str,
    device: torch.device,
    method: str = "gradcam",
    layer: str | int | nn.Module | None = None,
    logit_mode: str = "full",
    batch_size: int = 256,
) -> CamResult:
    """Average CAM over the five folds, exactly like the probability ensemble.

    ``raw_spectra`` are **raw** parquet spectra (before ``Preprocessor``); each
    fold gets its own preprocessing so sample_zscore / global_zscore both work.
    """
    raw_spectra = np.asarray(raw_spectra, dtype=np.float32)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    n = len(raw_spectra)

    if len(class_ids) != n:
        raise ValueError("class_ids must have one entry per spectrum")

    cams, native, weights, native_len, layer_name = [], [], [], None, ""

    for model, prep in zip(models, preprocessors, strict=True):
        model = model.to(device).eval()
        cam_engine = GradCAM1D(model, layer=layer, method=method)
        layer_name = cam_engine.layer_name

        fold_cam = np.zeros((n, raw_spectra.shape[1]), dtype=np.float32)
        fold_native = None
        fold_weights = []

        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = np.stack([prep(raw) for raw in raw_spectra[start:stop]]).astype(np.float32)
            x = torch.as_tensor(batch, dtype=torch.float32, device=device)

            res = cam_engine(x, class_ids[start:stop], task, logit_mode=logit_mode)
            fold_cam[start:stop] = res.cam
            native_len = res.native_length
            fold_weights.append(res.weights)

            if fold_native is None:
                fold_native = np.zeros((n, res.cam_native.shape[1]), dtype=np.float32)
            fold_native[start:stop] = res.cam_native

        cams.append(fold_cam)
        native.append(fold_native)
        weights.append(np.concatenate(fold_weights, axis=0))
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return CamResult(
        cam=np.mean(cams, axis=0),
        cam_native=np.mean(native, axis=0),
        weights=np.mean(weights, axis=0),
        native_length=int(native_len or 0),
        layer_name=layer_name,
    )
