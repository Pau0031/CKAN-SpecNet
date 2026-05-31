from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Literal

import cv2
import imageio.v3 as iio
import numpy as np
import polars as pl
from doctr.models import ocr_predictor


Axis = Literal["x", "y"]
Curve = dict[str, np.ndarray]
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


@dataclass(frozen=True)
class Config:
    crop_inset: int = 3
    bin_threshold: int = 220
    ink_gray_threshold: int = 238
    ink_energy_threshold: float = 0.06

    stroke_width_min: float = 1.0
    stroke_width_max: float = 18.0
    stroke_radius_factor: float = 0.50

    c_max: int = 20
    max_candidates_normal: int = 7
    max_candidates_event: int = 20
    candidate_quantiles: tuple[float, ...] = (0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88)

    event_height_factor: float = 2.2
    event_gradient_factor: float = 2.3
    event_depression_factor: float = 1.2
    event_dilate_factor: float = 2.8

    point_center_penalty: float = 0.012
    point_white_penalty: float = 0.45
    point_dark_reward: float = 1.10
    point_ridge_reward: float = 0.45
    bridge_point_penalty: float = 4.5

    envelope_miss_weight: float = 9.2
    off_ink_weight: float = 0.72
    edge_ridge_reward: float = 0.42
    edge_energy_reward: float = 0.018
    length_weight: float = 0.004

    slope_weight: float = 0.018
    acceleration_weight: float = 0.085
    nonlinear_knee: float = 5.0
    support_relief: float = 0.92
    event_support_floor: float = 0.75
    max_abs_slope: float = 540.0

    snap_final: bool = True
    snap_radius_factor: float = 1.20
    snap_distance_weight: float = 0.85
    snap_energy_weight: float = 0.018
    snap_ridge_weight: float = 1.55

    ocr_min_conf: float = 0.12
    x_value_bounds: tuple[float, float] = (150.0, 5200.0)
    output_points: int = 1800

    fallback_x_start: float = 4000.0
    fallback_x_end: float = 500.0
    fallback_y_min: float = 0.0
    fallback_y_max: float = 100.0


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def clamp(self, w: int, h: int) -> Box:
        left = max(0, min(self.left, w - 2))
        top = max(0, min(self.top, h - 2))
        right = max(left + 1, min(self.right, w - 1))
        bottom = max(top + 1, min(self.bottom, h - 1))
        return Box(left, top, right, bottom)


@dataclass(frozen=True)
class LinearAxisFit:
    slope: float
    intercept: float
    rmse: float
    count: int
    source: str = "linear"

    def value_at(self, pixel):
        return self.slope * np.asarray(pixel, dtype=float) + self.intercept

    def pixel_at(self, value: float) -> float:
        return float((value - self.intercept) / self.slope)


@dataclass(frozen=True)
class PiecewiseAxisFit:
    pixels: np.ndarray
    values: np.ndarray
    rmse: float
    count: int
    source: str = "piecewise"

    def value_at(self, pixel):
        x = np.asarray(pixel, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)

        order = np.argsort(self.pixels)
        px = self.pixels[order].astype(float)
        val = self.values[order].astype(float)

        y = np.interp(x, px, val)

        if len(px) >= 2:
            left = x < px[0]
            if np.any(left):
                slope = (val[1] - val[0]) / (px[1] - px[0])
                y[left] = val[0] + slope * (x[left] - px[0])

            right = x > px[-1]
            if np.any(right):
                slope = (val[-1] - val[-2]) / (px[-1] - px[-2])
                y[right] = val[-1] + slope * (x[right] - px[-1])

        return float(y[0]) if scalar else y

    def pixel_at(self, value: float) -> float:
        order = np.argsort(self.values)
        val = self.values[order].astype(float)
        px = self.pixels[order].astype(float)

        if len(val) == 1:
            return float(px[0])

        if value < val[0]:
            slope = (px[1] - px[0]) / (val[1] - val[0])
            return float(px[0] + slope * (value - val[0]))

        if value > val[-1]:
            slope = (px[-1] - px[-2]) / (val[-1] - val[-2])
            return float(px[-1] + slope * (value - val[-1]))

        return float(np.interp(value, val, px))


@dataclass(frozen=True)
class OcrWord:
    text: str
    conf: float
    left: float
    top: float
    right: float
    bottom: float
    cx: float
    cy: float


@dataclass(frozen=True)
class AxisLabel:
    text: str
    axis_value: float
    tick_pixel: float
    left: float
    top: float
    right: float
    bottom: float
    conf: float
    inlier: bool = True


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def normalize_u8(values: np.ndarray, lo_q: float = 1, hi_q: float = 99.5) -> np.ndarray:
    values = values.astype(np.float32)
    lo, hi = np.nanpercentile(values, [lo_q, hi_q])

    if hi <= lo:
        return np.zeros(values.shape, np.uint8)

    return np.clip((values - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def read_gif(path: Path) -> np.ndarray:
    image = iio.imread(path)

    if image.ndim == 4:
        image = image[0]

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[-1] == 4:
        image = image[..., :3]

    return image.astype(np.uint8)


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".gif":
        return read_gif(path)

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image: {path.resolve()}")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def validate_input(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"input file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"input must be a file: {path}")

    if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise ValueError(f"unsupported image format: {path.suffix}; supported: {supported}")


def threshold_dark(image: np.ndarray, threshold: int) -> np.ndarray:
    _, mask = cv2.threshold(gray(image), threshold, 255, cv2.THRESH_BINARY_INV)
    return mask


def line_segments(mask: np.ndarray):
    h, w = mask.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 8), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 8)))

    hm = cv2.morphologyEx(mask, cv2.MORPH_OPEN, h_kernel)
    vm = cv2.morphologyEx(mask, cv2.MORPH_OPEN, v_kernel)

    h_segments = []
    v_segments = []

    for y in np.flatnonzero(np.count_nonzero(hm > 0, axis=1)):
        xs = np.flatnonzero(hm[y] > 0)

        for run in np.split(xs, np.where(np.diff(xs) > 4)[0] + 1):
            if len(run) >= w * 0.15:
                h_segments.append(
                    {
                        "x1": int(run[0]),
                        "x2": int(run[-1]),
                        "y": int(y),
                        "length": int(len(run)),
                    }
                )

    for x in np.flatnonzero(np.count_nonzero(vm > 0, axis=0)):
        ys = np.flatnonzero(vm[:, x] > 0)

        for run in np.split(ys, np.where(np.diff(ys) > 4)[0] + 1):
            if len(run) >= h * 0.10:
                v_segments.append(
                    {
                        "x": int(x),
                        "y1": int(run[0]),
                        "y2": int(run[-1]),
                        "length": int(len(run)),
                    }
                )

    return h_segments, v_segments


def detect_plot_box(image: np.ndarray, cfg: Config) -> tuple[Box, str]:
    mask = threshold_dark(image, cfg.bin_threshold)
    h, w = mask.shape
    h_segments, v_segments = line_segments(mask)
    candidates = []

    for bottom in h_segments:
        if not (h * 0.25 <= bottom["y"] <= h * 0.94):
            continue
        if bottom["length"] < w * 0.30:
            continue

        for left in v_segments:
            if left["x"] > w * 0.40:
                continue
            if abs(left["y2"] - bottom["y"]) > max(22, h * 0.025):
                continue
            if left["length"] < h * 0.12:
                continue

            right = bottom["x2"]
            top_candidates = [
                seg
                for seg in h_segments
                if abs(seg["x1"] - left["x"]) <= max(45, w * 0.025)
                and abs(seg["x2"] - right) <= max(110, w * 0.055)
                and h * 0.04 <= seg["y"] < bottom["y"] - h * 0.10
                and seg["length"] >= w * 0.25
            ]

            top = max([seg["y"] for seg in top_candidates], default=left["y1"])
            box = Box(left["x"], top, right, bottom["y"]).clamp(w, h)

            if box.width >= w * 0.25 and box.height >= h * 0.10:
                score = box.width * box.height + bottom["length"] * 10 + left["length"] * 10
                candidates.append((score, box))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1], "axis_or_box"

    return Box(int(w * 0.08), int(h * 0.12), int(w * 0.95), int(h * 0.75)).clamp(w, h), "fallback"


def crop_plot(image: np.ndarray, box: Box, cfg: Config) -> np.ndarray:
    inset = cfg.crop_inset
    return image[box.top + inset:box.bottom - inset, box.left + inset:box.right - inset]


def border_axis_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(70, w // 3), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(70, h // 3)))

    hm = cv2.morphologyEx(mask, cv2.MORPH_OPEN, h_kernel)
    vm = cv2.morphologyEx(mask, cv2.MORPH_OPEN, v_kernel)

    rows = np.count_nonzero(hm > 0, axis=1)
    cols = np.count_nonzero(vm > 0, axis=0)

    for y in np.flatnonzero(rows > w * 0.70):
        if y < h * 0.12 or y > h * 0.86:
            out[max(0, y - 1):min(h, y + 2), :] = hm[max(0, y - 1):min(h, y + 2), :]

    for x in np.flatnonzero(cols > h * 0.55):
        if x < w * 0.08 or x > w * 0.96:
            out[:, max(0, x - 1):min(w, x + 2)] = vm[:, max(0, x - 1):min(w, x + 2)]

    return cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)


def ink_energy_from_gray(g: np.ndarray) -> np.ndarray:
    g = g.astype(np.float32)
    border = np.r_[g[:5, :].ravel(), g[-5:, :].ravel(), g[:, :5].ravel(), g[:, -5:].ravel()]
    return np.maximum(0, float(np.median(border)) - g)


def ink_mask_from_gray(g: np.ndarray, energy: np.ndarray, cfg: Config) -> np.ndarray:
    e = normalize_u8(energy)
    mask = ((g <= cfg.ink_gray_threshold) | (e >= int(cfg.ink_energy_threshold * 255))).astype(np.uint8) * 255

    mask[border_axis_mask(mask) > 0] = 0

    mask[:2, :] = 0
    mask[-2:, :] = 0
    mask[:, :2] = 0
    mask[:, -2:] = 0

    return mask


def estimate_stroke_width(mask: np.ndarray, cfg: Config) -> float:
    binary = (mask > 0).astype(np.uint8)

    if cv2.countNonZero(binary) == 0:
        return 2.0

    values = cv2.distanceTransform(binary, cv2.DIST_L2, 3)[binary > 0]

    if len(values) == 0:
        return 2.0

    return float(np.clip(2.0 * np.percentile(values, 82), cfg.stroke_width_min, cfg.stroke_width_max))


def split_runs(indices: np.ndarray) -> list[np.ndarray]:
    if len(indices) == 0:
        return []

    return np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)


def column_runs(mask: np.ndarray) -> dict[int, list[dict]]:
    out = {}

    for x in range(mask.shape[1]):
        runs = []

        for run_id, run in enumerate(split_runs(np.flatnonzero(mask[:, x] > 0))):
            top = float(run.min())
            bottom = float(run.max())

            runs.append(
                {
                    "x": int(x),
                    "run_id": int(run_id),
                    "top": top,
                    "bottom": bottom,
                    "center": 0.5 * (top + bottom),
                    "height": bottom - top + 1.0,
                    "ys": run.astype(int),
                }
            )

        if runs:
            out[x] = runs

    return out


def interpolate_nan(values: np.ndarray) -> np.ndarray:
    x = np.arange(len(values))
    ok = np.isfinite(values)

    if ok.sum() == 0:
        return np.zeros_like(values, dtype=float)

    if ok.sum() == 1:
        return np.full_like(values, float(values[ok][0]), dtype=float)

    return np.interp(x, x[ok], values[ok])


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window) | 1)
    pad = window // 2
    padded = np.pad(values.astype(np.float64), pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1)


def prepare_fields(mask: np.ndarray, energy: np.ndarray, stroke_width: float) -> dict:
    _, w = mask.shape
    runs_by_x = column_runs(mask)

    top = np.full(w, np.nan, dtype=float)
    bottom = np.full(w, np.nan, dtype=float)
    center = np.full(w, np.nan, dtype=float)
    height = np.zeros(w, dtype=float)
    weighted_center = np.full(w, np.nan, dtype=float)
    ridge_y = np.full(w, np.nan, dtype=float)
    energy_y = np.full(w, np.nan, dtype=float)

    ridge = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3).astype(float)

    if ridge.max() > 0:
        ridge /= float(ridge.max())

    for x, runs in runs_by_x.items():
        ys_all = np.flatnonzero(mask[:, x] > 0)

        top[x] = float(ys_all.min())
        bottom[x] = float(ys_all.max())
        center[x] = 0.5 * (top[x] + bottom[x])
        height[x] = bottom[x] - top[x] + 1.0

        run = max(runs, key=lambda item: item["height"])
        ys = run["ys"]
        weights = np.maximum(energy[ys, x].astype(float), 1.0)

        weighted_center[x] = float(np.sum(ys * weights) / np.sum(weights))
        ridge_y[x] = float(ys[int(np.argmax(ridge[ys, x]))])
        energy_y[x] = float(ys[int(np.argmax(energy[ys, x]))])

    top_i = interpolate_nan(top)
    bottom_i = interpolate_nan(bottom)
    baseline = rolling_median(top_i, max(9, int(round(18 * stroke_width)) | 1))

    return {
        "runs_by_x": runs_by_x,
        "top": top_i.astype(np.float32),
        "bottom": bottom_i.astype(np.float32),
        "center": interpolate_nan(center).astype(np.float32),
        "weighted_center": interpolate_nan(weighted_center).astype(np.float32),
        "ridge_y": interpolate_nan(ridge_y).astype(np.float32),
        "energy_y": interpolate_nan(energy_y).astype(np.float32),
        "height": height.astype(np.float32),
        "local_height": rolling_median(height, max(7, int(round(6 * stroke_width)) | 1)).astype(np.float32),
        "dist_to_ink": cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32),
        "ridge": ridge.astype(np.float32),
        "baseline": baseline.astype(np.float32),
        "depression": np.maximum(0.0, bottom_i - baseline).astype(np.float32),
        "gradient": np.abs(np.gradient(bottom_i)).astype(np.float32),
    }


def event_mask(fields: dict, stroke_width: float, cfg: Config) -> np.ndarray:
    event = fields["gradient"] >= max(2.0, cfg.event_gradient_factor * stroke_width)
    event |= fields["height"] >= np.maximum(
        cfg.event_height_factor * stroke_width,
        1.35 * np.maximum(fields["local_height"], 1.0),
    )
    event |= fields["depression"] >= max(1.0, cfg.event_depression_factor * stroke_width)

    k = max(3, int(round(cfg.event_dilate_factor * stroke_width)) | 1)

    return cv2.morphologyEx(
        event.astype(np.uint8)[None, :] * 255,
        cv2.MORPH_CLOSE,
        np.ones((1, k), np.uint8),
    )[0] > 0


def candidate_cost(
    y: float,
    x: int,
    run_center: float,
    crop_gray: np.ndarray,
    energy: np.ndarray,
    fields: dict,
    stroke_width: float,
    energy_scale: float,
    cfg: Config,
) -> float:
    yy = int(np.clip(round(y), 0, crop_gray.shape[0] - 1))
    center_penalty = abs(float(y) - run_center) / max(1.0, stroke_width)
    white_penalty = np.clip((float(crop_gray[yy, x]) - 210.0) / 45.0, 0.0, 1.0)
    dark = float(energy[yy, x]) / energy_scale
    ridge = float(fields["ridge"][yy, x])

    return float(
        cfg.point_center_penalty * center_penalty
        + cfg.point_white_penalty * white_penalty
        - cfg.point_dark_reward * min(1.0, dark)
        - cfg.point_ridge_reward * ridge
    )


def dedup_candidates(items: list[dict], limit: int, stroke_width: float) -> list[dict]:
    ordered = sorted(items, key=lambda item: item["cost"])
    radius = max(0.45, 0.45 * stroke_width)
    kept = []

    for item in ordered:
        if all(abs(item["y"] - old["y"]) > radius for old in kept):
            kept.append(item)

        if len(kept) >= limit:
            break

    return sorted(kept, key=lambda item: item["y"])


def build_candidates(
    crop_gray: np.ndarray,
    mask: np.ndarray,
    energy: np.ndarray,
    fields: dict,
    stroke_width: float,
    cfg: Config,
):
    h, w = mask.shape
    events = event_mask(fields, stroke_width, cfg)
    energy_scale = max(1.0, float(np.percentile(energy[energy > 0], 95)) if np.any(energy > 0) else 1.0)
    layers = []

    for x in range(w):
        runs = fields["runs_by_x"].get(x)
        items = []
        limit = cfg.max_candidates_event if events[x] else cfg.max_candidates_normal

        if runs:
            for run in runs:
                ys = run["ys"]
                center = float(run["center"])

                values = [
                    int(round(run["top"])),
                    int(round(run["bottom"])),
                    int(round(center)),
                    int(round(fields["weighted_center"][x])),
                    int(round(fields["ridge_y"][x])),
                    int(round(fields["energy_y"][x])),
                ]

                if events[x] and len(ys) > 2:
                    values.extend(np.quantile(ys, cfg.candidate_quantiles).round().astype(int).tolist())

                for y in sorted(set(int(np.clip(value, 0, h - 1)) for value in values)):
                    items.append(
                        {
                            "x": int(x),
                            "y": float(y),
                            "cost": candidate_cost(
                                y,
                                x,
                                center,
                                crop_gray,
                                energy,
                                fields,
                                stroke_width,
                                energy_scale,
                                cfg,
                            ),
                            "is_event": bool(events[x]),
                        }
                    )
        else:
            y0 = int(np.clip(round(fields["weighted_center"][x]), 0, h - 1))
            radius = max(2, int(round(2.5 * stroke_width)))
            step = max(1, int(round(stroke_width)))

            for dy in range(-radius, radius + 1, step):
                y = int(np.clip(y0 + dy, 0, h - 1))
                items.append(
                    {
                        "x": int(x),
                        "y": float(y),
                        "cost": float(cfg.bridge_point_penalty + 0.05 * abs(dy)),
                        "is_event": bool(events[x]),
                    }
                )

        layers.append(dedup_candidates(items, limit, stroke_width))

    return layers, events


def pack_layers_to_arrays(layers: list[list[dict]], cfg: Config):
    w, c = len(layers), cfg.c_max

    y = np.zeros((w, c), dtype=np.float32)
    point_cost = np.full((w, c), np.inf, dtype=np.float32)
    valid = np.zeros((w, c), dtype=bool)
    is_event = np.zeros((w, c), dtype=bool)

    for x, layer in enumerate(layers):
        for k, item in enumerate(sorted(layer, key=lambda item: item["cost"])[:c]):
            y[x, k] = float(item["y"])
            point_cost[x, k] = float(item["cost"])
            valid[x, k] = True
            is_event[x, k] = bool(item.get("is_event", False))

    return {
        "y": y,
        "point_cost": point_cost,
        "valid": valid,
        "is_event": is_event,
    }


def nonlinear_abs_np(values: np.ndarray, knee: float) -> np.ndarray:
    values = np.abs(values).astype(np.float32)
    excess = np.maximum(values - knee, 0.0)
    return np.where(values <= knee, values, knee + np.log1p(excess)).astype(np.float32)


def precompute_edges_numpy(packed: dict, fields: dict, energy: np.ndarray, stroke_width: float, cfg: Config):
    y = packed["y"]
    valid = packed["valid"]

    w, c = y.shape
    h = fields["dist_to_ink"].shape[0]

    edge_cost = np.full((w, c, c), np.inf, dtype=np.float32)
    edge_support = np.zeros((w, c, c), dtype=np.float32)
    slope = np.zeros((w, c, c), dtype=np.float32)

    stroke_radius = max(0.5, cfg.stroke_radius_factor * stroke_width)
    top = fields["top"]
    bottom = fields["bottom"]
    height = fields["height"]
    local_height = fields["local_height"]
    dist = fields["dist_to_ink"]
    ridge = fields["ridge"]

    energy_scale = max(1.0, float(np.percentile(energy[energy > 0], 95)) if np.any(energy > 0) else 1.0)

    for x in range(1, w):
        valid_edge = valid[x - 1, :, None] & valid[x, None, :]
        y0 = y[x - 1, :, None]
        y1 = y[x, None, :]
        dy = y1 - y0
        slope[x] = dy.astype(np.float32)

        if height[x - 1] <= 0 and height[x] <= 0:
            env_cost = np.zeros((c, c), dtype=np.float32)
            coverage = np.ones((c, c), dtype=np.float32)
        else:
            slab_top = min(float(top[x - 1]), float(top[x]))
            slab_bottom = max(float(bottom[x - 1]), float(bottom[x]))
            slab_height = max(1.0, slab_bottom - slab_top + 1.0)

            seg_min = np.minimum(y0, y1) - stroke_radius
            seg_max = np.maximum(y0, y1) + stroke_radius

            miss = np.maximum(0.0, seg_min - slab_top) + np.maximum(0.0, slab_bottom - seg_max)
            coverage = 1.0 - np.minimum(1.0, miss / slab_height)

            tallness = slab_height / max(1.0, stroke_width)
            local_ratio = slab_height / max(1.0, float(max(local_height[x - 1], local_height[x], stroke_width)))

            weight = 1.0

            if tallness >= 2.0:
                weight += min(6.0, 0.95 * tallness)

            if local_ratio >= 1.25:
                weight += min(4.5, local_ratio)

            env_cost = cfg.envelope_miss_weight * weight * (miss / max(1.0, stroke_width)) ** 1.45

        yi0 = np.clip(np.rint(y0[:, 0]).astype(np.int32), 0, h - 1)
        yi1 = np.clip(np.rint(y1[0, :]).astype(np.int32), 0, h - 1)
        yim = np.clip(np.rint(0.5 * (y0 + y1)).astype(np.int32), 0, h - 1)

        off = 0.25 * dist[yi0, x - 1][:, None] + 0.50 * dist[yim, x] + 0.25 * dist[yi1, x][None, :]

        ridge_score = (
            0.25 * ridge[yi0, x - 1][:, None]
            + 0.50 * ridge[yim, x]
            + 0.25 * ridge[yi1, x][None, :]
        )

        energy_score = (
            0.25 * energy[yi0, x - 1][:, None]
            + 0.50 * energy[yim, x]
            + 0.25 * energy[yi1, x][None, :]
        ) / energy_scale

        cost = (
            env_cost
            + cfg.off_ink_weight * off
            + cfg.length_weight * np.hypot(1.0, dy)
            - cfg.edge_ridge_reward * ridge_score
            - cfg.edge_energy_reward * energy_score
        )

        edge_cost[x] = np.where((np.abs(dy) <= cfg.max_abs_slope) & valid_edge, cost, np.inf).astype(np.float32)
        edge_support[x] = np.clip(np.maximum(coverage, ridge_score), 0.0, 1.0).astype(np.float32)

    return {
        "edge_cost": edge_cost,
        "edge_support": edge_support,
        "slope": slope,
    }


def solve_second_order_viterbi_numpy(packed: dict, edges: dict, cfg: Config) -> Curve:
    y = packed["y"]
    point_cost = packed["point_cost"]
    valid = packed["valid"]
    is_event = packed["is_event"]

    edge_cost = edges["edge_cost"]
    edge_support = edges["edge_support"]
    slope = edges["slope"]

    w, c = y.shape

    if w == 0:
        return {
            "px": np.array([], dtype=np.float64),
            "py": np.array([], dtype=np.float64),
            "steep": np.array([], dtype=bool),
        }

    if w == 1:
        k = int(np.nanargmin(point_cost[0]))
        return {
            "px": np.array([0.0], dtype=np.float64),
            "py": np.array([float(y[0, k])], dtype=np.float64),
            "steep": np.array([bool(is_event[0, k])], dtype=bool),
        }

    init_valid = valid[0, :, None] & valid[1, None, :]
    support = edge_support[1]

    dp = (
        point_cost[0, :, None]
        + point_cost[1, None, :]
        + edge_cost[1]
        + cfg.slope_weight * (1.0 - cfg.support_relief * support) * nonlinear_abs_np(slope[1], cfg.nonlinear_knee)
    ).astype(np.float32)

    dp = np.where(init_valid, dp, np.inf).astype(np.float32)
    parents = np.full((w, c, c), -1, dtype=np.int16)

    for x in range(2, w):
        prev_slope = slope[x - 1][:, :, None]
        cur_slope = slope[x][None, :, :]

        support = edge_support[x][None, :, :]
        support = np.where(
            is_event[x - 1, None, :, None] | is_event[x, None, None, :],
            np.maximum(support, cfg.event_support_floor),
            support,
        )

        transition = (
            dp[:, :, None]
            + edge_cost[x][None, :, :]
            + point_cost[x, None, None, :]
            + cfg.slope_weight * (1.0 - cfg.support_relief * support) * nonlinear_abs_np(cur_slope, cfg.nonlinear_knee)
            + cfg.acceleration_weight
            * (1.0 - 0.70 * support)
            * nonlinear_abs_np(cur_slope - prev_slope, cfg.nonlinear_knee)
        )

        invalid = ~(valid[x - 2, :, None, None] & valid[x - 1, None, :, None] & valid[x, None, None, :])
        transition = np.where(invalid, np.inf, transition)

        parent_i = np.argmin(transition, axis=0).astype(np.int16)
        new_dp = np.min(transition, axis=0).astype(np.float32)

        if not np.isfinite(new_dp).any():
            best_j, best_k = np.unravel_index(np.nanargmin(dp), dp.shape)
            new_dp = np.full((c, c), np.inf, dtype=np.float32)

            for k in np.flatnonzero(valid[x]):
                new_dp[best_k, k] = dp[best_j, best_k] + point_cost[x, k] + 25.0
                parent_i[best_k, k] = best_j

        parents[x] = parent_i
        dp = new_dp

    j, k = np.unravel_index(np.nanargmin(dp), dp.shape)

    path = np.zeros(w, dtype=np.int16)
    path[w - 1] = k
    path[w - 2] = j

    for x in range(w - 1, 1, -1):
        i = parents[x, path[x - 1], path[x]]
        path[x - 2] = i if i >= 0 else int(np.nanargmin(point_cost[x - 2]))

    idx = np.arange(w)

    return {
        "px": idx.astype(np.float64),
        "py": y[idx, path].astype(np.float64),
        "steep": is_event[idx, path].astype(bool),
    }


def snap_curve(curve: Curve, mask: np.ndarray, energy: np.ndarray, fields: dict, stroke_width: float, cfg: Config) -> Curve:
    if not cfg.snap_final or len(curve["px"]) == 0:
        return curve

    _, w = mask.shape
    ridge = fields["ridge"]
    radius = max(1, int(round(cfg.snap_radius_factor * stroke_width)))

    py = curve["py"].copy()

    for x in range(w):
        ys = np.flatnonzero(mask[:, x] > 0)

        if len(ys) == 0:
            continue

        near = ys[np.abs(ys.astype(float) - py[x]) <= radius]

        if len(near) == 0:
            continue

        score = (
            cfg.snap_distance_weight * np.abs(near.astype(float) - py[x]) / max(1.0, stroke_width)
            - cfg.snap_energy_weight * energy[near, x].astype(float)
            - cfg.snap_ridge_weight * ridge[near, x].astype(float)
        )

        py[x] = float(near[int(np.argmin(score))])

    return {**curve, "py": py}


def path_abs(curve: Curve, box: Box, cfg: Config) -> Curve:
    return {
        **curve,
        "px_abs": curve["px"] + box.left + cfg.crop_inset,
        "py_abs": curve["py"] + box.top + cfg.crop_inset,
    }


def extract_curve(image: np.ndarray, box: Box, cfg: Config) -> dict:
    crop = crop_plot(image, box, cfg)
    crop_gray = gray(crop)
    energy = ink_energy_from_gray(crop_gray)
    mask = ink_mask_from_gray(crop_gray, energy, cfg)
    stroke_width = estimate_stroke_width(mask, cfg)
    fields = prepare_fields(mask, energy, stroke_width)

    layers, _ = build_candidates(crop_gray, mask, energy, fields, stroke_width, cfg)
    packed = pack_layers_to_arrays(layers, cfg)
    edges = precompute_edges_numpy(packed, fields, energy, stroke_width, cfg)

    curve = solve_second_order_viterbi_numpy(packed, edges, cfg)
    curve = snap_curve(curve, mask, energy, fields, stroke_width, cfg)

    return {
        "stroke_width": stroke_width,
        "curve": curve,
        "curve_abs": path_abs(curve, box, cfg),
    }


@cache
def doctr_model():
    return ocr_predictor(pretrained=True, assume_straight_pages=True)


class DoctrOcr:
    def __call__(self, image: np.ndarray) -> list[OcrWord]:
        result = doctr_model()([image])
        h, w = image.shape[:2]
        words = []

        for block in result.pages[0].blocks:
            for line in block.lines:
                for word in line.words:
                    (x0, y0), (x1, y1) = word.geometry
                    left, top, right, bottom = x0 * w, y0 * h, x1 * w, y1 * h

                    words.append(
                        OcrWord(
                            text=str(word.value),
                            conf=float(word.confidence),
                            left=float(left),
                            top=float(top),
                            right=float(right),
                            bottom=float(bottom),
                            cx=float((left + right) / 2),
                            cy=float((top + bottom) / 2),
                        )
                    )

        return words


def axis_token(text: str) -> str | None:
    raw = str(text).strip()

    if not raw:
        return None

    allowed = set("0123456789LIl|OoDDSs+-−– ")

    if any(ch not in allowed for ch in raw):
        return None

    token = raw.translate(
        str.maketrans(
            {
                "L": "1",
                "I": "1",
                "l": "1",
                "|": "1",
                "O": "0",
                "o": "0",
                "D": "0",
                "S": "5",
                "s": "5",
                "+": "",
                "-": "",
                "−": "",
                "–": "",
                " ": "",
            }
        )
    )

    return token if re.fullmatch(r"\d+", token) else None


def x_candidates(text: str, cfg: Config) -> list[float]:
    token = axis_token(text)

    if token is None:
        return []

    value = float(token)
    lo, hi = cfg.x_value_bounds

    return [value] if lo <= value <= hi else []


def y_candidates(text: str) -> list[float]:
    raw = str(text).strip().upper()

    if raw in {"LOD", "LOO", "L00", "I00", "IOO"}:
        return [100.0]

    token = axis_token(text)

    if token is None:
        return []

    value = float(token)
    legal = set(range(0, 101, 10)) | {5, 6}

    if value in legal:
        return [value]

    if len(token) >= 3:
        return []

    values = set()

    for end in range(1, len(token)):
        value = float(token[:end])

        if value in legal:
            values.add(value)

    return sorted(values)


def tick_ocr(ocr: list[OcrWord], axis: Axis, cfg: Config) -> list[tuple[OcrWord, list[float]]]:
    out = []

    for word in ocr:
        if word.conf < cfg.ocr_min_conf:
            continue

        values = x_candidates(word.text, cfg) if axis == "x" else y_candidates(word.text)

        if values:
            out.append((word, values))

    return out


def group_items(items: list, key, tol: float) -> list[list]:
    groups: list[list] = []

    for item in sorted(items, key=key):
        value = key(item)

        for group in groups:
            center = float(np.median([key(x) for x in group]))

            if abs(value - center) <= tol:
                group.append(item)
                break
        else:
            groups.append([item])

    return groups


def clean_labels(labels: list[AxisLabel]) -> list[AxisLabel]:
    return [label for label in labels if label.inlier]


def fallback_x(box: Box, cfg: Config) -> LinearAxisFit:
    slope = (cfg.fallback_x_end - cfg.fallback_x_start) / max(1, box.width)

    return LinearAxisFit(
        float(slope),
        float(cfg.fallback_x_start - slope * box.left),
        float("nan"),
        2,
        "box_fallback",
    )


def fallback_y(box: Box, cfg: Config) -> LinearAxisFit:
    slope = (cfg.fallback_y_min - cfg.fallback_y_max) / max(1, box.height)

    return LinearAxisFit(
        float(slope),
        float(cfg.fallback_y_max - slope * box.top),
        float("nan"),
        2,
        "box_fallback",
    )


def fit_x(labels: list[AxisLabel]) -> PiecewiseAxisFit | None:
    clean = clean_labels(labels)

    if len(clean) < 3:
        return None

    return PiecewiseAxisFit(
        pixels=np.array([label.tick_pixel for label in clean], dtype=float),
        values=np.array([label.axis_value for label in clean], dtype=float),
        rmse=0.0,
        count=len(clean),
        source="doctr_axis_region_piecewise",
    )


def fit_y_stable(labels: list[AxisLabel], box: Box, cfg: Config) -> LinearAxisFit:
    clean = [
        label
        for label in clean_labels(labels)
        if 0 <= label.axis_value <= 90 and not np.isclose(label.axis_value, 100.0)
    ]

    preferred_values = {90, 80, 70, 60, 50, 40, 30, 20, 10}
    preferred = [label for label in clean if label.axis_value in preferred_values]

    if len(preferred) >= 2:
        clean = preferred

    if len(clean) < 2:
        return fallback_y(box, cfg)

    clean = sorted(clean, key=lambda item: item.axis_value, reverse=True)

    value = np.array([label.axis_value for label in clean], dtype=float)
    pixel = np.array([label.tick_pixel for label in clean], dtype=float)

    slope, intercept = np.polyfit(pixel, value, 1)

    if slope >= 0:
        return fallback_y(box, cfg)

    pred = slope * pixel + intercept
    residual = np.abs(value - pred)

    if len(clean) >= 4:
        keep = residual <= max(2.0, float(np.median(residual) + 2.5 * np.std(residual)))

        if keep.sum() >= 2:
            value = value[keep]
            pixel = pixel[keep]
            slope, intercept = np.polyfit(pixel, value, 1)
            pred = slope * pixel + intercept
            residual = np.abs(value - pred)

    return LinearAxisFit(
        float(slope),
        float(intercept),
        float(np.sqrt(np.mean(residual**2))),
        len(value),
        "doctr_axis_region_linear",
    )


def select_x_ticks(ocr: list[OcrWord], image: np.ndarray, box: Box, cfg: Config) -> list[AxisLabel]:
    h, w = image.shape[:2]

    ticks = [
        (word, values)
        for word, values in tick_ocr(ocr, "x", cfg)
        if box.bottom - h * 0.06 <= word.cy <= min(h, box.bottom + h * 0.12)
        and max(0, box.left - w * 0.08) <= word.cx <= min(w, box.right + w * 0.10)
    ]

    best: list[AxisLabel] = []

    for group in group_items(ticks, key=lambda item: item[0].cy, tol=max(10, h * 0.012)):
        labels = [
            AxisLabel(
                text=word.text,
                axis_value=float(values[0]),
                tick_pixel=float(word.cx),
                left=word.left,
                top=word.top,
                right=word.right,
                bottom=word.bottom,
                conf=word.conf,
                inlier=True,
            )
            for word, values in group
        ]

        if not labels:
            continue

        labels = sorted(labels, key=lambda item: item.tick_pixel)
        monotonic = [labels[0]]

        for label in labels[1:]:
            if label.axis_value < monotonic[-1].axis_value:
                monotonic.append(label)

        if len(monotonic) < 3:
            continue

        values = np.array([label.axis_value for label in monotonic], dtype=float)

        if values.max() - values.min() < 1000:
            continue

        if len(monotonic) > len(best):
            best = monotonic

    return best


def make_inferred_y100_label(original: list[AxisLabel], stable: list[AxisLabel], box: Box, cfg: Config) -> AxisLabel:
    fit = fit_y_stable(stable, box, cfg)
    y100 = fit.pixel_at(100.0)
    source = next((label for label in original if np.isclose(label.axis_value, 100.0)), None)
    base = source or stable[0]

    return AxisLabel(
        text=f"{base.text}*" if source else "100*",
        axis_value=100.0,
        tick_pixel=float(y100),
        left=base.left,
        top=base.top,
        right=base.right,
        bottom=base.bottom,
        conf=base.conf,
        inlier=False,
    )


def select_y_ticks(ocr: list[OcrWord], image: np.ndarray, box: Box, cfg: Config) -> list[AxisLabel]:
    h, w = image.shape[:2]

    ticks = [
        (word, values)
        for word, values in tick_ocr(ocr, "y", cfg)
        if max(0, box.left - w * 0.14) <= word.cx <= min(w, box.left + w * 0.055)
        and max(0, box.top - h * 0.12) <= word.cy <= min(h, box.bottom + h * 0.10)
    ]

    choices: list[tuple[tuple[float, int, float], list[AxisLabel]]] = []

    for group in group_items(ticks, key=lambda item: item[0].cx, tol=max(14, w * 0.012)):
        labels = [
            AxisLabel(
                text=word.text,
                axis_value=float(max(values)),
                tick_pixel=float(word.cy),
                left=word.left,
                top=word.top,
                right=word.right,
                bottom=word.bottom,
                conf=word.conf,
                inlier=True,
            )
            for word, values in group
            if 0 <= max(values) <= 100
        ]

        stable = [
            label
            for label in labels
            if 0 <= label.axis_value <= 90 and not np.isclose(label.axis_value, 100.0)
        ]

        preferred_values = {90, 80, 70, 60, 50, 40, 30, 20, 10}
        preferred = [label for label in stable if label.axis_value in preferred_values]

        if len(preferred) >= 2:
            stable = preferred

        if len(stable) < 2:
            continue

        stable = sorted(stable, key=lambda item: item.axis_value, reverse=True)
        values = np.array([label.axis_value for label in stable], dtype=float)

        if not np.all(np.diff(values) < 0):
            continue

        fit = fit_y_stable(stable, box, cfg)
        pixels = np.array([label.tick_pixel for label in stable], dtype=float)
        residual = np.abs(values - fit.value_at(pixels))

        marked = [
            replace(label, inlier=bool(res <= 8))
            for label, res in zip(stable, residual)
        ]

        clean = clean_labels(marked)

        if len(clean) < 2:
            continue

        display_group = clean + [make_inferred_y100_label(labels, clean, box, cfg)]

        score = (
            float(max(label.axis_value for label in clean) - min(label.axis_value for label in clean)),
            len(clean),
            -fit.rmse,
        )

        choices.append((score, display_group))

    return max(choices, key=lambda item: item[0])[1] if choices else []


def calibrate_axes(image: np.ndarray, box: Box, ocr: list[OcrWord], cfg: Config):
    x_labels = select_x_ticks(ocr, image, box, cfg)
    y_labels = select_y_ticks(ocr, image, box, cfg)

    x_fit = fit_x(x_labels) or fallback_x(box, cfg)
    y_fit = fit_y_stable(y_labels, box, cfg)

    return x_fit, y_fit, x_labels, y_labels


def resample_digitized_arrays(data: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    if n <= 0 or len(data["x"]) <= n:
        return data

    order = np.argsort(data["x"])
    x = data["x"][order]

    keep = np.r_[True, np.diff(x) > 1e-9]
    order = order[keep]
    x = data["x"][order]

    if len(x) <= n:
        desc = np.argsort(data["x"][order])[::-1]
        order = order[desc]
        return {key: value[order] for key, value in data.items()}

    grid = np.linspace(float(x.min()), float(x.max()), n)

    out = {
        "x": grid,
        "y": np.interp(grid, x, data["y"][order]),
        "px_abs": np.interp(grid, x, data["px_abs"][order]),
        "py_abs": np.interp(grid, x, data["py_abs"][order]),
    }

    desc = np.argsort(out["x"])[::-1]
    return {key: value[desc] for key, value in out.items()}


def digitize_curve(curve_abs: Curve, x_fit, y_fit, cfg: Config) -> dict[str, np.ndarray]:
    if len(curve_abs["px_abs"]) == 0:
        return {
            "x": np.array([], dtype=np.float64),
            "y": np.array([], dtype=np.float64),
            "px_abs": np.array([], dtype=np.float64),
            "py_abs": np.array([], dtype=np.float64),
        }

    data = {
        "x": np.asarray(x_fit.value_at(curve_abs["px_abs"]), dtype=np.float64),
        "y": np.asarray(y_fit.value_at(curve_abs["py_abs"]), dtype=np.float64),
        "px_abs": curve_abs["px_abs"].astype(np.float64),
        "py_abs": curve_abs["py_abs"].astype(np.float64),
    }

    order = np.argsort(data["x"])[::-1]
    data = {key: value[order] for key, value in data.items()}

    return resample_digitized_arrays(data, cfg.output_points)


def digitized_frame(data: dict[str, np.ndarray]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": data["x"],
            "y": data["y"],
        }
    )


def overlay_curve_abs(image: np.ndarray, curve_abs: Curve) -> np.ndarray:
    out = image.copy()

    px = curve_abs.get("px_abs", np.array([]))
    py = curve_abs.get("py_abs", np.array([]))

    if len(px) == 0:
        return out

    pts = np.column_stack([px, py]).astype(float)

    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(
            out,
            (int(round(a[0])), int(round(a[1]))),
            (int(round(b[0])), int(round(b[1]))),
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return out


def draw_axis_debug(
    image: np.ndarray,
    box: Box,
    curve_abs: Curve,
    x_labels: list[AxisLabel],
    y_labels: list[AxisLabel],
    x_fit,
    y_fit,
) -> np.ndarray:
    out = overlay_curve_abs(image.copy(), curve_abs)
    cv2.rectangle(out, (box.left, box.top), (box.right, box.bottom), (255, 0, 0), 2)

    for labels, color in [(x_labels, (255, 140, 0)), (y_labels, (0, 180, 255))]:
        for label in labels:
            c = color if label.inlier else (255, 0, 0)

            cv2.rectangle(
                out,
                (int(label.left), int(label.top)),
                (int(label.right), int(label.bottom)),
                c,
                2,
            )

            cv2.putText(
                out,
                f"{label.axis_value:g}",
                (int(label.left), max(15, int(label.top) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                c,
                1,
                cv2.LINE_AA,
            )

            if color == (0, 180, 255):
                cv2.circle(out, (box.left, int(round(label.tick_pixel))), 4, c, -1)
            else:
                cv2.circle(out, (int(round(label.tick_pixel)), box.bottom), 4, c, -1)

    return out


def spectrum_image(data: dict[str, np.ndarray], width: int = 1200, height: int = 700) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, np.uint8)

    if len(data["x"]) == 0:
        return canvas

    ml, mr, mt, mb = 95, 45, 45, 75
    pw = width - ml - mr
    ph = height - mt - mb

    x = data["x"].astype(float)
    y = data["y"].astype(float)

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min = min(0.0, float(np.nanmin(y)))
    y_max = max(100.0, float(np.nanmax(y)))

    px = ml + (x_max - x) / max(1e-9, x_max - x_min) * pw
    py = mt + (y_max - y) / max(1e-9, y_max - y_min) * ph

    left, right = ml, ml + pw
    top, bottom = mt, mt + ph

    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 0, 0), 1)

    x_ticks = np.linspace(x_max, x_min, 8)
    y_ticks = np.linspace(y_min, y_max, 6)

    for value in x_ticks:
        tx = int(round(left + (x_max - value) / max(1e-9, x_max - x_min) * pw))
        cv2.line(canvas, (tx, bottom), (tx, bottom + 6), (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{value:.0f}",
            (tx - 24, bottom + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    for value in y_ticks:
        ty = int(round(top + (y_max - value) / max(1e-9, y_max - y_min) * ph))
        cv2.line(canvas, (left - 6, ty), (left, ty), (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{value:.0f}",
            (left - 48, ty + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    pts = np.column_stack([px, py]).astype(np.int32)

    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(canvas, tuple(a), tuple(b), (255, 0, 0), 1, cv2.LINE_AA)

    return canvas

def digitize(image_path: Path, output_dir: Path, cfg: Config) -> dict:
    validate_input(image_path)

    image = read_image(image_path)
    box, box_type = detect_plot_box(image, cfg)
    trace = extract_curve(image, box, cfg)

    ocr_engine = DoctrOcr()
    ocr = ocr_engine(image)
    x_fit, y_fit, x_labels, y_labels = calibrate_axes(image, box, ocr, cfg)
    data = digitize_curve(trace["curve_abs"], x_fit, y_fit, cfg)

    output_dir.mkdir(parents=True, exist_ok=True)

    axis_debug_path = output_dir / "axis_debug.png"
    spectrum_path = output_dir / "spectrum.png"
    csv_path = output_dir / "digitized_spectrum.csv"

    save_rgb(axis_debug_path, draw_axis_debug(image, box, trace["curve_abs"], x_labels, y_labels, x_fit, y_fit))
    save_rgb(spectrum_path, spectrum_image(data))
    digitized_frame(data).write_csv(csv_path)

    return {
        "source": str(image_path),
        "box_type": box_type,
        "box": box,
        "stroke_width": float(trace["stroke_width"]),
        "x_fit_source": getattr(x_fit, "source", type(x_fit).__name__),
        "x_fit_count": int(x_fit.count),
        "x_left_value": float(x_fit.value_at(box.left)),
        "x_right_value": float(x_fit.value_at(box.right)),
        "y_fit_source": y_fit.source,
        "y_fit_count": int(y_fit.count),
        "y_fit_rmse": float(y_fit.rmse),
        "y_top_value": float(y_fit.value_at(box.top)),
        "y_bottom_value": float(y_fit.value_at(box.bottom)),
        "data_x_min": float(np.nanmin(data["x"])) if len(data["x"]) else np.nan,
        "data_x_max": float(np.nanmax(data["x"])) if len(data["x"]) else np.nan,
        "data_y_min": float(np.nanmin(data["y"])) if len(data["y"]) else np.nan,
        "data_y_max": float(np.nanmax(data["y"])) if len(data["y"]) else np.nan,
        "points": len(data["x"]),
        "axis_debug": str(axis_debug_path),
        "spectrum": str(spectrum_path),
        "digitized_csv": str(csv_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Digitize an IR spectrum image into a numerical spectrum.")
    parser.add_argument("--input", required=True, type=Path, help="Input spectrum image.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory.")
    parser.add_argument("--output-points", type=int, default=1800)
    parser.add_argument("--fallback-x-start", type=float, default=4000.0)
    parser.add_argument("--fallback-x-end", type=float, default=500.0)
    parser.add_argument("--fallback-y-min", type=float, default=0.0)
    parser.add_argument("--fallback-y-max", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        output_points=args.output_points,
        fallback_x_start=args.fallback_x_start,
        fallback_x_end=args.fallback_x_end,
        fallback_y_min=args.fallback_y_min,
        fallback_y_max=args.fallback_y_max,
    )

    row = digitize(args.input, args.out, cfg)
    box = row["box"]

    print("OK:", row["source"])
    print("box:", box.left, box.top, box.right, box.bottom, row["box_type"])
    print("stroke_width:", round(row["stroke_width"], 3))
    print("x_fit:", row["x_fit_source"], "count:", row["x_fit_count"])
    print("x_range:", row["x_left_value"], "->", row["x_right_value"])
    print("y_fit:", row["y_fit_source"], "count:", row["y_fit_count"], "rmse:", row["y_fit_rmse"])
    print("y_range:", row["y_top_value"], "->", row["y_bottom_value"])
    print("data:", row["data_x_min"], row["data_x_max"], row["data_y_min"], row["data_y_max"])
    print("points:", row["points"])
    print("axis_debug:", row["axis_debug"])
    print("spectrum:", row["spectrum"])
    print("digitized_csv:", row["digitized_csv"])


if __name__ == "__main__":
    main()