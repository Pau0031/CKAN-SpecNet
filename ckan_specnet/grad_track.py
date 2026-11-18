"""Minority-class gradient logging (functional groups with prevalence < 5%) + training loss logging.

Design rationale: the Poly1 loss used in this repository

    l_i = w_{y_i} * CE_i + eps * (1 - p_{i,y_i}),      CE_i = -log p_{i,y_i}

Differentiating with respect to the logits:

    dl_i / dz_i = (w_{y_i} + eps * p_{i,y_i}) * (p_i - onehot(y_i))

The CE term and the Poly1 term are therefore **collinear** in logits space: Poly1
scales the CE gradient per sample by (w + eps*p_t) / w. "Minority-class gradient"
must consequently be logged as several mutually independent pieces of information:

1. Per-sample probability / gradient scaling factor: p_t, scale = scale_ce + scale_poly,
   and the poly/CE amplification ratio.
   -- shows how much extra gradient Poly1 actually gives the minority class.
2. Net gradient in logits space: G_c = 1/B * sum_{i in c} g_i,
   logging its norm, its share of the "gradient mass" over all classes of the
   task, and its cosine with the task-total net gradient G_all.
   -- shows in which direction the minority class pushes the logits and whether
   that conflicts with the overall direction.
3. Parameter-gradient attribution of the last classifier layer (the last Linear of
   base_heads[task]): A_c = 1/B * sum_{i in c} g_i (x) h_i, where h is the input of
   that Linear. It satisfies sum_c A_c = dL_task/dW exactly, so its norm, share and
   cosine with the total gradient can be logged.
   -- shows the minority class's gradient contribution and conflict in **parameter
   space** (the quantity gradient-surgery-style analyses care about).
4. Gradient propagated back to the shared representation: U_c = G_c @ W,
   logging its norm, share and cosine with the total propagated gradient.
   -- shows how strongly the minority class pulls the shared backbone.

Every statistic is a detached side computation and does not affect the loss or the
backward pass; the aggregation only accumulates tensors and synchronises with the
CPU once per epoch.

RunLogger is included as well: it appends the training loss curves and the gradient
statistics above row by row into CSV files.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ckan_specnet.core import TaskCatalog, unwrap_model

# --------------------------------------------------------------------------------------
# Functional-group prevalence (from the paper's statistics table; see the internal
# minority-class gradient / loss logging note)
# --------------------------------------------------------------------------------------
FUNCTIONAL_GROUP_PRESENCE_RATE: dict[str, float] = {
    "alkane": 83.23,
    "alkene": 13.70,
    "alkyne": 1.75,
    "aromatics": 62.00,
    "alkyl_halides": 27.91,
    "alcohols": 32.90,
    "esters": 13.75,
    "ketones": 9.66,
    "aldehydes": 2.97,
    "carbonyl_oxygen": 24.75,
    "ether": 17.99,
    "acyl_halides": 1.34,
    "amines": 22.28,
    "amides": 2.99,
    "nitriles": 4.11,
    "nitro": 5.72,
    "isocyanate": 0.78,
    "isothiocyanate": 0.91,
    "ortho": 28.54,
    "meta": 27.41,
    "para": 30.80,
}

DEFAULT_MINORITY_THRESHOLD_PCT = 5.0

# CSV column definitions (this order is the column order in the files)
TRAIN_LOG_COLUMNS = [
    "fold",
    "epoch",
    "steps",
    "samples",
    "train_loss",
    "train_loss_ce",
    "train_loss_poly",
    "train_loss_extra_ce",
    "train_reg_loss",
    "grad_norm",
    "lr",
    "val_loss",
    "val_acc",
    "val_f1",
    "val_balanced_accuracy",
    "best_score",
    "is_best",
    "early_stop",
    "epoch_seconds",
]

LOSS_BY_TASK_COLUMNS = [
    "fold",
    "epoch",
    "task",
    "train_loss",
    "train_loss_ce",
    "train_loss_poly",
    "train_loss_extra_ce",
    "val_loss",
]

GRAD_COLUMNS = [
    # identity
    "fold",
    "epoch",
    "task",
    "class_id",
    "group",
    "presence_rate_pct",
    "selection",
    # sample count / batch coverage
    "n_samples",
    "sample_frac",
    "n_steps",
    "n_steps_with_class",
    "frac_steps_present",
    # loss configuration (carried by every row so poly1 / ce runs can be compared directly)
    "epsilon",
    "extra_ce_weight",
    "extra_ce_use_class_weight",
    "use_class_weight",
    "class_weight_coeff",
    # loss values (per-sample mean over the samples of this class)
    "loss_total",
    "loss_ce",
    "loss_poly",
    "loss_extra_ce",
    # probability / gradient scaling factors
    "mean_pt",
    "std_pt",
    "p10_pt",
    "p50_pt",
    "p90_pt",
    "mean_scale",
    "std_scale",
    "mean_poly_over_ce_grad",
    # per-sample logits gradient norm
    "sample_grad_norm_mean",
    "sample_grad_norm_std",
    "sample_grad_norm_p50",
    "sample_grad_norm_p90",
    # net gradient in logits space (G_c)
    "net_grad_norm_step_mean",
    "net_grad_norm_step_std",
    "grad_share_step_mean",
    "grad_share_step_std",
    "cos_grad_to_task_total_step_mean",
    "cos_grad_to_task_total_step_std",
    "net_grad_norm_epoch",
    "grad_share_epoch",
    "cos_grad_to_task_total_epoch",
    # parameter-gradient attribution of the last classifier layer (A_c)
    "head_grad_norm_step_mean",
    "head_grad_share_step_mean",
    "cos_head_to_task_total_step_mean",
    "head_grad_norm_epoch",
    "head_grad_share_epoch",
    "cos_head_to_task_total_epoch",
    # gradient propagated to the shared representation (U_c)
    "head_input_grad_norm_step_mean",
    "head_input_grad_share_step_mean",
    "cos_head_input_to_task_total_step_mean",
    "head_input_grad_norm_epoch",
    "head_input_grad_share_epoch",
    "cos_head_input_to_task_total_epoch",
]

# Maximum number of per-sample values kept per class (for quantiles); beyond this
# no further values are collected and only mean/std are kept.
_MAX_VALUES_PER_CLASS = 200_000


def group_of_task(task: str) -> str:
    for suffix in ("_4class", "_3class"):
        if task.endswith(suffix):
            return task[: -len(suffix)]
    return task


@dataclass(frozen=True)
class MinoritySpec:
    """One (task, class) combination that is tracked."""

    task: str
    class_id: int
    group: str
    presence_rate_pct: float
    selection: str

    @property
    def key(self) -> str:
        return f"{self.task}#{self.class_id}"

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "class_id": self.class_id,
            "group": self.group,
            "presence_rate_pct": self.presence_rate_pct,
            "selection": self.selection,
        }


def task_class_ids_for_group(group: str, tasks: TaskCatalog) -> list[tuple[str, int]]:
    """Every "presence / multiplicity" class of one functional group (class 0 is always
    "absent" and therefore never a minority class)."""
    out: list[tuple[str, int]] = []

    if group in tasks.binary:
        out.append((group, 1))

    if group in tasks.ternary:
        task = f"{group}_3class"
        out.extend((task, class_id) for class_id in range(1, tasks.num_classes[task]))

    if group in tasks.quaternary:
        task = f"{group}_4class"
        out.extend((task, class_id) for class_id in range(1, tasks.num_classes[task]))

    return out


def minority_specs(
    tasks: TaskCatalog,
    *,
    threshold_pct: float = DEFAULT_MINORITY_THRESHOLD_PCT,
    mode: str = "group",
    class_counts: dict[str, np.ndarray] | None = None,
    groups: list[str] | None = None,
) -> list[MinoritySpec]:
    """Select the (task, class) entries whose gradients are logged.

    mode="group" (default): use the functional-group prevalence table, take the
        groups with prevalence < threshold_pct and log their binary positive class
        plus every non-zero class of their 3-/4-class tasks.
    mode="class": use the actual class frequencies of the training labels and log
        every class with frequency < threshold_pct%.
    """
    specs: list[MinoritySpec] = []
    seen: set[str] = set()

    if mode == "group":
        selected = (
            list(groups)
            if groups is not None
            else [
                group
                for group, rate in FUNCTIONAL_GROUP_PRESENCE_RATE.items()
                if rate < threshold_pct
            ]
        )
        for group in selected:
            rate = FUNCTIONAL_GROUP_PRESENCE_RATE.get(group, float("nan"))
            for task, class_id in task_class_ids_for_group(group, tasks):
                spec = MinoritySpec(task, class_id, group, float(rate), "group_presence_table")
                if spec.key not in seen:
                    seen.add(spec.key)
                    specs.append(spec)

    elif mode == "class":
        if class_counts is None:
            raise ValueError("mode='class' requires class_counts (from the training labels)")
        for task in tasks.all:
            counts = class_counts.get(task)
            if counts is None:
                continue
            counts = np.asarray(counts, dtype=np.float64)
            total = counts.sum()
            if total <= 0:
                continue
            for class_id, count in enumerate(counts):
                if count <= 0:
                    continue
                pct = 100.0 * count / total
                if pct < threshold_pct:
                    spec = MinoritySpec(
                        task, int(class_id), group_of_task(task), float(pct), "class_frequency"
                    )
                    if spec.key not in seen:
                        seen.add(spec.key)
                        specs.append(spec)

    else:
        raise ValueError(f"Unknown minority mode: {mode}")

    order = {task: index for index, task in enumerate(tasks.all)}
    specs.sort(key=lambda spec: (order.get(spec.task, len(order)), spec.class_id))

    return specs


def class_counts_from_targets(
    y: dict[str, np.ndarray],
    tasks: TaskCatalog,
) -> dict[str, np.ndarray]:
    return {
        task: np.bincount(
            np.asarray(y[task], dtype=np.int64), minlength=tasks.num_classes[task]
        )
        for task in tasks.all
    }


def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = a.norm() * b.norm()
    cos = (a * b).sum() / denom.clamp_min(1e-12)
    return torch.where(denom > 0, cos, denom.new_full((), float("nan")))


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1e-12),
        denominator.new_full((), float("nan")),
    )


def _as_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


class _ClassAccumulator:
    """Accumulator for one (task, class) within a single epoch (tensor accumulation, no per-step sync)."""

    def __init__(self, task: str, class_id: int, group: str, presence_rate_pct: float, selection: str):
        self.task = task
        self.class_id = class_id
        self.group = group
        self.presence_rate_pct = presence_rate_pct
        self.selection = selection

        self.steps = 0
        self.present_steps = 0
        self.n = 0.0

        self._t: dict[str, torch.Tensor] = {}
        self._step: dict[str, list[torch.Tensor]] = {}
        self.pt_chunks: list[torch.Tensor] = []
        self.gn_chunks: list[torch.Tensor] = []
        self.n_values = 0

        self.g_sum: torch.Tensor | None = None
        self.a_sum: torch.Tensor | None = None
        self.u_sum: torch.Tensor | None = None
        self.g_mass_sum: torch.Tensor | None = None
        self.a_mass_sum: torch.Tensor | None = None
        self.u_mass_sum: torch.Tensor | None = None

    # ---- accumulate ----
    def add(self, name: str, value: torch.Tensor) -> None:
        current = self._t.get(name)
        self._t[name] = value if current is None else current + value

    def add_step(self, name: str, value: torch.Tensor) -> None:
        self._step.setdefault(name, []).append(value.reshape(()))

    def add_vector(self, name: str, value: torch.Tensor | None) -> None:
        if value is None:
            return
        current = getattr(self, name)
        setattr(self, name, value.detach().clone() if current is None else current + value)

    def add_mass(self, name: str, value: torch.Tensor | None) -> None:
        if value is None:
            return
        current = getattr(self, name)
        setattr(self, name, value if current is None else current + value)

    def reset(self) -> None:
        self.steps = 0
        self.present_steps = 0
        self.n = 0.0
        self._t.clear()
        self._step.clear()
        self.pt_chunks.clear()
        self.gn_chunks.clear()
        self.n_values = 0
        self.g_sum = None
        self.a_sum = None
        self.u_sum = None
        self.g_mass_sum = None
        self.a_mass_sum = None
        self.u_mass_sum = None

    def maybe_collect(self, pt: torch.Tensor, gn: torch.Tensor) -> None:
        if self.n_values >= _MAX_VALUES_PER_CLASS:
            return
        self.pt_chunks.append(pt)
        self.gn_chunks.append(gn)
        self.n_values += int(pt.numel())

    # ---- read out ----
    def total(self, name: str) -> torch.Tensor | None:
        return self._t.get(name)

    def sum_value(self, name: str) -> float:
        value = self._t.get(name)
        return _as_float(value) if value is not None else float("nan")

    def mean_value(self, name: str, count: float) -> float:
        value = self._t.get(name)
        if value is None or count <= 0:
            return float("nan")
        return _as_float(value) / count

    def mean_std(self, sum_name: str, sq_name: str, count: float) -> tuple[float, float]:
        total = self._t.get(sum_name)
        sq_total = self._t.get(sq_name)
        if total is None or sq_total is None or count <= 0:
            return float("nan"), float("nan")
        mean = _as_float(total) / count
        var = _as_float(sq_total) / count - mean * mean
        return mean, math.sqrt(var) if var > 0 else 0.0

    def step_mean_std(self, name: str) -> tuple[float, float]:
        values = self._step.get(name)
        if not values:
            return float("nan"), float("nan")
        stacked = torch.stack(values)
        return _as_float(stacked.mean()), _as_float(stacked.std(unbiased=False))

    def quantiles(self, chunks: list[torch.Tensor], qs=(0.1, 0.5, 0.9)) -> list[float]:
        if not chunks:
            return [float("nan")] * len(qs)
        values = torch.cat(chunks)
        if values.numel() == 0:
            return [float("nan")] * len(qs)
        q = torch.tensor(list(qs), dtype=values.dtype, device=values.device)
        return [_as_float(v) for v in torch.quantile(values, q)]


class GradientTracker:
    """Log the gradient statistics of the minority (task, class) entries."""

    def __init__(
        self,
        tasks: TaskCatalog,
        specs: list[MinoritySpec],
        losses: dict[str, nn.Module],
        *,
        track_params: bool = True,
        fold: int = 1,
    ):
        self.tasks = tasks
        self.losses = losses
        self.specs = list(specs)
        self.specs_by_task: dict[str, list[MinoritySpec]] = {}
        for spec in self.specs:
            self.specs_by_task.setdefault(spec.task, []).append(spec)

        self.fold = int(fold)
        self.epoch = 0
        self.track_params = bool(track_params)

        self._capturing = False
        self._head_weight: dict[str, torch.Tensor] = {}
        self._head_inputs: dict[str, torch.Tensor] = {}
        self._handles: list = []
        self._acc: dict[str, _ClassAccumulator] = {}
        self._all_acc: dict[str, _ClassAccumulator] = {}
        self._epoch_samples = 0
        self._step_samples_counted = False

        for spec in self.specs:
            self._acc[spec.key] = _ClassAccumulator(
                spec.task,
                spec.class_id,
                spec.group,
                float(spec.presence_rate_pct),
                spec.selection,
            )
        for task in self.specs_by_task:
            self._all_acc[task] = _ClassAccumulator(
                task, -1, group_of_task(task), float("nan"), "task_all_classes"
            )

    # ------------------------------------------------------------------ attach hooks
    def attach(self, model: nn.Module) -> None:
        """Attach a forward hook to the last Linear of base_heads[task] to capture its input h."""
        if not self.track_params:
            return

        base = unwrap_model(model)
        heads = getattr(base, "base_heads", None)
        if heads is None:
            self.track_params = False
            return

        for task in self.specs_by_task:
            head = heads[task] if task in heads else None
            last = None
            if isinstance(head, nn.Sequential) and len(head) > 0 and isinstance(head[-1], nn.Linear):
                last = head[-1]
            if last is None:
                continue
            self._head_weight[task] = last.weight
            self._handles.append(last.register_forward_hook(self._make_hook(task)))

        if not self._head_weight:
            self.track_params = False

    def _make_hook(self, task: str):
        def hook(_module, inputs, _output):
            if self._capturing and inputs and torch.is_tensor(inputs[0]):
                self._head_inputs[task] = inputs[0].detach()

        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._head_inputs.clear()

    @property
    def tasks_with_params(self) -> list[str]:
        return sorted(self._head_weight)

    # ------------------------------------------------------------------ epoch / step
    def begin_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._epoch_samples = 0
        self._capturing = False
        self._head_inputs.clear()
        for acc in list(self._acc.values()) + list(self._all_acc.values()):
            acc.reset()

    def begin_step(self) -> None:
        self._capturing = True
        self._step_samples_counted = False
        self._head_inputs.clear()

    def end_step(self) -> None:
        self._capturing = False
        self._head_inputs.clear()

    # ------------------------------------------------------------------ statistics
    def observe_task(self, task: str, stats: dict, target: torch.Tensor) -> None:
        """Called after the loss is computed (does not touch the graph, detached side statistics only)."""
        if not self._capturing:
            return

        specs = self.specs_by_task.get(task)
        if not specs:
            return

        prob = stats.get("prob")
        if prob is None or prob.dim() != 2 or prob.shape[0] == 0:
            return

        target = target.detach()
        batch_size, n_classes = prob.shape

        if not self._step_samples_counted:
            self._epoch_samples += int(batch_size)
            self._step_samples_counted = True

        pt = stats["pt"]
        scale = stats["scale"]
        scale_ce = stats["scale_ce"]
        scale_poly = stats["scale_poly"]
        per_sample = stats["per_sample"]
        ce = stats["ce"]
        poly = stats["poly"]
        extra = stats["extra_ce"]

        onehot = torch.zeros_like(prob)
        onehot.scatter_(1, target.view(-1, 1), 1.0)
        # per-sample logits gradient: dl_i / dz_i = scale_i * (p_i - onehot(y_i))
        grad = scale.unsqueeze(1) * (prob - onehot)
        sample_grad_norm = grad.norm(dim=1)

        counts = torch.bincount(target, minlength=n_classes).tolist()

        # net gradient G_c of every class and the "gradient mass" denominator
        class_grad: list[torch.Tensor | None] = [None] * n_classes
        grad_norms: list[torch.Tensor] = []
        class_masks: list[torch.Tensor | None] = [None] * n_classes
        for class_id in range(n_classes):
            if counts[class_id] == 0:
                continue
            mask = target == class_id
            class_masks[class_id] = mask
            net = grad[mask].sum(0) / batch_size
            class_grad[class_id] = net
            grad_norms.append(net.norm())
        grad_all = grad.sum(0) / batch_size
        grad_mass = torch.stack(grad_norms).sum() if grad_norms else grad_all.new_zeros(())

        # parameter-space attribution A_c = 1/B sum_{i in c} g_i (x) h_i; propagated gradient U_c = G_c @ W
        head_input = self._head_inputs.get(task)
        head_weight = self._head_weight.get(task)
        use_params = (
            self.track_params
            and head_input is not None
            and head_weight is not None
            and head_input.shape[0] == batch_size
        )

        class_head: list[torch.Tensor | None] = [None] * n_classes
        class_input_grad: list[torch.Tensor | None] = [None] * n_classes
        head_all = None
        head_mass = None
        input_all = None
        input_mass = None

        if use_params:
            # note the detach: head_weight is a Parameter, without detach the graph would hang off the statistics
            head_weight_detached = head_weight.detach()
            head_all = (grad.t() @ head_input) / batch_size
            head_norms = []
            input_norms = []
            for class_id in range(n_classes):
                if class_grad[class_id] is None:
                    continue
                mask = class_masks[class_id]
                head_class = (grad[mask].t() @ head_input[mask]) / batch_size
                class_head[class_id] = head_class
                head_norms.append(head_class.norm())
                input_class = class_grad[class_id] @ head_weight_detached
                class_input_grad[class_id] = input_class
                input_norms.append(input_class.norm())
            head_mass = torch.stack(head_norms).sum() if head_norms else grad_all.new_zeros(())
            input_all = grad_all @ head_weight_detached
            input_mass = torch.stack(input_norms).sum() if input_norms else grad_all.new_zeros(())

        for spec in specs:
            acc = self._acc[spec.key]
            acc.steps += 1
            class_id = spec.class_id
            if counts[class_id] == 0:
                continue

            acc.present_steps += 1
            acc.n += float(counts[class_id])
            mask = class_masks[class_id]

            self._accumulate_samples(
                acc,
                pt=pt[mask],
                scale=scale[mask],
                scale_ce=scale_ce[mask],
                scale_poly=scale_poly[mask],
                sample_grad_norm=sample_grad_norm[mask],
                per_sample=per_sample[mask],
                ce=ce[mask],
                poly=poly[mask],
                extra=extra[mask],
            )

            net = class_grad[class_id]
            head_class = class_head[class_id]
            input_class = class_input_grad[class_id]

            acc.add_vector("g_sum", net)
            acc.add_vector("a_sum", head_class)
            acc.add_vector("u_sum", input_class)
            acc.add_mass("g_mass_sum", grad_mass)

            self._accumulate_step_metrics(
                acc,
                net=net,
                grad_all=grad_all,
                grad_mass=grad_mass,
                head_class=head_class,
                head_all=head_all,
                head_mass=head_mass,
                input_class=input_class,
                input_all=input_all,
                input_mass=input_mass,
            )

        # whole task (class_id = -1) as the reference row
        acc_all = self._all_acc[task]
        acc_all.steps += 1
        acc_all.present_steps += 1
        acc_all.n += float(batch_size)

        self._accumulate_samples(
            acc_all,
            pt=pt,
            scale=scale,
            scale_ce=scale_ce,
            scale_poly=scale_poly,
            sample_grad_norm=sample_grad_norm,
            per_sample=per_sample,
            ce=ce,
            poly=poly,
            extra=extra,
        )

        acc_all.add_vector("g_sum", grad_all)
        acc_all.add_vector("a_sum", head_all)
        acc_all.add_vector("u_sum", input_all)
        acc_all.add_mass("g_mass_sum", grad_mass)
        acc_all.add_mass("a_mass_sum", head_mass)
        acc_all.add_mass("u_mass_sum", input_mass)

        self._accumulate_step_metrics(
            acc_all,
            net=grad_all,
            grad_all=grad_all,
            grad_mass=grad_mass,
            head_class=head_all,
            head_all=head_all,
            head_mass=head_mass,
            input_class=input_all,
            input_all=input_all,
            input_mass=input_mass,
        )

    @staticmethod
    def _accumulate_samples(
        acc: _ClassAccumulator,
        *,
        pt: torch.Tensor,
        scale: torch.Tensor,
        scale_ce: torch.Tensor,
        scale_poly: torch.Tensor,
        sample_grad_norm: torch.Tensor,
        per_sample: torch.Tensor,
        ce: torch.Tensor,
        poly: torch.Tensor,
        extra: torch.Tensor,
    ) -> None:
        acc.maybe_collect(pt, sample_grad_norm)
        acc.add("pt_sum", pt.sum())
        acc.add("pt_sq", (pt * pt).sum())
        acc.add("scale_sum", scale.sum())
        acc.add("scale_sq", (scale * scale).sum())
        acc.add("gn_sum", sample_grad_norm.sum())
        acc.add("gn_sq", (sample_grad_norm * sample_grad_norm).sum())
        # gradient amplification introduced by Poly1 (per sample, then averaged): scale_poly / scale_ce
        acc.add("poly_over_ce_sum", (scale_poly / scale_ce.clamp_min(1e-12)).sum())
        acc.add("loss_sum", per_sample.sum())
        acc.add("ce_sum", ce.sum())
        acc.add("poly_sum", poly.sum())
        acc.add("extra_ce_sum", extra.sum())

    @staticmethod
    def _accumulate_step_metrics(
        acc: _ClassAccumulator,
        *,
        net: torch.Tensor,
        grad_all: torch.Tensor,
        grad_mass: torch.Tensor,
        head_class: torch.Tensor | None,
        head_all: torch.Tensor | None,
        head_mass: torch.Tensor | None,
        input_class: torch.Tensor | None,
        input_all: torch.Tensor | None,
        input_mass: torch.Tensor | None,
    ) -> None:
        net_norm = net.norm()
        acc.add_step("net_grad_norm", net_norm)
        acc.add_step("grad_share", _safe_ratio(net_norm, grad_mass))
        acc.add_step("cos_grad_to_task_total", _safe_cos(net, grad_all))

        if head_class is not None and head_all is not None and head_mass is not None:
            head_norm = head_class.norm()
            acc.add_step("head_grad_norm", head_norm)
            acc.add_step("head_grad_share", _safe_ratio(head_norm, head_mass))
            acc.add_step("cos_head_to_task_total", _safe_cos(head_class, head_all))

        if input_class is not None and input_all is not None and input_mass is not None:
            input_norm = input_class.norm()
            acc.add_step("head_input_grad_norm", input_norm)
            acc.add_step("head_input_grad_share", _safe_ratio(input_norm, input_mass))
            acc.add_step("cos_head_input_to_task_total", _safe_cos(input_class, input_all))

    # ------------------------------------------------------------------ output
    def _loss_meta(self, task: str, class_id: int) -> dict:
        loss = self.losses.get(task)
        epsilon = _as_float(getattr(loss, "epsilon", float("nan")))
        extra_ce_weight = _as_float(getattr(loss, "ce_weight", float("nan")), default=0.0)
        extra_ce_use_class_weight = int(bool(getattr(loss, "ce_use_class_weight", False)))
        use_class_weight = int(bool(getattr(loss, "use_class_weight", True)))
        weight = getattr(loss, "weight", None)

        coeff = 1.0
        if weight is not None and 0 <= class_id < len(weight):
            coeff = _as_float(weight[class_id])

        return {
            "epsilon": epsilon,
            "extra_ce_weight": extra_ce_weight,
            "extra_ce_use_class_weight": extra_ce_use_class_weight,
            "use_class_weight": use_class_weight,
            "class_weight_coeff": coeff,
        }

    def _row(self, acc: _ClassAccumulator, mass_ref: _ClassAccumulator, task: str) -> dict:
        n = acc.n
        row = {
            "fold": self.fold,
            "epoch": self.epoch,
            "task": acc.task,
            "class_id": acc.class_id,
            "group": acc.group,
            "presence_rate_pct": acc.presence_rate_pct,
            "selection": acc.selection,
            "n_samples": n,
            "sample_frac": (n / self._epoch_samples) if self._epoch_samples > 0 else float("nan"),
            "n_steps": acc.steps,
            "n_steps_with_class": acc.present_steps,
            "frac_steps_present": (acc.present_steps / acc.steps) if acc.steps > 0 else float("nan"),
        }
        row.update(self._loss_meta(task, acc.class_id))

        row["loss_total"] = acc.mean_value("loss_sum", n)
        row["loss_ce"] = acc.mean_value("ce_sum", n)
        row["loss_poly"] = acc.mean_value("poly_sum", n)
        row["loss_extra_ce"] = acc.mean_value("extra_ce_sum", n)

        mean_pt, std_pt = acc.mean_std("pt_sum", "pt_sq", n)
        row["mean_pt"] = mean_pt
        row["std_pt"] = std_pt
        p10, p50, p90 = acc.quantiles(acc.pt_chunks)
        row["p10_pt"], row["p50_pt"], row["p90_pt"] = p10, p50, p90

        mean_scale, std_scale = acc.mean_std("scale_sum", "scale_sq", n)
        row["mean_scale"] = mean_scale
        row["std_scale"] = std_scale
        row["mean_poly_over_ce_grad"] = acc.mean_value("poly_over_ce_sum", n)

        gn_mean, gn_std = acc.mean_std("gn_sum", "gn_sq", n)
        row["sample_grad_norm_mean"] = gn_mean
        row["sample_grad_norm_std"] = gn_std
        gn_p50, gn_p90 = acc.quantiles(acc.gn_chunks, qs=(0.5, 0.9))
        row["sample_grad_norm_p50"] = gn_p50
        row["sample_grad_norm_p90"] = gn_p90

        for prefix in ("net_grad_norm", "grad_share", "cos_grad_to_task_total"):
            mean, std = acc.step_mean_std(prefix)
            row[f"{prefix}_step_mean"] = mean
            row[f"{prefix}_step_std"] = std

        row["net_grad_norm_epoch"] = _as_float(acc.g_sum.norm()) if acc.g_sum is not None else float("nan")
        if acc.g_sum is not None and mass_ref.g_mass_sum is not None:
            row["grad_share_epoch"] = _as_float(
                _safe_ratio(acc.g_sum.norm(), mass_ref.g_mass_sum)
            )
            row["cos_grad_to_task_total_epoch"] = _as_float(
                _safe_cos(acc.g_sum, mass_ref.g_sum)
            )
        else:
            row["grad_share_epoch"] = float("nan")
            row["cos_grad_to_task_total_epoch"] = float("nan")

        for prefix in ("head_grad_norm", "head_grad_share", "cos_head_to_task_total"):
            mean, std = acc.step_mean_std(prefix)
            row[f"{prefix}_step_mean"] = mean

        if acc.a_sum is not None and mass_ref.a_mass_sum is not None and mass_ref.a_sum is not None:
            row["head_grad_norm_epoch"] = _as_float(acc.a_sum.norm())
            row["head_grad_share_epoch"] = _as_float(
                _safe_ratio(acc.a_sum.norm(), mass_ref.a_mass_sum)
            )
            row["cos_head_to_task_total_epoch"] = _as_float(_safe_cos(acc.a_sum, mass_ref.a_sum))
        else:
            row["head_grad_norm_epoch"] = float("nan")
            row["head_grad_share_epoch"] = float("nan")
            row["cos_head_to_task_total_epoch"] = float("nan")

        for prefix in (
            "head_input_grad_norm",
            "head_input_grad_share",
            "cos_head_input_to_task_total",
        ):
            mean, std = acc.step_mean_std(prefix)
            row[f"{prefix}_step_mean"] = mean

        if acc.u_sum is not None and mass_ref.u_mass_sum is not None and mass_ref.u_sum is not None:
            row["head_input_grad_norm_epoch"] = _as_float(acc.u_sum.norm())
            row["head_input_grad_share_epoch"] = _as_float(
                _safe_ratio(acc.u_sum.norm(), mass_ref.u_mass_sum)
            )
            row["cos_head_input_to_task_total_epoch"] = _as_float(
                _safe_cos(acc.u_sum, mass_ref.u_sum)
            )
        else:
            row["head_input_grad_norm_epoch"] = float("nan")
            row["head_input_grad_share_epoch"] = float("nan")
            row["cos_head_input_to_task_total_epoch"] = float("nan")

        return row

    def epoch_rows(self) -> list[dict]:
        rows = []
        for task in self.specs_by_task:
            mass_ref = self._all_acc[task]
            for spec in self.specs_by_task[task]:
                rows.append(self._row(self._acc[spec.key], mass_ref, task))
            rows.append(self._row(mass_ref, mass_ref, task))

        order = {task: index for index, task in enumerate(self.tasks.all)}
        rows.sort(key=lambda row: (order.get(row["task"], len(order)), row["class_id"]))
        return rows

    def specs_payload(
        self,
        *,
        config: dict,
        class_counts: dict[str, np.ndarray] | None = None,
    ) -> dict:
        return specs_payload(self.specs, config=config, class_counts=class_counts)


def specs_payload(
    specs: list[MinoritySpec],
    *,
    config: dict,
    class_counts: dict[str, np.ndarray] | None = None,
) -> dict:
    """Record the minority-class definition / actual class frequencies / metric meaning of this run as JSON."""
    frequencies = {}
    if class_counts is not None:
        for task, counts in class_counts.items():
            counts = np.asarray(counts, dtype=np.float64)
            total = counts.sum()
            if total <= 0:
                continue
            frequencies[task] = [
                {"class_id": int(i), "count": int(c), "pct": float(100.0 * c / total)}
                for i, c in enumerate(counts)
            ]

    return {
        "config": config,
        "specs": [spec.to_dict() for spec in specs],
        "actual_train_class_frequency": frequencies,
        "metric_notes": {
            "net_grad_norm": "||G_c||, G_c = 1/B * sum_{i in c} dL_i/dz_i",
            "grad_share": "||G_c|| / sum_{c'} ||G_c'||  (c' runs over every class of the task, majority classes included)",
            "cos_grad_to_task_total": "cos(G_c, G_all), G_all = 1/B * sum_i dL_i/dz_i",
            "head_grad_norm": "||A_c||_F, A_c = 1/B * sum_{i in c} g_i (x) h_i, "
            "h is the input of base_heads[task][-1]; sum_c A_c = dL_task/dW holds exactly",
            "head_input_grad_norm": "||U_c||, U_c = G_c @ W (gradient the minority class propagates back to the shared representation)",
            "class_id=-1": "reference row for every class of the task (task_all_classes)",
        },
    }


class RunLogger:
    """Append CSV rows incrementally (the header is written once at run start, overwriting
    the old file) so that data survives an interruption."""

    FILES = {
        "train_log": TRAIN_LOG_COLUMNS,
        "train_loss_by_task": LOSS_BY_TASK_COLUMNS,
        "minority_grad_by_epoch": GRAD_COLUMNS,
    }

    def __init__(self, out_dir: Path, enabled: list[str] | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        names = list(self.FILES) if enabled is None else list(enabled)
        self._columns = {name: self.FILES[name] for name in names}
        self._handles: dict[str, object] = {}
        self._writers: dict[str, csv.DictWriter] = {}
        for name, columns in self._columns.items():
            handle = open(self.path(name), "w", newline="", encoding="utf-8")
            writer = csv.writer(handle)
            writer.writerow(columns)
            handle.flush()
            self._handles[name] = handle
            self._writers[name] = writer

    def path(self, name: str) -> Path:
        return self.out_dir / f"{name}.csv"

    @staticmethod
    def _format(value):
        if value is None:
            return ""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            value = float(value)
            return "" if not math.isfinite(value) else value
        return value

    def log(self, name: str, row: dict) -> None:
        writer = self._writers[name]
        columns = self._columns[name]
        writer.writerow([self._format(row.get(column)) for column in columns])
        self._handles[name].flush()

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()
        self._writers.clear()


def sanitize_json(value):
    """Turn NaN/Inf into None so that json.dumps emits valid JSON."""
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    return value


def write_json(path: Path, payload: dict) -> None:
    Path(path).write_text(
        json.dumps(sanitize_json(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )
