"""Self-check script: verify that the formulas used by minority-class gradient
tracking agree exactly with autograd.

Usage (from the repository root):
    python scripts/check_grad_tracking.py

What is verified:
    [1] Poly1Loss(ce_weight=0) / Poly1+CE / PlainCELoss loss values match the
        hand-written formula bit for bit;
    [2] per-sample logits gradient dl_i/dz_i = (scale_ce + scale_poly) * (p_i - onehot)
        agrees with autograd;
    [3] parameter-space attribution sum_c A_c = dL_task/dW
        (W = base_heads[task][-1].weight, with tracked + untracked covering all
        classes) agrees with autograd;
    [4] gradients propagated back to the shared representation
        sum_c U_c = sum_i dL/dh_i agree with autograd;
    [5] minority-class selection result and the column set of the output rows.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ckan_specnet.core import TASKS, ModelConfig, unwrap_model
from ckan_specnet.eval import PlainCELoss, Poly1Loss, classification_loss_terms
from ckan_specnet.grad_track import GRAD_COLUMNS, GradientTracker, minority_specs
from ckan_specnet.model import build_model

torch.manual_seed(0)

DEVICE = torch.device("cpu")
B, L = 24, 64

cfg = replace(
    ModelConfig(),
    conv_channels=(4, 8),
    conv_kernels=(5, 3),
    pool_sizes=(2, None),
    eca_positions=(1,),
    adaptive_pool_size=8,
    fc_hidden=32,
    head_hidden=16,
    contrib_num_basis=8,
    contrib_hidden=4,
)

# ---------------- 1. loss values match the previous implementation ----------------
logits = torch.randn(B, 3)
target = torch.randint(0, 3, (B,))
weight = torch.tensor([0.5, 2.0, 7.0])

loss_new = Poly1Loss(n_classes=3, epsilon=1.0, weight=weight)(logits, target)
ce = torch.nn.functional.cross_entropy(logits, target, weight=weight, reduction="none")
pt = torch.softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
loss_old = (ce + 1.0 * (1 - pt)).mean()
print(f"[1] Poly1(ce_weight=0) vs previous implementation: {loss_new.item():.10f} vs {loss_old.item():.10f} "
      f"diff={abs(loss_new.item()-loss_old.item()):.3e}")
assert abs(loss_new.item() - loss_old.item()) < 1e-9

# ce_weight>0: l = w*CE + eps*(1-p) + lam*CE
loss_pc = Poly1Loss(n_classes=3, epsilon=1.0, weight=weight, ce_weight=0.25)(logits, target)
ref = (ce + 1.0 * (1 - pt) + 0.25 * torch.nn.functional.cross_entropy(logits, target, reduction="none")).mean()
print(f"[1b] Poly1+CE: {loss_pc.item():.10f} vs {ref.item():.10f}")
assert abs(loss_pc.item() - ref.item()) < 1e-9

ce_only = PlainCELoss(n_classes=3, weight=weight)(logits, target)
print(f"[1c] PlainCELoss(weighted CE): {ce_only.item():.10f} vs {ce.mean().item():.10f}")
assert abs(ce_only.item() - ce.mean().item()) < 1e-9

# ---------------- 2. per-sample logits gradient formula ----------------
logits2 = torch.randn(B, 3, requires_grad=True)
target2 = torch.randint(0, 3, (B,))
loss2, stats = classification_loss_terms(
    logits2, target2, epsilon=1.0, weight=weight, extra_ce_weight=0.25
)
prob2 = torch.softmax(logits2, dim=1)
nll2 = torch.nn.functional.cross_entropy(logits2, target2, reduction="none")
w2 = weight.gather(0, target2)
pt2 = prob2.gather(1, target2[:, None]).squeeze(1)
per_sample_ref = w2 * nll2 + 1.0 * (1 - pt2) + 0.25 * nll2
analytic = stats["scale"].unsqueeze(1) * (
    prob2.detach() - torch.nn.functional.one_hot(target2, 3).to(torch.float32)
)
auto = torch.autograd.grad(per_sample_ref.sum(), logits2, retain_graph=True)[0]
print(f"[2] dl_i/dz_i analytic vs autograd: max|diff|={ (analytic-auto).abs().max().item():.3e}")
assert (analytic - auto).abs().max().item() < 1e-6

# ---------------- 3. parameter-space attribution A_c / propagated gradient U_c ----------------
model = build_model(L, cfg, TASKS).to(DEVICE)
tracker = GradientTracker(TASKS, minority_specs(TASKS), {}, track_params=True, fold=1)
tracker.track_params = True
tracker.attach(model)

captured = {}


def capture_hook(task):
    def hook(_m, inputs, _o):
        if task in tracker.specs_by_task and inputs and torch.is_tensor(inputs[0]):
            captured[task] = inputs[0]
        return None
    return hook


heads = unwrap_model(model).base_heads
for task in tracker.specs_by_task:
    heads[task][-1].register_forward_hook(capture_hook(task))

losses = {
    task: Poly1Loss(n_classes=TASKS.num_classes[task], epsilon=1.0,
                    weight=torch.ones(TASKS.num_classes[task]) * (1.0 + i % 3))
    for i, task in enumerate(TASKS.all)
}

x = torch.randn(B, L)
y = {task: torch.randint(0, TASKS.num_classes[task], (B,)) for task in TASKS.all}

tracker.begin_epoch(1)
tracker.begin_step()
out = model(x)
total = None
per_task = {}
task_stats = {}
for task in TASKS.all:
    lt, st = losses[task](out[task], y[task], return_stats=True)
    per_task[task] = lt
    task_stats[task] = st
    tracker.observe_task(task, st, y[task])
    total = lt if total is None else total + lt

# keep the gradient of h so that dL/dh can be read back
for task, h in captured.items():
    if h.requires_grad:
        h.retain_grad()

total.backward()
tracker.end_step()

ok = True
checked = 0
for task in tracker.specs_by_task:
    W = heads[task][-1].weight
    acc_all = tracker._all_acc[task]
    a_sum = acc_all.a_sum
    grad_true = W.grad
    diff = (a_sum - grad_true).abs().max().item()
    scale_ref = max(grad_true.abs().max().item(), 1e-8)
    rel = diff / scale_ref

    # sum_tracked A_c + sum_untracked A_c == A_all (per-class split is complete)
    h = captured[task]
    st = task_stats[task]
    prob_t = st["prob"]
    scale_t = st["scale"]
    target_t = y[task]
    onehot = torch.nn.functional.one_hot(target_t, prob_t.shape[1]).to(torch.float32)
    g_t = scale_t.unsqueeze(1) * (prob_t - onehot)
    tracked_classes = {spec.class_id for spec in tracker.specs_by_task[task]}
    spec_sum = None
    for spec in tracker.specs_by_task[task]:
        a_c = tracker._acc[spec.key].a_sum
        if a_c is None:
            continue
        spec_sum = a_c.clone() if spec_sum is None else spec_sum + a_c
    rest = None
    for c in range(prob_t.shape[1]):
        if c in tracked_classes:
            continue
        m = target_t == c
        if int(m.sum()) == 0:
            continue
        a_c = (g_t[m].t() @ h[m]) / prob_t.shape[0]
        rest = a_c if rest is None else rest + a_c
    complete = float("inf")
    if spec_sum is not None:
        whole = spec_sum if rest is None else spec_sum + rest
        complete = (whole - a_sum).abs().max().item()

    # propagated gradient: sum_c U_c = sum_i dL/dh_i = h.grad.sum(0)
    u_sum = acc_all.u_sum
    u_diff = float("nan")
    u_rel = float("nan")
    if h is not None and h.grad is not None and u_sum is not None:
        ref = h.grad.sum(0)
        u_diff = (u_sum - ref).abs().max().item()
        u_rel = u_diff / max(ref.abs().max().item(), 1e-8)

    print(
        f"[3] {task:22s} |A_all-W.grad|max={diff:.3e} (rel={rel:.2e}) "
        f"|tracked+rest-A_all|max={complete:.3e} |U_all-Sum(dL/dh)|={u_diff:.3e} (rel={u_rel:.2e})"
    )
    if rel > 1e-5 or complete > 1e-5:
        ok = False
    if u_rel == u_rel and u_rel > 1e-5:
        ok = False
    checked += 1

print(f"[3] checked tasks={checked} ok={ok}")
assert ok

# ---------------- 4. minority-class selection ----------------
specs = minority_specs(TASKS)
print(f"[4] default minority specs: {len(specs)}")
for s in specs:
    print("   ", s.to_dict())
assert len(specs) == 19

rows = tracker.epoch_rows()
print(f"[5] epoch_rows: {len(rows)}, n_columns={len(rows[0])}")
missing = set(GRAD_COLUMNS) - set(rows[0])
extra = set(rows[0]) - set(GRAD_COLUMNS)
print(f"[5] difference vs GRAD_COLUMNS: missing={missing} extra={extra}")
assert not missing and not extra
print("[5] example row:", {k: rows[0][k] for k in
      ("task", "class_id", "n_samples", "sample_frac", "mean_pt", "mean_scale",
       "grad_share_epoch", "cos_grad_to_task_total_epoch", "head_grad_share_epoch",
       "cos_head_to_task_total_epoch", "head_input_grad_share_epoch")})
print("ALL DONE")
