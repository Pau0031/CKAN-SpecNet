#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_p_check.py
==================

Paired significance testing (p-values) for

    Table 5  Binary functional-group recognition performance
             (21 binary tasks x 1000 spectra)
    Table 6  Functional-group multiplicity prediction performance
             (12 multiclass tasks x 1000 spectra)

comparing **raw spectra** vs **digitized spectra**, i.e. the runs behind
``results/raw_1000_test_9_1`` and ``results/digital_1000_test_9_1``
(CKAN-SpecNet, 5-fold soft-voting ensemble, n = 1000 paired spectra).

Statistical design
------------------
Both conditions are evaluated on the *same* 1,000 spectra with the *same*
5-fold ensemble, so every comparison is **paired**.  Three complementary
paired designs are reported:

1. **Across spectra** (n = 1,000) -- PRIMARY p-value
   Paired randomization (sign-flip) test and paired bootstrap (percentile
   CI).  The spectrum is the unit of resampling and all 33 task decisions of
   one spectrum move together, so the test is cluster-robust against the
   correlation between the task decisions of a spectrum.  This matches the
   sampling unit implied by ``n = 1000`` in the table.

2. **Across tasks** (n = 21 binary / 12 multiclass)
   Paired t-test + Wilcoxon signed-rank + sign test on the per-task metric
   difference.  This matches the estimand of the tables: the reported macro
   precision/recall/F1 are averages *over tasks*.

3. **Across folds** (n = 5)
   Paired t-test on the per-fold macro metric (matches the "+- std"
   convention of the tables), reported both naive and with the
   Nadeau & Bengio (2003) correction for overlapping training sets.

Accuracy additionally gets an exact **McNemar** test on per-spectrum
correctness (pooled and per task), and **TOST** equivalence tests are given
for a "digitization does not degrade performance" claim.

Outputs -> results/anylize/p_check/

Usage
-----
    .venv/bin/python scripts/row_and_digital_comparation/analyze_p_check.py
    .venv/bin/python scripts/row_and_digital_comparation/analyze_p_check.py --n-boot 20000 --n-perm 20000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

METRICS = ["accuracy", "precision", "recall", "f1", "qwk"]
METRIC_LABEL = {
    "accuracy": "Accuracy",
    "precision": "Macro Precision",
    "recall": "Macro Recall",
    "f1": "Macro F1",
    "qwk": "QWK",
}
SUMMARY_KEY = {
    "accuracy": "acc",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "qwk": "qwk",
}


# --------------------------------------------------------------------------
# metric estimators (fast, confusion-matrix based, identical to sklearn)
# --------------------------------------------------------------------------


def onehot(y: np.ndarray, pred: np.ndarray, c: int) -> np.ndarray:
    """(n, C*C) one-hot encoding of the (true, pred) pair of every decision."""
    flat = y.astype(np.int64) * c + pred.astype(np.int64)
    out = np.zeros((y.size, c * c), dtype=np.int32)
    out[np.arange(y.size), flat] = 1
    return out


def qwk_from_counts(counts: np.ndarray, labels: np.ndarray) -> float:
    """Quadratic weighted kappa from a confusion matrix (rows true, cols pred).

    Equivalent to ``sklearn.metrics.cohen_kappa_score(y, pred, weights='quadratic')``.
    """
    sub = counts[np.ix_(labels, labels)].astype(np.float64)
    n = sub.sum()
    if n <= 0 or len(labels) < 2:
        return float("nan")
    o = sub / n
    e = np.outer(sub.sum(1), sub.sum(0)) / (n * n)
    k = len(labels)
    idx = np.arange(k)
    w = (idx[:, None] - idx[None, :]) ** 2 / (k - 1) ** 2
    den = float((w * e).sum())
    if den <= 0:
        return float("nan")
    return float(1.0 - (w * o).sum() / den)


def metrics_from_counts(
    counts: np.ndarray,
    n_classes: int,
    macro_labels: np.ndarray,
    kappa_labels: np.ndarray | None,
) -> dict:
    """Confusion matrix -> the metrics reported in Table 5 / Table 6.

    ``macro_labels``  labels used for macro precision/recall/F1.  Fixed to the
                      labels observed in the FULL test set so that the metric
                      definition does not drift when a resample happens to drop
                      a rare class (identical to the saved ``*_task_metrics.csv``).
    ``kappa_labels``  labels used for QWK (None -> not reported, as for the
                      binary tasks in the paper).
    """
    n = counts.sum()
    acc = float(np.trace(counts) / n * 100.0) if n else float("nan")

    sub = counts[np.ix_(macro_labels, macro_labels)].astype(np.float64)
    tp = np.diag(sub).copy()
    fp = sub.sum(0) - tp
    fn = sub.sum(1) - tp

    def _div(num, den):
        out = np.zeros_like(num)
        np.divide(num, den, out=out, where=den > 0)
        return out

    prec = _div(tp, tp + fp)
    rec = _div(tp, tp + fn)
    f1 = _div(2.0 * tp, 2.0 * tp + fp + fn)

    return {
        "accuracy": acc,
        "precision": float(prec.mean() * 100.0),
        "recall": float(rec.mean() * 100.0),
        "f1": float(f1.mean() * 100.0),
        "qwk": float(qwk_from_counts(counts, kappa_labels)) if kappa_labels is not None else float("nan"),
    }


def counts_of(y: np.ndarray, proba: np.ndarray, c: int) -> np.ndarray:
    pred = proba.argmax(1)
    return np.bincount(y * c + pred, minlength=c * c).reshape(c, c).astype(np.int64)


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------


def eval_subset_key(meta_path: Path) -> str:
    """Evaluation-subset name stored in one cached ensemble file.

    ``analyze_raw_vs_digital.py`` keys every cache entry by the ``_eval_name``
    value of the parquet it read, so the raw cache and the digitized cache may
    carry different subset names.  The name is read back here instead of being
    hardcoded, so the script also works when both parquets share one label.
    """
    payload = json.loads(meta_path.read_text())
    if not payload:
        raise ValueError(f"empty ensemble metadata: {meta_path}")
    return next(iter(payload))


def load_pair(pred_dir: Path):
    """Load cached raw / digitized per-sample x per-task x per-fold probabilities."""
    preds = pred_dir / "predictions"
    raw_meta_path = preds / "raw_ensemble_probabilities.meta.json"
    dig_meta_path = preds / "digital_ensemble_probabilities.meta.json"
    raw_key = eval_subset_key(raw_meta_path)
    dig_key = eval_subset_key(dig_meta_path)

    meta = json.loads(raw_meta_path.read_text())[raw_key]
    tasks: list[str] = meta["tasks"]
    num_classes: dict[str, int] = meta["num_classes"]

    raw = np.load(preds / "raw_ensemble_probabilities.npz", allow_pickle=True)
    dig = np.load(preds / "digital_ensemble_probabilities.npz", allow_pickle=True)

    rid = raw[f"{raw_key}__sample_id"]
    did = dig[f"{dig_key}__sample_id"]
    assert set(rid.tolist()) == set(did.tolist()), "sample ids differ between raw and digitized"
    pos = {s: i for i, s in enumerate(did.tolist())}
    perm = np.array([pos[s] for s in rid.tolist()], dtype=int)  # align digitized -> raw order

    data: dict[str, dict] = {}
    for t in tasks:
        y = raw[f"y__{t}"]
        assert np.array_equal(y, dig[f"y__{t}"][perm]), f"label mismatch for task {t}"
        data[t] = {
            "y": y,
            "raw": {"ens": raw[f"ens__{t}"], "folds": [raw[f"fold{k}__{t}"] for k in range(1, 6)]},
            "dig": {"ens": dig[f"ens__{t}"][perm], "folds": [dig[f"fold{k}__{t}"][perm] for k in range(1, 6)]},
        }
    return tasks, num_classes, data, rid


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------


def paired_tests(delta: np.ndarray) -> dict:
    """Paired t-test + Wilcoxon signed-rank + sign test on paired deltas."""
    d = np.asarray(delta, dtype=np.float64)
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if (n > 1 and sd > 0) else float("nan")

    if np.allclose(d, 0.0):
        t_stat, p_t, p_w = 0.0, 1.0, 1.0
        n_pos = n_neg = 0
        p_sign = 1.0
    else:
        t_stat = float(stats.ttest_rel(d, np.zeros_like(d)).statistic)
        p_t = float(stats.ttest_rel(d, np.zeros_like(d)).pvalue)
        try:
            p_w = float(stats.wilcoxon(d, alternative="two-sided").pvalue)
        except ValueError:
            p_w = float("nan")
        n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
        nz = n_pos + n_neg
        p_sign = float(stats.binomtest(n_pos, nz, 0.5, alternative="two-sided").pvalue) if nz else 1.0

    if n > 1 and sd > 0:
        half = stats.t.ppf(0.975, n - 1) * se
        ci = (mean - half, mean + half)
        dz = mean / sd
    else:
        ci, dz = (float("nan"), float("nan")), float("nan")

    return {
        "n_pairs": int(n),
        "mean_delta": mean,
        "sd_delta": sd,
        "se_delta": float(se),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "cohens_dz": float(dz),
        "t_stat": float(t_stat),
        "p_paired_t": float(p_t),
        "p_wilcoxon": float(p_w),
        "n_better": n_pos,
        "n_worse": n_neg,
        "p_sign_test": float(p_sign),
    }


def nadeau_bengio_ttest(delta: np.ndarray, n_train: int, n_test: int) -> dict:
    """Nadeau & Bengio (2003) corrected resampled t-test for k-fold CV.

    Var(mean delta) is inflated by (1/k + n_test/n_train) to account for the
    overlap between the k training sets.
    """
    d = np.asarray(delta, dtype=np.float64)
    k = d.size
    if k < 2:
        return {"p_corrected": float("nan")}
    mean = float(d.mean())
    s2 = float(d.var(ddof=1))
    if s2 == 0:
        return {"p_corrected": 1.0 if mean == 0 else 0.0,
                "var_inflation": float("nan"), "t_corrected": float("nan")}
    infl = 1.0 / k + n_test / n_train
    t_corr = mean / np.sqrt(infl * s2)
    return {
        "var_inflation": float(infl),
        "t_corrected": float(t_corr),
        "p_corrected": float(2 * stats.t.sf(abs(t_corr), k - 1)),
        "p_naive": float(2 * stats.t.sf(abs(mean / np.sqrt(s2 / k)), k - 1)),
    }


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test on per-decision correctness (a = raw, b = digitized)."""
    ca, cb = pred_a == y, pred_b == y
    b = int((ca & ~cb).sum())   # raw right, digitized wrong
    c = int((~ca & cb).sum())   # raw wrong, digitized right
    n = b + c
    p_exact = 1.0 if n == 0 else float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    chi2 = 0.0 if n == 0 else (abs(b - c) - 1.0) ** 2 / n
    return {
        "n_discordant": n,
        "raw_only_correct": b,
        "dig_only_correct": c,
        "net_change": c - b,
        "p_mcnemar_exact": p_exact,
        "p_mcnemar_chi2_cc": float(stats.chi2.sf(chi2, 1)),
    }


def tost_p(delta: np.ndarray, margin: float) -> tuple[float, tuple[float, float]]:
    """Two one-sided tests for equivalence within +-margin (paired)."""
    d = np.asarray(delta, dtype=np.float64)
    n = d.size
    if n < 2:
        return float("nan"), (float("nan"), float("nan"))
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    if sd == 0:
        return float("nan"), (float("nan"), float("nan"))
    se, df = sd / np.sqrt(n), n - 1
    p_low = float(stats.t.sf((mean + margin) / se, df))   # H0: delta <= -margin
    p_up = float(stats.t.cdf((mean - margin) / se, df))   # H0: delta >= +margin
    half = stats.t.ppf(0.95, df) * se                     # 90% CI = TOST acceptance region
    return max(p_low, p_up), (mean - half, mean + half)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaN-safe)."""
    idx = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    out = [float("nan")] * len(pvals)
    if not idx:
        return out
    m = len(idx)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: pvals[j])):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        out[i] = running
    return out


# --------------------------------------------------------------------------
# analysis blocks
# --------------------------------------------------------------------------


def task_context(tasks, num_classes, data):
    """Per-task fixed label sets, matching ``ckan_specnet.eval.task_metrics`` exactly.

    ``macro_labels`` = ``np.unique(y_true)``  (evaluate.py: ``observed``)
    ``kappa_labels`` = ``range(n_classes)``   (evaluate.py: ``all_labels``), only for >2 classes
    """
    ctx = {}
    for t in tasks:
        c = num_classes[t]
        y = data[t]["y"]
        observed = np.unique(y)
        ctx[t] = {
            "c": c,
            "macro_labels": observed,
            # evaluate.py: qwk is NaN for binary tasks or when <2 classes are observed
            "kappa_labels": np.arange(c) if (c > 2 and observed.size > 1) else None,
        }
    return ctx


def build_task_level(tasks, num_classes, data, ctx):
    """Per-task ensemble metric values and per-task deltas."""
    recs = {}
    for t in tasks:
        c = num_classes[t]
        y = data[t]["y"]
        rec = {"n_classes": c, "n": int(y.size), "observed_labels": ctx[t]["macro_labels"].tolist()}
        for cond in ("raw", "dig"):
            counts = counts_of(y, data[t][cond]["ens"], c)
            rec[cond] = metrics_from_counts(counts, c, ctx[t]["macro_labels"], ctx[t]["kappa_labels"])
        rec["delta"] = {k: rec["dig"][k] - rec["raw"][k] for k in METRICS}
        recs[t] = rec
    return recs


def build_fold_level(tasks, num_classes, data, ctx):
    """Per-fold per-task metric dict."""
    folds = []
    for f in range(5):
        per_task = {}
        for t in tasks:
            c = num_classes[t]
            y = data[t]["y"]
            per_task[t] = {
                cond: metrics_from_counts(counts_of(y, data[t][cond]["folds"][f], c), c,
                                          ctx[t]["macro_labels"], ctx[t]["kappa_labels"])
                for cond in ("raw", "dig")
            }
        folds.append(per_task)
    return folds


def spectrum_tests(tasks, num_classes, data, ctx, group_tasks, n_boot, n_perm, rng):
    """Cluster-robust paired randomization test + paired bootstrap over spectra.

    Resampling / permutation unit = spectrum; all task decisions of a spectrum
    stay together, which preserves the correlation between tasks.
    """
    n = data[group_tasks[0]]["y"].size
    slices, a_list, b_list = {}, [], []
    off = 0
    for t in group_tasks:
        c = ctx[t]["c"]
        y = data[t]["y"]
        a = onehot(y, data[t]["raw"]["ens"].argmax(1), c)
        b = onehot(y, data[t]["dig"]["ens"].argmax(1), c)
        slices[t] = (off, off + c * c, c)
        a_list.append(a)
        b_list.append(b)
        off += c * c
    A = np.concatenate(a_list, axis=1)
    B = np.concatenate(b_list, axis=1)
    A_sum, B_sum = A.sum(0), B.sum(0)
    D = B - A  # int32 diff; counts under a flip are A_sum + D[mask].sum(0)

    def group_delta(craw, cdig):
        acc = {k: [] for k in METRICS}
        for t in group_tasks:
            s, e, c = slices[t]
            mr = metrics_from_counts(craw[s:e].reshape(c, c), c, ctx[t]["macro_labels"], ctx[t]["kappa_labels"])
            md = metrics_from_counts(cdig[s:e].reshape(c, c), c, ctx[t]["macro_labels"], ctx[t]["kappa_labels"])
            for k in METRICS:
                if np.isfinite(mr[k]) and np.isfinite(md[k]):
                    acc[k].append(md[k] - mr[k])
        return {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}

    obs = group_delta(A_sum.astype(np.int64), B_sum.astype(np.int64))  # identical to observed ensemble delta

    boot = {k: np.empty(n_boot) for k in METRICS}
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d = group_delta(A[idx].sum(0).astype(np.int64), B[idx].sum(0).astype(np.int64))
        for k in METRICS:
            boot[k][i] = d[k]

    perm = {k: np.empty(n_perm) for k in METRICS}
    for i in range(n_perm):
        mask = rng.random(n) < 0.5
        dsum = D[mask].sum(0).astype(np.int64)
        d = group_delta(A_sum + dsum, B_sum - dsum)
        for k in METRICS:
            perm[k][i] = d[k]

    out = {}
    for k in METRICS:
        if not np.isfinite(obs[k]):
            out[k] = {}
            continue
        bd, pd_ = boot[k], perm[k]
        out[k] = {
            "delta_obs": obs[k],
            "boot_mean": float(bd.mean()),
            "boot_se": float(bd.std(ddof=1)),
            "boot_ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
            "boot_ci90": [float(np.percentile(bd, 5.0)), float(np.percentile(bd, 95.0))],
            "p_bootstrap": float(min(1.0, 2.0 * min((bd <= 0).mean(), (bd >= 0).mean()))),
            "p_permutation": float(min(1.0, (1.0 + (np.abs(pd_) >= abs(obs[k]) - 1e-12).sum()) / (n_perm + 1.0))),
            "perm_null_sd": float(pd_.std(ddof=1)),
            "perm_null_abs_p95": float(np.percentile(np.abs(pd_), 95)),
        }
    return out, boot, perm


def verify_against_saved(task_recs, results_dir: Path) -> dict:
    """Check recomputed per-task metrics against the published *_task_metrics.csv."""
    import pandas as pd

    report, worst = {}, 0.0
    for cond, folder, legacy_name in [
        ("raw", "raw_1000_test_9_1", "raw_selected_spectra_task_metrics.csv"),
        ("dig", "digital_1000_test_9_1", "digital_selected_spectra_task_metrics.csv"),
    ]:
        run_dir = results_dir / folder
        found = sorted(run_dir.glob("*_task_metrics.csv"))
        path = found[0] if found else run_dir / legacy_name
        if not path.exists():
            report[cond] = {"status": "missing", "path": str(path)}
            continue
        df = pd.read_csv(path).set_index("task")
        diffs = []
        for t, rec in task_recs.items():
            if t not in df.index:
                continue
            for col in ("accuracy", "precision", "recall", "f1", "qwk"):
                a, b = rec[cond][col], float(df.loc[t, col])
                if np.isfinite(a) and np.isfinite(b):
                    diffs.append(abs(a - b))
        m = max(diffs) if diffs else float("nan")
        worst = max(worst, m)
        report[cond] = {"status": "ok", "n_values_compared": len(diffs), "max_abs_diff": float(m)}
    report["max_abs_diff_overall"] = float(worst)
    report["identical"] = bool(worst < 1e-9)
    return report


def verify_std_convention(fold_level, tasks, num_classes, results_dir: Path) -> dict:
    """Check that the table's '+-' is the population std (ddof=0) across the 5 folds."""
    import pandas as pd

    out = {}
    for cond, fname, pref in [
        ("raw", "raw_1000_test_9_1/summary.csv", "raw"),
        ("dig", "digital_1000_test_9_1/summary.csv", "digital"),
    ]:
        path = results_dir / fname
        if not path.exists():
            continue
        sv = pd.read_csv(path).iloc[0]
        for gname, gt in [("binary", [t for t in tasks if num_classes[t] == 2]),
                          ("multiclass", [t for t in tasks if num_classes[t] > 2])]:
            for k in METRICS:
                if gname == "binary" and k == "qwk":
                    continue
                vals = []
                for f in range(5):
                    vals.append(np.mean([fold_level[f][t][cond][k] for t in gt
                                         if np.isfinite(fold_level[f][t][cond][k])]))
                key = f"{gname}_{SUMMARY_KEY[k]}_std"
                if key not in sv:
                    continue
                saved = float(sv[key])
                out[f"{pref}/{gname}/{k}"] = {
                    "saved_std": saved,
                    "std_ddof0_across_folds": float(np.std(vals, ddof=0)),
                    "abs_diff": float(abs(np.std(vals, ddof=0) - saved)),
                }
    out["max_abs_diff"] = float(max(v["abs_diff"] for k, v in out.items() if isinstance(v, dict)))
    return out


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return "<0.001"
    return f"{p:.3f}"


def pm(value: float, std: float, nd: int = 2) -> str:
    return f"{value:.{nd}f} \u00b1 {std:.{nd}f}"


# --------------------------------------------------------------------------
# outputs: tables, figures, report
# --------------------------------------------------------------------------


def write_outputs(args, results, task_recs, fold_level, nulls):
    import pandas as pd

    tables_dir = args.out_dir / "tables"
    figures_dir = args.out_dir / "figures"
    R = args.results_dir

    raw_sum = pd.read_csv(R / "raw_1000_test_9_1" / "summary.csv").iloc[0]
    dig_sum = pd.read_csv(R / "digital_1000_test_9_1" / "summary.csv").iloc[0]

    # -------- Table 5 / Table 6 (with p) --------------------------------
    specs = {
        "table5_binary": {
            "title": "Table 5. Binary functional-group recognition performance using raw and "
                     "digitized spectra on the selected evaluation datasets.",
            "group": "binary",
            "cols": [("Accuracy (%)", "acc"), ("Macro Precision (%)", "precision"),
                     ("Macro Recall (%)", "recall"), ("Macro F1 (%)", "f1")],
            "prefix": "binary",
        },
        "table6_multiclass": {
            "title": "Table 6. Functional-group multiplicity prediction performance using raw and "
                     "digitized spectra on the selected evaluation datasets.",
            "group": "multiclass",
            "cols": [("Accuracy (%)", "acc"), ("Macro Precision (%)", "precision"),
                     ("Macro Recall (%)", "recall"), ("Macro F1 (%)", "f1"), ("QWK", "qwk")],
            "prefix": "multiclass",
        },
    }

    table_rows_csv = []
    for key, spec in specs.items():
        g = results["groups"][spec["group"]]
        p = spec["prefix"]
        nd = 2

        def cell(sumrow, met):
            v = float(sumrow[f"{p}_{met}"])
            s = float(sumrow[f"{p}_{met}_std"])
            return pm(v, s, 3 if met == "qwk" else 2)

        header = ["Evaluation dataset", "n"] + [c[0] for c in spec["cols"]]
        rows = [
            ["raw spectra", "1000"] + [cell(raw_sum, m) for _, m in spec["cols"]],
            ["digitized spectra", "1000"] + [cell(dig_sum, m) for _, m in spec["cols"]],
        ]
        pvals = []
        for _, m in spec["cols"]:
            key_m = {"acc": "accuracy", "precision": "precision", "recall": "recall",
                     "f1": "f1", "qwk": "qwk"}[m]
            pvals.append(g["by_spectrum"][key_m]["p_permutation"])
        rows.append(["p value", "\u2014"] + [fmt_p(x) for x in pvals])

        # markdown A: dedicated p row
        md = [f"**{spec['title']}**", "",
              "| " + " | ".join(header) + " |",
              "|" + "---|" * len(header)]
        for r in rows:
            md.append("| " + " | ".join(r) + " |")
        md += ["", "p values: two-sided paired randomization (sign-flip) test over the 1,000 "
                   "spectra; each metric tested separately (Holm-adjusted p = "
               + ", ".join(fmt_p(g["holm_within_table"].get(
                   {"acc": "accuracy", "precision": "precision", "recall": "recall",
                    "f1": "f1", "qwk": "qwk"}[m], float("nan"))) for _, m in spec["cols"])
               + ")."]
        (tables_dir / f"{key}_with_p.md").write_text("\n".join(md) + "\n")

        # markdown B: inline p in each digitized cell
        md2 = [f"**{spec['title']}** (p in parentheses, digitized row)", "",
               "| " + " | ".join(header) + " |", "|" + "---|" * len(header),
               "| " + " | ".join(rows[0]) + " |"]
        dig_cells = []
        for (_, m), pv in zip(spec["cols"], pvals):
            dig_cells.append(f"{cell(dig_sum, m)} (p = {fmt_p(pv)})")
        md2.append("| digitized spectra | 1000 | " + " | ".join(dig_cells) + " |")
        (tables_dir / f"{key}_with_p_inline.md").write_text("\n".join(md2) + "\n")

        for (cname, m), pv in zip(spec["cols"], pvals):
            km = {"acc": "accuracy", "precision": "precision", "recall": "recall",
                  "f1": "f1", "qwk": "qwk"}[m]
            table_rows_csv.append({
                "table": spec["title"].split(".")[0],
                "group": spec["group"],
                "metric": cname,
                "raw_mean": float(raw_sum[f"{p}_{m}"]),
                "raw_std": float(raw_sum[f"{p}_{m}_std"]),
                "digitized_mean": float(dig_sum[f"{p}_{m}"]),
                "digitized_std": float(dig_sum[f"{p}_{m}_std"]),
                "delta_digitized_minus_raw": g["by_spectrum"][km]["delta_obs"],
                "p_value": pv,
                "p_holm": g["holm_within_table"].get(km, float("nan")),
            })
    pd.DataFrame(table_rows_csv).to_csv(tables_dir / "tables5_6_with_p.csv", index=False)

    # -------- full test grid --------------------------------------------
    full = []
    for gname in ("binary", "multiclass"):
        g = results["groups"][gname]
        for k in METRICS:
            if not g["by_spectrum"].get(k):
                continue
            s, t_, f_, nb = (g["by_spectrum"][k], g["by_task"].get(k, {}),
                             g["by_fold"].get(k, {}), g["by_fold_nadeau_bengio"].get(k, {}))
            full.append({
                "group": gname,
                "metric": METRIC_LABEL[k],
                "delta_pp": s["delta_obs"],
                "p_randomization_spectra": s["p_permutation"],
                "p_bootstrap_spectra": s["p_bootstrap"],
                "boot_ci95_low": s["boot_ci95"][0],
                "boot_ci95_high": s["boot_ci95"][1],
                "p_paired_t_tasks": t_.get("p_paired_t", float("nan")),
                "p_wilcoxon_tasks": t_.get("p_wilcoxon", float("nan")),
                "p_sign_test_tasks": t_.get("p_sign_test", float("nan")),
                "n_tasks_digitized_better": t_.get("n_better", ""),
                "n_tasks_digitized_worse": t_.get("n_worse", ""),
                "p_paired_t_folds": f_.get("p_paired_t", float("nan")),
                "p_fold_nadeau_bengio": nb.get("p_corrected", float("nan")),
                "p_mcnemar_accuracy": (g["mcnemar_pooled"]["p_mcnemar_exact"]
                                       if k == "accuracy" else float("nan")),
                "p_holm_within_table": g["holm_within_table"].get(k, float("nan")),
            })
    df_full = pd.DataFrame(full)
    df_full.to_csv(tables_dir / "tableS_paired_tests_full.csv", index=False)

    md = ["**Table S-p. Paired significance tests, raw vs digitized spectra (all designs).**", "",
          "| Group | Metric | \u0394 (pp) | p randomization (spectra) | p bootstrap (spectra) | "
          "bootstrap 95% CI | p paired t (tasks) | p Wilcoxon (tasks) | p Holm |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df_full.iterrows():
        md.append(f"| {r['group']} | {r['metric']} | {r['delta_pp']:+.3f} | "
                  f"{fmt_p(r['p_randomization_spectra'])} | {fmt_p(r['p_bootstrap_spectra'])} | "
                  f"[{r['boot_ci95_low']:+.3f}, {r['boot_ci95_high']:+.3f}] | "
                  f"{fmt_p(r['p_paired_t_tasks'])} | {fmt_p(r['p_wilcoxon_tasks'])} | "
                  f"{fmt_p(r['p_holm_within_table'])} |")
    (tables_dir / "tableS_paired_tests_full.md").write_text("\n".join(md) + "\n")

    # -------- per-task / per-fold / McNemar / TOST -----------------------
    rows = []
    for gname in ("binary", "multiclass"):
        g = results["groups"][gname]
        for t in g["tasks"]:
            row = {"group": gname, "task": t, "n_classes": task_recs[t]["n_classes"]}
            for k in METRICS:
                row[f"raw_{k}"] = task_recs[t]["raw"][k]
                row[f"dig_{k}"] = task_recs[t]["dig"][k]
                row[f"delta_{k}"] = task_recs[t]["delta"][k]
            mc = g["mcnemar_per_task"][t]
            row.update({"mcnemar_raw_only_correct": mc["raw_only_correct"],
                        "mcnemar_dig_only_correct": mc["dig_only_correct"],
                        "p_mcnemar": mc["p_mcnemar_exact"]})
            rows.append(row)
    pd.DataFrame(rows).to_csv(tables_dir / "tableS_per_task_deltas.csv", index=False)

    rows = []
    for gname in ("binary", "multiclass"):
        g = results["groups"][gname]
        for k, vals in g["fold_deltas"].items():
            for i, v in enumerate(vals, start=1):
                rows.append({"group": gname, "metric": METRIC_LABEL[k], "fold": i, "delta_pp": v})
    pd.DataFrame(rows).to_csv(tables_dir / "tableS_per_fold_deltas.csv", index=False)

    rows = []
    for gname in ("binary", "multiclass"):
        g = results["groups"][gname]
        for k in METRICS:
            if k not in g["tost"]:
                continue
            for mk, mv in g["tost"][k].items():
                rows.append({"group": gname, "metric": METRIC_LABEL[k], "margin": mk,
                             "p_tost": mv["p_tost"], "ci90_low": mv["ci90_delta"][0],
                             "ci90_high": mv["ci90_delta"][1], "equivalent": mv["equivalent"]})
    pd.DataFrame(rows).to_csv(tables_dir / "tableS_tost_equivalence.csv", index=False)

    # -------- figures ----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
        for j, (gname, k) in enumerate([("binary", "accuracy"), ("binary", "f1"),
                                        ("multiclass", "accuracy"), ("multiclass", "f1")]):
            ax = axes[j // 2][j % 2]
            g = results["groups"][gname]
            tasks_g = g["tasks"]
            d = np.array([task_recs[t]["delta"][k] for t in tasks_g]) * 1.0
            d = d[np.isfinite(d)]
            order = np.argsort(d)
            ax.axvline(0, color="k", lw=0.8)
            ax.plot(d[order], np.arange(d.size), "o", ms=4, color="#1f77b4")
            ax.set_yticks(np.arange(d.size))
            ax.set_yticklabels([tasks_g[i] for i in order], fontsize=5)
            ax.set_title(f"{gname} \u2013 \u0394{METRIC_LABEL[k]} (pp)\n"
                         f"mean {d.mean():+.3f} pp, p_perm = "
                         f"{fmt_p(g['by_spectrum'][k]['p_permutation'])}", fontsize=9)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(alpha=0.25, axis="x")
        fig.tight_layout()
        fig.savefig(figures_dir / "figP1_per_task_deltas.png", dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for j, (gname, k) in enumerate([("binary", "accuracy"), ("binary", "f1"),
                                        ("multiclass", "accuracy"), ("multiclass", "qwk")]):
            ax = axes[j // 2][j % 2]
            g = results["groups"][gname]
            pm_ = nulls[gname]["perm"][k]
            pm_ = pm_[np.isfinite(pm_)]
            ax.hist(pm_, bins=60, color="#c8c8c8", edgecolor="none")
            obs = g["by_spectrum"][k]["delta_obs"]
            ax.axvline(obs, color="crimson", lw=1.6,
                       label=f"observed \u0394 = {obs:+.3f} pp")
            ax.axvline(-obs, color="crimson", lw=1.0, ls="--", alpha=0.6)
            ax.set_title(f"{gname} \u2013 {METRIC_LABEL[k]}   "
                         f"p = {fmt_p(g['by_spectrum'][k]['p_permutation'])}", fontsize=9)
            ax.set_xlabel("null distribution of \u0394 (pp), sign-flip permutations", fontsize=8)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures_dir / "figP2_null_distributions.png", dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        for j, gname in enumerate(["binary", "multiclass"]):
            ax = axes[j]
            g = results["groups"][gname]
            ks = [k for k in METRICS if g["by_fold"].get(k)]
            w = 0.8 / max(1, len(ks))
            for i, k in enumerate(ks):
                v = np.array(g["fold_deltas"][k])
                ax.bar(np.arange(1, 6) + i * w - 0.4, v, width=w, label=METRIC_LABEL[k])
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(range(1, 6))
            ax.set_xlabel("fold")
            ax.set_ylabel("\u0394 (dig \u2212 raw, pp)")
            ax.set_title(f"{gname}: per-fold \u0394 (fold-level models)", fontsize=9)
            ax.legend(fontsize=6)
            ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(figures_dir / "figP3_per_fold_deltas.png", dpi=200)
        plt.close(fig)
        print(f"Wrote 3 figures to {figures_dir}")
    except Exception as exc:  # pragma: no cover
        print(f"[warn] figures skipped: {exc}")

    print(f"Wrote tables to {tables_dir}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # this file lives in scripts/row_and_digital_comparation/, so the repository root
    # is two levels up (parents[1] would be scripts/)
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--pred-dir", type=Path, default=root / "results" / "anylize" / "raw_and_digital")
    ap.add_argument("--results-dir", type=Path, default=root / "results")
    ap.add_argument("--out-dir", type=Path, default=root / "results" / "anylize" / "p_check")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-train", type=int, default=28000,
                    help="training-set size, only used for the Nadeau-Bengio correction")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tables_dir = args.out_dir / "tables"
    figures_dir = args.out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 78)
    print("Paired significance testing: raw vs digitized spectra")
    print("=" * 78)

    tasks, num_classes, data, sample_ids = load_pair(args.pred_dir)
    ctx = task_context(tasks, num_classes, data)
    binary_tasks = [t for t in tasks if num_classes[t] == 2]
    multi_tasks = [t for t in tasks if num_classes[t] > 2]
    n = data[tasks[0]]["y"].size
    print(f"tasks          : {len(tasks)} (binary {len(binary_tasks)}, multiclass {len(multi_tasks)})")
    print(f"paired spectra : {n}")

    task_recs = build_task_level(tasks, num_classes, data, ctx)
    fold_level = build_fold_level(tasks, num_classes, data, ctx)

    verif = verify_against_saved(task_recs, args.results_dir)
    std_verif = verify_std_convention(fold_level, tasks, num_classes, args.results_dir)
    print("\n[verification] recomputed metrics vs published *_task_metrics.csv")
    for cond in ("raw", "dig"):
        v = verif[cond]
        print(f"   {cond:4s}: {v.get('n_values_compared', 0)} values, max |diff| = {v.get('max_abs_diff', float('nan')):.2e}")
    print(f"   -> point estimates identical to published tables: {verif['identical']}")
    print(f"   -> '+-' convention (ddof=0 across folds): max |diff| = {std_verif['max_abs_diff']:.2e}")

    results = {
        "meta": {
            "n_spectra": int(n),
            "n_binary_tasks": len(binary_tasks),
            "n_multiclass_tasks": len(multi_tasks),
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "seed": args.seed,
            "n_train_assumed": args.n_train,
            "verification_vs_saved_point_estimates": verif,
            "verification_std_convention": std_verif,
            "primary_test": "paired randomization (sign-flip) test over the 1000 spectra",
        },
        "groups": {},
    }

    nulls = {}
    for gname, gtasks in [("binary", binary_tasks), ("multiclass", multi_tasks)]:
        print("\n" + "-" * 78)
        print(f"GROUP: {gname} ({len(gtasks)} tasks)  -- running {args.n_perm} permutations / {args.n_boot} bootstraps")
        print("-" * 78)

        by_task = {}
        for k in METRICS:
            d = np.array([task_recs[t]["delta"][k] for t in gtasks], dtype=float)
            d = d[np.isfinite(d)]
            if d.size:
                by_task[k] = paired_tests(d)

        by_spec, boot_d, perm_d = spectrum_tests(
            tasks, num_classes, data, ctx, gtasks, args.n_boot, args.n_perm, rng)
        nulls[gname] = {"boot": boot_d, "perm": perm_d}

        by_fold, by_fold_nb = {}, {}
        for k in METRICS:
            vals = []
            for f in range(5):
                a = np.mean([fold_level[f][t]["raw"][k] for t in gtasks if np.isfinite(fold_level[f][t]["raw"][k])])
                b = np.mean([fold_level[f][t]["dig"][k] for t in gtasks if np.isfinite(fold_level[f][t]["dig"][k])])
                vals.append(b - a)
            vals = np.array(vals, dtype=float)
            if np.isfinite(vals).all():
                by_fold[k] = paired_tests(vals)
                by_fold_nb[k] = nadeau_bengio_ttest(vals, args.n_train, n)

        pooled_b = pooled_c = 0
        per_task_mc = {}
        for t in gtasks:
            mc = mcnemar(data[t]["y"], data[t]["raw"]["ens"].argmax(1), data[t]["dig"]["ens"].argmax(1))
            per_task_mc[t] = mc
            pooled_b += mc["raw_only_correct"]
            pooled_c += mc["dig_only_correct"]
        ntot = pooled_b + pooled_c
        p_pool = 1.0 if ntot == 0 else float(stats.binomtest(min(pooled_b, pooled_c), ntot, 0.5,
                                                              alternative="two-sided").pvalue)
        mc_pooled = {
            "n_decisions": int(len(gtasks) * n),
            "raw_only_correct": pooled_b,
            "dig_only_correct": pooled_c,
            "n_discordant": ntot,
            "net_change": pooled_c - pooled_b,
            "delta_acc_pp": (pooled_c - pooled_b) / (len(gtasks) * n) * 100.0,
            "p_mcnemar_exact": p_pool,
        }

        tost = {}
        for k in METRICS:
            d = np.array([task_recs[t]["delta"][k] for t in gtasks], dtype=float)
            d = d[np.isfinite(d)]
            if d.size < 2:
                continue
            tost[k] = {f"margin_{m:g}pp": {
                "p_tost": tost_p(d, m)[0],
                "ci90_delta": list(tost_p(d, m)[1]),
                "equivalent": bool(tost_p(d, m)[0] < 0.05),
            } for m in (0.5, 1.0, 2.0)}

        fam = [k for k in METRICS if by_spec.get(k)]
        holm_map = dict(zip(fam, holm([by_spec[k]["p_permutation"] for k in fam])))

        results["groups"][gname] = {
            "n_tasks": len(gtasks),
            "tasks": gtasks,
            "by_task": by_task,
            "by_spectrum": by_spec,
            "by_fold": by_fold,
            "by_fold_nadeau_bengio": by_fold_nb,
            "mcnemar_pooled": mc_pooled,
            "mcnemar_per_task": per_task_mc,
            "tost": tost,
            "holm_within_table": holm_map,
            "task_deltas": {t: task_recs[t]["delta"] for t in gtasks},
            "task_values": {t: {"raw": task_recs[t]["raw"], "dig": task_recs[t]["dig"]} for t in gtasks},
            "fold_deltas": {k: [
                float(np.mean([fold_level[f][t]["dig"][k] for t in gtasks if np.isfinite(fold_level[f][t]["dig"][k])])
                      - np.mean([fold_level[f][t]["raw"][k] for t in gtasks if np.isfinite(fold_level[f][t]["raw"][k])]))
                for f in range(5)] for k in METRICS if by_fold.get(k)},
        }

    # ---------------- console summary ----------------
    for gname in ("binary", "multiclass"):
        g = results["groups"][gname]
        print(f"\n### {gname.upper()} (n={g['n_tasks']} tasks)")
        print(f"{'metric':16s} {'raw':>9s} {'dig':>9s} {'delta':>8s} {'p_perm':>8s} {'p_boot':>8s} "
              f"{'p_task':>8s} {'p_fold':>8s} {'p_holm':>8s}")
        for k in [m for m in METRICS if g["by_spectrum"].get(m)]:
            rv = np.mean([g["task_values"][t]["raw"][k] for t in g["tasks"] if np.isfinite(g["task_values"][t]["raw"][k])])
            dv = np.mean([g["task_values"][t]["dig"][k] for t in g["tasks"] if np.isfinite(g["task_values"][t]["dig"][k])])
            s = g["by_spectrum"][k]
            print(f"{METRIC_LABEL[k]:16s} {rv:9.4f} {dv:9.4f} {s['delta_obs']:+8.4f} "
                  f"{s['p_permutation']:8.4f} {s['p_bootstrap']:8.4f} "
                  f"{g['by_task'].get(k, {}).get('p_paired_t', float('nan')):8.4f} "
                  f"{g['by_fold'].get(k, {}).get('p_paired_t', float('nan')):8.4f} "
                  f"{g['holm_within_table'].get(k, float('nan')):8.4f}")
        mc = g["mcnemar_pooled"]
        print(f"  McNemar accuracy (pooled {mc['n_decisions']} decisions): raw-only {mc['raw_only_correct']} "
              f"vs dig-only {mc['dig_only_correct']} -> p = {mc['p_mcnemar_exact']:.4f}")

    # ---------------- persist ----------------
    (args.out_dir / "p_check_summary.json").write_text(json.dumps(results, indent=2, default=float))
    np.savez_compressed(
        args.out_dir / "null_distributions.npz",
        **{f"{g}__boot__{k}": nulls[g]["boot"][k] for g in nulls for k in METRICS},
        **{f"{g}__perm__{k}": nulls[g]["perm"][k] for g in nulls for k in METRICS},
    )
    print(f"\nWrote {args.out_dir/'p_check_summary.json'}")

    write_outputs(args, results, task_recs, fold_level, nulls)

    return results, nulls


if __name__ == "__main__":
    main()
