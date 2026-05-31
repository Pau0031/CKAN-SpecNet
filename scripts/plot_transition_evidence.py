#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

LAUNCH_DIR = Path.cwd()

import matplotlib.pyplot as plt
import polars as pl

from ckan_specnet.core import Config, EVAL_NAME_COL, configure_torch_runtime, device_of
from ckan_specnet.data import frame_to_xy
from ckan_specnet.eval import load_manifest, tasks_from_manifest
from ckan_specnet.paths import PlotPaths
from ckan_specnet.plot import (
    configure_matplotlib,
    get_kan_contribution_sample,
    load_plot_ensemble,
    plot_transition_evidence,
    predict_plot_ensemble,
    select_correct_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate transition-evidence plots.")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pdf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = PlotPaths.from_args(args, LAUNCH_DIR)

    configure_torch_runtime()
    configure_matplotlib()

    manifest = load_manifest(paths.run_dir)
    tasks = tasks_from_manifest(manifest)

    if args.task not in tasks.all:
        raise ValueError(f"Unknown task: {args.task}")

    test_df = pl.read_parquet(paths.test)
    subset = test_df.filter(pl.col(EVAL_NAME_COL) == args.eval_name)

    if subset.height == 0:
        raise ValueError(f"Empty eval subset: {args.eval_name}")

    x, y, _ = frame_to_xy(subset, tasks)

    cfg = replace(
        Config(),
        test_parquet=paths.test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = device_of()
    loaded = load_plot_ensemble(run_dir=paths.run_dir, tasks=tasks)

    pred_result = predict_plot_ensemble(
        models=loaded["models"],
        preprocessors=loaded["preprocessors"],
        x=x,
        tasks=tasks,
        cfg=cfg,
        device=device,
    )

    if args.sample_index is not None:
        sample_idx = int(args.sample_index)

        if not 0 <= sample_idx < len(x):
            raise IndexError(
                f"sample-index={sample_idx} out of range for "
                f"{args.eval_name} with {len(x)} samples."
            )

        final_class = int(pred_result["y_pred"][args.task][sample_idx])
    else:
        if args.class_id is None:
            raise ValueError("Either --sample-index or --class-id is required.")

        final_class = int(args.class_id)
        sample_idx = select_correct_sample(
            y_true=y,
            pred_result=pred_result,
            task=args.task,
            cls=final_class,
            rank=args.rank,
        )

        if sample_idx is None:
            raise RuntimeError(
                f"No correctly predicted sample found for "
                f"task={args.task}, class={final_class}."
            )

    if final_class <= 0:
        raise ValueError(
            f"Selected class is {final_class}. Transition evidence requires class > 0."
        )

    sample = get_kan_contribution_sample(
        models=loaded["models"],
        preprocessors=loaded["preprocessors"],
        x=x,
        y=y,
        pred_result=pred_result,
        task=args.task,
        sample_idx=sample_idx,
        device=device,
    )

    paths.out.mkdir(parents=True, exist_ok=True)
    max_class = min(final_class, tasks.num_classes[args.task] - 1)

    for target_class in range(1, max_class + 1):
        reference_class = target_class - 1
        save_path = (
            paths.out
            / f"{args.task}_sample{sample_idx}_true{sample['true']}_pred{sample['pred']}_C{reference_class}_to_C{target_class}.png"
        )

        fig = plot_transition_evidence(
            sample=sample,
            target_class=target_class,
            reference_class=reference_class,
            smooth_window=args.smooth_window,
            save_path=save_path,
            save_pdf=not args.no_pdf,
        )

        plt.close(fig)
        print(f"Saved: {save_path}")

    print(f"Done: {paths.out}")


if __name__ == "__main__":
    main()
