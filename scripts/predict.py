#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

LAUNCH_DIR = Path.cwd()

import numpy as np
import polars as pl

from ckan_specnet.core import Config, EVAL_NAME_COL, configure_torch_runtime, device_of
from ckan_specnet.data import frame_to_xy
from ckan_specnet.eval import (
    load_manifest,
    predict_saved_ensemble_streaming,
    tasks_from_manifest,
)
from ckan_specnet.paths import PredictPaths
from ckan_specnet.plot import class_meaning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-sample prediction with released CKAN-SpecNet models."
    )
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-name", type=str, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = PredictPaths.from_args(args, LAUNCH_DIR)

    configure_torch_runtime()

    manifest = load_manifest(paths.run_dir)
    tasks = tasks_from_manifest(manifest)

    test_df = pl.read_parquet(paths.test)
    subset = test_df.filter(pl.col(EVAL_NAME_COL) == args.eval_name)

    if subset.height == 0:
        raise ValueError(f"Empty eval subset: {args.eval_name}")

    if not 0 <= args.sample_index < subset.height:
        raise IndexError(
            f"sample-index={args.sample_index} out of range for "
            f"{args.eval_name} with {subset.height} samples."
        )

    x, y, _ = frame_to_xy(subset.slice(args.sample_index, 1), tasks)

    cfg = replace(
        Config(),
        test_parquet=paths.test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    pred_result = predict_saved_ensemble_streaming(
        run_dir=paths.run_dir,
        manifest=manifest,
        x=x,
        tasks=tasks,
        cfg=cfg,
        device=device_of(),
    )

    rows = []

    for task in tasks.all:
        prob = pred_result["prob"][task][0]
        pred_class = int(np.argmax(prob))
        true_class = int(y[task][0])

        rows.append(
            {
                "task": task,
                "n_classes": tasks.num_classes[task],
                "true_class": true_class,
                "true_label": class_meaning(task, true_class),
                "pred_class": pred_class,
                "pred_label": class_meaning(task, pred_class),
                "probability": float(prob[pred_class]),
                "correct": pred_class == true_class,
            }
        )

    paths.out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(paths.out)
    print(f"Saved prediction: {paths.out}")


if __name__ == "__main__":
    main()
