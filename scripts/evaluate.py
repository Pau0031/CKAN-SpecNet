#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

LAUNCH_DIR = Path.cwd()

import polars as pl

from ckan_specnet.core import Config, configure_torch_runtime, device_of
from ckan_specnet.eval import (
    evaluate_saved_ensemble_on_eval_sets,
    load_manifest,
    save_evaluation_tables,
    tasks_from_manifest,
)
from ckan_specnet.paths import EvalPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate released five-fold CKAN-SpecNet models."
    )
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = EvalPaths.from_args(args, LAUNCH_DIR)

    configure_torch_runtime()

    manifest = load_manifest(paths.run_dir)
    tasks = tasks_from_manifest(manifest)
    test_df = pl.read_parquet(paths.test)

    cfg = replace(
        Config(),
        test_parquet=paths.test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    results = evaluate_saved_ensemble_on_eval_sets(
        run_dir=paths.run_dir,
        manifest=manifest,
        test_df=test_df,
        tasks=tasks,
        cfg=cfg,
        device=device_of(),
    )

    summary = save_evaluation_tables(results, paths.out)
    print(summary)
    print(f"Saved evaluation tables to: {paths.out}")


if __name__ == "__main__":
    main()
