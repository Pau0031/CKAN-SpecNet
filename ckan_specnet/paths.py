from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def cli_path(path: Path, base_dir: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else base_dir / path


@dataclass(frozen=True)
class EvalPaths:
    test: Path
    run_dir: Path
    out: Path

    @classmethod
    def from_args(cls, args, base_dir: Path) -> "EvalPaths":
        return cls(
            test=cli_path(args.test, base_dir),
            run_dir=cli_path(args.run_dir, base_dir),
            out=cli_path(args.out, base_dir),
        )


@dataclass(frozen=True)
class PredictPaths:
    test: Path
    run_dir: Path
    out: Path

    @classmethod
    def from_args(cls, args, base_dir: Path) -> "PredictPaths":
        return cls(
            test=cli_path(args.test, base_dir),
            run_dir=cli_path(args.run_dir, base_dir),
            out=cli_path(args.out, base_dir),
        )


@dataclass(frozen=True)
class PlotPaths:
    test: Path
    run_dir: Path
    out: Path

    @classmethod
    def from_args(cls, args, base_dir: Path) -> "PlotPaths":
        return cls(
            test=cli_path(args.test, base_dir),
            run_dir=cli_path(args.run_dir, base_dir),
            out=cli_path(args.out, base_dir),
        )


@dataclass(frozen=True)
class TrainPaths:
    parquet: Path
    test: Path
    out: Path

    @classmethod
    def from_args(cls, args, base_dir: Path) -> "TrainPaths":
        return cls(
            parquet=cli_path(args.parquet, base_dir),
            test=cli_path(args.test, base_dir),
            out=cli_path(args.out, base_dir),
        )


@dataclass(frozen=True)
class DigitizePaths:
    input: Path
    out: Path

    @classmethod
    def from_args(cls, args, base_dir: Path) -> "DigitizePaths":
        return cls(
            input=cli_path(args.input, base_dir),
            out=cli_path(args.out, base_dir),
        )
