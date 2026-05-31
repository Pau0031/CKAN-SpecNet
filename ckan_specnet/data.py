from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from ckan_specnet.core import (
    COMPONENTS,
    EVAL_NAME_COL,
    SAMPLE_ID_COL,
    SOURCE,
    SPECTRUM,
    Config,
    TaskCatalog,
)


def source_summary(path: Path) -> pl.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    return (
        pl.scan_parquet(path)
        .group_by([SOURCE, COMPONENTS])
        .agg(pl.len().alias("n"))
        .sort([SOURCE, COMPONENTS])
        .collect()
    )


def spectrum_hash(value) -> str:
    arr = np.asarray(value, dtype=np.float32)
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


def add_sample_id(df: pl.DataFrame) -> pl.DataFrame:
    if SAMPLE_ID_COL in df.columns:
        return df

    ids = [
        f"{src}|{comp}|{spectrum_hash(spec)}"
        for src, comp, spec in zip(
            df[SOURCE].to_list(),
            df[COMPONENTS].to_list(),
            df[SPECTRUM].to_list(),
            strict=True,
        )
    ]

    return df.with_columns(pl.Series(SAMPLE_ID_COL, ids))


def load_frame(
    path: Path,
    sources: tuple[str, ...],
    single_component_only: bool,
    add_id: bool = True,
) -> pl.DataFrame:
    query = pl.scan_parquet(path).filter(pl.col(SOURCE).is_in(sources))

    if single_component_only:
        query = query.filter(pl.col(COMPONENTS) == 1)

    df = query.collect()
    return add_sample_id(df) if add_id else df


def take_rows(df: pl.DataFrame, index: np.ndarray) -> pl.DataFrame:
    index = np.asarray(index, dtype=np.int64)

    return (
        df.with_row_index("_row_id")
        .filter(pl.col("_row_id").is_in(index.tolist()))
        .drop("_row_id")
    )


def clip_label(df: pl.DataFrame, column: str, upper: int) -> np.ndarray:
    return np.minimum(df[column].to_numpy(), upper).astype(np.int64)


def build_targets(df: pl.DataFrame, tasks: TaskCatalog) -> dict[str, np.ndarray]:
    base_tasks = set(tasks.binary + tasks.ternary + tasks.quaternary)
    missing = sorted(task for task in base_tasks if task not in df.columns)

    if missing:
        raise ValueError(f"Missing task columns: {missing}")

    y = {task: clip_label(df, task, 1) for task in tasks.binary}
    y |= {f"{task}_3class": clip_label(df, task, 2) for task in tasks.ternary}
    y |= {f"{task}_4class": clip_label(df, task, 3) for task in tasks.quaternary}

    return y


def frame_to_xy(
    df: pl.DataFrame,
    tasks: TaskCatalog,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    if df.height == 0:
        raise ValueError("Empty dataframe.")

    x = np.vstack(df[SPECTRUM].to_list()).astype(np.float32)
    y = build_targets(df, tasks)
    sources = df[SOURCE].to_numpy()

    return x, y, sources


def take(y: dict[str, np.ndarray], index: np.ndarray) -> dict[str, np.ndarray]:
    return {task: values[index] for task, values in y.items()}


def prepare_or_load_test_parquet(cfg: Config, tasks: TaskCatalog) -> pl.DataFrame:
    path = cfg.test_parquet

    if path.is_file():
        test_df = pl.read_parquet(path)

        if EVAL_NAME_COL not in test_df.columns:
            raise ValueError(f"{path} missing column: {EVAL_NAME_COL}")

        if SAMPLE_ID_COL not in test_df.columns:
            test_df = add_sample_id(test_df)
            test_df.write_parquet(path)

        return test_df

    train_candidate_df = load_frame(
        path=cfg.parquet,
        sources=cfg.train_sources,
        single_component_only=cfg.single_component_only,
        add_id=True,
    )

    _, _, sources_all = frame_to_xy(train_candidate_df, tasks)

    _, test_idx = train_test_split(
        np.arange(train_candidate_df.height),
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=sources_all,
    )

    main_test_df = take_rows(train_candidate_df, test_idx).with_columns(
        pl.lit("main_test").alias(EVAL_NAME_COL)
    )

    eval_dfs = [main_test_df]

    for eval_name, sources in cfg.eval_sources.items():
        df = load_frame(
            path=cfg.parquet,
            sources=sources,
            single_component_only=cfg.single_component_only,
            add_id=True,
        )

        if df.height == 0:
            message = f"Empty eval set: {eval_name}, sources={sources}"
            if cfg.require_eval_sources:
                raise RuntimeError(message)
            continue

        eval_dfs.append(df.with_columns(pl.lit(eval_name).alias(EVAL_NAME_COL)))

    test_df = pl.concat(eval_dfs, how="diagonal_relaxed")
    path.parent.mkdir(parents=True, exist_ok=True)
    test_df.write_parquet(path)

    return test_df


def load_train_data_excluding_test(
    cfg: Config,
    tasks: TaskCatalog,
    test_df: pl.DataFrame,
) -> dict:
    train_candidate_df = load_frame(
        path=cfg.parquet,
        sources=cfg.train_sources,
        single_component_only=cfg.single_component_only,
        add_id=True,
    )

    train_df = train_candidate_df.join(
        test_df.select(SAMPLE_ID_COL).unique(),
        on=SAMPLE_ID_COL,
        how="anti",
    )

    if train_df.height == 0:
        raise ValueError("Train dataframe is empty after excluding test.parquet.")

    x_train, y_train, train_sources = frame_to_xy(train_df, tasks)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "train_sources": train_sources,
        "train_df_height": train_df.height,
    }


def load_eval_sets_from_test_parquet(
    test_df: pl.DataFrame,
    tasks: TaskCatalog,
) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    eval_sets = {}

    for eval_name in sorted(test_df[EVAL_NAME_COL].unique().to_list()):
        df = test_df.filter(pl.col(EVAL_NAME_COL) == eval_name)
        x, y, _ = frame_to_xy(df, tasks)
        eval_sets[eval_name] = (x, y)

    return eval_sets


class Preprocessor:
    def __init__(self, mode: str = "sample_zscore", eps: float = 1e-6):
        self.mode = mode
        self.eps = eps

    def fit(self, _: np.ndarray) -> "Preprocessor":
        return self

    def __call__(self, spectrum: np.ndarray) -> np.ndarray:
        x = np.asarray(spectrum, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.mode == "none":
            return x.astype(np.float32)

        if self.mode == "sample_zscore":
            return ((x - x.mean()) / (x.std() + self.eps)).astype(np.float32)

        raise ValueError(f"Unknown normalization: {self.mode}")


class SpectraDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: dict[str, np.ndarray] | None,
        preprocessor: Preprocessor,
    ):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = y
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        x = torch.as_tensor(self.preprocessor(self.x[index]), dtype=torch.float32)

        if self.y is None:
            return x

        y = {
            task: torch.as_tensor(values[index], dtype=torch.long)
            for task, values in self.y.items()
        }

        return x, y


def make_loader(
    x: np.ndarray,
    y: dict[str, np.ndarray] | None,
    preprocessor: Preprocessor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        SpectraDataset(x, y, preprocessor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
