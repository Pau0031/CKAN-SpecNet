from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SPECTRUM = "spectrum"
SOURCE = "source_name"
COMPONENTS = "component_count"
EVAL_NAME_COL = "_eval_name"
SAMPLE_ID_COL = "_sample_id"

COMMON_X = np.arange(552, 3844, 2.0, dtype=np.float32)


@dataclass(frozen=True)
class Config:
    parquet: Path = Path("data/unified.parquet")
    test_parquet: Path = Path("data/test.parquet")
    save_dir: Path = Path("runs")

    train_sources: tuple[str, ...] = ("sdbs", "nist_gas")
    eval_sources: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "swgdrug": ("swgdrug",),
            "xps_digitized": ("xps_digitized",),
        }
    )

    require_eval_sources: bool = True
    single_component_only: bool = True

    seed: int = 42
    test_size: float = 0.2
    folds: int = 5

    batch_size: int = 1024
    epochs: int = 10
    patience: int = 5
    lr: float = 1e-3
    weight_decay: float = 5e-2

    normalize: str = "sample_zscore"
    compile_model: bool = False
    num_workers: int = 0
    save_fp16: bool = False


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "cnn_contrib_kan"

    conv_channels: tuple[int, ...] = (32, 64, 128, 256)
    conv_kernels: tuple[int, ...] = (15, 13, 11, 7)
    pool_sizes: tuple[int | None, ...] = (3, 2, None, None)
    eca_positions: tuple[int, ...] = (2, 3)

    adaptive_pool_size: int = 64
    adaptive_pool_mode: str = "avgmax"

    fc_hidden: int = 1024
    head_hidden: int = 256
    dropout_fc: float = 0.7
    dropout_head: float = 0.3
    act_name: str = "relu"

    kan_grid_size: int = 3
    kan_spline_order: int = 3
    kan_scale_noise: float = 0.01
    kan_scale_base: float = 0.5
    kan_scale_spline: float = 0.1
    kan_grid_eps: float = 0.02
    kan_grid_range: tuple[float, float] = (-2.0, 2.0)

    contrib_num_basis: int = 64
    contrib_hidden: int = 32
    contrib_alpha_init: float = 0.01
    contrib_signal_norm: bool = False
    contrib_init_std: float = 1e-3

    contrib_lap_lambda: float = 1e-3
    contrib_lap_on: str = "weight"


@dataclass(frozen=True)
class TaskCatalog:
    binary: tuple[str, ...]
    ternary: tuple[str, ...]
    quaternary: tuple[str, ...]
    num_classes: dict[str, int]
    all: tuple[str, ...]

    @staticmethod
    def build() -> "TaskCatalog":
        binary = (
            "alkane",
            "alkene",
            "alkyne",
            "aromatics",
            "esters",
            "ketones",
            "ortho",
            "meta",
            "para",
            "alkyl_halides",
            "alcohols",
            "ether",
            "amines",
            "carbonyl_oxygen",
            "aldehydes",
            "acyl_halides",
            "amides",
            "nitriles",
            "nitro",
            "isocyanate",
            "isothiocyanate",
        )
        ternary = (
            "aldehydes",
            "acyl_halides",
            "amides",
            "nitriles",
            "nitro",
            "isocyanate",
            "isothiocyanate",
        )
        quaternary = (
            "alkyl_halides",
            "alcohols",
            "ether",
            "amines",
            "carbonyl_oxygen",
        )

        num_classes = {task: 2 for task in binary}
        num_classes |= {f"{task}_3class": 3 for task in ternary}
        num_classes |= {f"{task}_4class": 4 for task in quaternary}

        return TaskCatalog(
            binary=binary,
            ternary=ternary,
            quaternary=quaternary,
            num_classes=num_classes,
            all=tuple(num_classes),
        )

    def to_dict(self) -> dict:
        return {
            "binary": list(self.binary),
            "ternary": list(self.ternary),
            "quaternary": list(self.quaternary),
            "num_classes": dict(self.num_classes),
            "all": list(self.all),
        }

    @staticmethod
    def from_dict(data: dict) -> "TaskCatalog":
        return TaskCatalog(
            binary=tuple(data["binary"]),
            ternary=tuple(data["ternary"]),
            quaternary=tuple(data["quaternary"]),
            num_classes={key: int(value) for key, value in data["num_classes"].items()},
            all=tuple(data["all"]),
        )


TASKS = TaskCatalog.build()


def configure_torch_runtime() -> None:
    torch.set_float32_matmul_precision("high")
    torch.multiprocessing.set_sharing_strategy("file_system")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_of() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unwrap_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
