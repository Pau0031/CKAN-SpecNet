from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ikan.KAN import KAN

from ckan_specnet.core import ModelConfig, TaskCatalog


def activation(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU,
        "mish": nn.Mish,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }.get(name.lower(), nn.ReLU)()


def enable_kan_speed(module: nn.Module) -> None:
    for submodule in module.modules():
        speed = getattr(submodule, "speed", None)
        if callable(speed):
            speed()


class ECA(nn.Module):
    def __init__(self, channels: int, b: int = 1, g: int = 2):
        super().__init__()
        kernel = int(abs((math.log2(channels) + b) / g))
        kernel = kernel if kernel % 2 else kernel + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel, padding=kernel // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.adaptive_avg_pool1d(x, 1).squeeze(-1).unsqueeze(1)
        y = torch.sigmoid(self.conv(y)).squeeze(1).unsqueeze(-1)
        return x * y


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        layers = []
        in_channels = 1

        for i, (out_channels, kernel) in enumerate(
            zip(cfg.conv_channels, cfg.conv_kernels, strict=True)
        ):
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel),
                nn.BatchNorm1d(out_channels),
                activation(cfg.act_name),
            ]

            if cfg.pool_sizes[i]:
                layers.append(nn.AvgPool1d(cfg.pool_sizes[i], cfg.pool_sizes[i]))

            if i in cfg.eca_positions:
                layers.append(ECA(out_channels))

            in_channels = out_channels

        self.net = nn.Sequential(*layers)
        self.out_channels = cfg.conv_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(1) if x.dim() == 2 else x)


class GlobalHead(nn.Module):
    def __init__(self, cfg: ModelConfig, channels: int):
        super().__init__()

        self.mode = cfg.adaptive_pool_mode
        self.avg = nn.AdaptiveAvgPool1d(cfg.adaptive_pool_size)
        self.max = nn.AdaptiveMaxPool1d(cfg.adaptive_pool_size)

        width = channels * cfg.adaptive_pool_size
        width *= 2 if self.mode == "avgmax" else 1

        self.proj = nn.Sequential(
            nn.Linear(width, cfg.fc_hidden),
            nn.BatchNorm1d(cfg.fc_hidden),
            activation(cfg.act_name),
            nn.Dropout(cfg.dropout_fc),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "avgmax":
            x = torch.cat([self.avg(x).flatten(1), self.max(x).flatten(1)], dim=1)
        else:
            x = self.avg(x).flatten(1)

        return self.proj(x)


def task_heads(cfg: ModelConfig, tasks: TaskCatalog) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            task: nn.Sequential(
                nn.Linear(cfg.fc_hidden, cfg.head_hidden),
                activation(cfg.act_name),
                nn.Dropout(cfg.dropout_head),
                nn.Linear(cfg.head_hidden, n_classes),
            )
            for task, n_classes in tasks.num_classes.items()
        }
    )


class CNN(nn.Module):
    def __init__(self, input_size: int, cfg: ModelConfig, tasks: TaskCatalog):
        super().__init__()
        self.tasks = tasks
        self.backbone = Backbone(cfg)
        self.global_head = GlobalHead(cfg, self.backbone.out_channels)
        self.heads = task_heads(cfg, tasks)

    def forward(self, x: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        z = self.global_head(self.backbone(x))
        return {task: self.heads[task](z) for task in self.tasks.all}


class ContributionKAN(nn.Module):
    def __init__(self, input_size: int, cfg: ModelConfig, tasks: TaskCatalog):
        super().__init__()

        self.input_size = input_size
        self.cfg = cfg
        self.tasks = tasks
        self.num_basis = cfg.contrib_num_basis

        self.register_buffer(
            "coord",
            torch.linspace(-1.0, 1.0, input_size).view(input_size, 1),
            persistent=False,
        )

        self.coord_kan = KAN(
            layers_hidden=[1, cfg.contrib_hidden, cfg.contrib_num_basis],
            grid_size=cfg.kan_grid_size,
            spline_order=cfg.kan_spline_order,
            scale_noise=cfg.kan_scale_noise,
            scale_base=cfg.kan_scale_base,
            scale_spline=cfg.kan_scale_spline,
            base_activation=nn.SiLU,
            grid_eps=cfg.kan_grid_eps,
            grid_range=cfg.kan_grid_range,
        )

        self.coeff = nn.ModuleDict(
            {
                task: nn.Linear(cfg.fc_hidden, n_classes * cfg.contrib_num_basis)
                for task, n_classes in tasks.num_classes.items()
            }
        )

        self.bias = nn.ModuleDict(
            {
                task: nn.Linear(cfg.fc_hidden, n_classes)
                for task, n_classes in tasks.num_classes.items()
            }
        )

        self.alpha = nn.ParameterDict(
            {
                task: nn.Parameter(torch.tensor(cfg.contrib_alpha_init))
                for task in tasks.all
            }
        )

        self.reset_parameters(cfg.contrib_init_std)
        self._laplacian_loss: torch.Tensor | None = None
        enable_kan_speed(self)

    def reset_parameters(self, std: float) -> None:
        for layer in self.coeff.values():
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            nn.init.zeros_(layer.bias)

        for layer in self.bias.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def basis(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        phi = self.coord_kan(self.coord.to(device=device, dtype=torch.float32))
        phi = (phi - phi.mean(0, keepdim=True)) / (phi.std(0, keepdim=True) + 1e-6)
        return phi.to(dtype=dtype)

    def signal(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1) if x.dim() == 3 else x

        if not self.cfg.contrib_signal_norm:
            return x

        return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)

    @staticmethod
    def laplacian_1d_loss(weight: torch.Tensor) -> torch.Tensor:
        if weight.shape[-1] < 3:
            return weight.new_tensor(0.0)

        lap = weight[:, :, 2:] - 2.0 * weight[:, :, 1:-1] + weight[:, :, :-2]
        return lap.pow(2).mean()

    def regularization_loss(self) -> torch.Tensor:
        lam = float(getattr(self.cfg, "contrib_lap_lambda", 0.0))

        if lam <= 0 or self._laplacian_loss is None:
            return self.coord.new_tensor(0.0)

        return lam * self._laplacian_loss

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        return_contrib: bool = False,
    ):
        x = self.signal(x)
        batch_size, length = x.shape

        if length != self.input_size:
            raise ValueError(f"Expected input length {self.input_size}, got {length}")

        phi = self.basis(z.dtype, z.device)
        signal_basis = torch.matmul(x.to(z.dtype), phi) / length

        logits = {}
        contrib = {} if return_contrib else None

        self._laplacian_loss = None
        lap_losses = []

        use_laplacian = (
            self.training
            and float(getattr(self.cfg, "contrib_lap_lambda", 0.0)) > 0
            and getattr(self.cfg, "contrib_lap_on", "weight") == "weight"
        )

        for task in self.tasks.all:
            n_classes = self.tasks.num_classes[task]
            coeff = self.coeff[task](z).view(batch_size, n_classes, self.num_basis)

            delta = torch.einsum("bcr,br->bc", coeff, signal_basis)
            delta = delta + self.bias[task](z)
            logits[task] = self.alpha[task].to(delta.dtype) * delta

            if return_contrib or use_laplacian:
                weight = torch.einsum("bcr,lr->bcl", coeff, phi)

                if use_laplacian:
                    lap_losses.append(self.laplacian_1d_loss(weight))

                if return_contrib:
                    contrib[task] = (
                        self.alpha[task].to(weight.dtype)
                        * weight
                        * x.to(weight.dtype).unsqueeze(1)
                        / length
                    )

        self._laplacian_loss = (
            torch.stack(lap_losses).mean() if lap_losses else x.new_tensor(0.0)
        )

        return (logits, contrib) if return_contrib else logits


class CNNContributionKAN(nn.Module):
    def __init__(self, input_size: int, cfg: ModelConfig, tasks: TaskCatalog):
        super().__init__()
        self.tasks = tasks
        self.backbone = Backbone(cfg)
        self.global_head = GlobalHead(cfg, self.backbone.out_channels)
        self.base_heads = task_heads(cfg, tasks)
        self.contrib = ContributionKAN(input_size, cfg, tasks)

    def regularization_loss(self) -> torch.Tensor:
        return self.contrib.regularization_loss()

    def forward(
        self,
        x: torch.Tensor,
        return_contrib: bool = False,
        logit_mode: str = "full",
    ):
        z = self.global_head(self.backbone(x))
        base = {task: self.base_heads[task](z) for task in self.tasks.all}

        if logit_mode == "base":
            return (base, None) if return_contrib else base

        if return_contrib:
            delta, contrib = self.contrib(x, z, return_contrib=True)
        else:
            delta = self.contrib(x, z)
            contrib = None

        if logit_mode == "delta":
            return (delta, contrib) if return_contrib else delta

        if logit_mode != "full":
            raise ValueError(f"Unknown logit_mode: {logit_mode}")

        out = {task: base[task] + delta[task] for task in self.tasks.all}
        return (out, contrib) if return_contrib else out


def build_model(input_size: int, cfg: ModelConfig, tasks: TaskCatalog) -> nn.Module:
    if cfg.model_name == "cnn":
        return CNN(input_size, cfg, tasks)

    if cfg.model_name == "cnn_contrib_kan":
        return CNNContributionKAN(input_size, cfg, tasks)

    raise ValueError(f"Unknown model_name: {cfg.model_name}")
