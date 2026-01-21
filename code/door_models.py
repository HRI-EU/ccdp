"""
Diffusion model components and utility classes for the door manipulation pipeline.

This module intentionally groups all PyTorch-only code so that the simulation
and dataset code can stay lightweight and be imported without pulling heavy
dependencies until needed.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn


def default_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Normalizer:
    """
    Min/Max normalizer with a symmetric range around zero.

    Values are mapped to roughly [-1, 1] based on the min/max over the dataset.
    """

    def __init__(self) -> None:
        self._fitted = False
        self._max: Optional[torch.Tensor] = None
        self._min: Optional[torch.Tensor] = None
        self._mean: Optional[torch.Tensor] = None
        self._diff: Optional[torch.Tensor] = None

    def fit(self, data: torch.Tensor) -> None:
        self._max = data.max(dim=0, keepdim=True).values
        self._min = data.min(dim=0, keepdim=True).values
        self._mean = 0.5 * (self._max + self._min)
        self._diff = self._max - self._min
        self._fitted = True

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        assert self._fitted, "Normalizer must be fitted before use."
        return (data - self._mean.to(data.device)) / self._diff.to(data.device)

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        assert self._fitted, "Normalizer must be fitted before use."
        return data * self._diff.to(data.device) + self._mean.to(data.device)


class TrajectoryDataset(torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Lightweight dataset that optionally pairs actions with conditioning labels.

    If labels are None the dataset yields only actions.
    """

    def __init__(self, actions: torch.Tensor, labels: Optional[torch.Tensor] = None) -> None:
        actions = actions.to(torch.float32)
        self.actions = actions
        self.labels = labels.to(torch.float32) if labels is not None else None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.actions.shape[0]

    def __getitem__(self, idx: int):
        if self.labels is not None:
            return self.actions[idx], self.labels[idx]
        return self.actions[idx]


class SinusoidalPosEmb(nn.Module):
    """Classic sinusoidal positional encoding used for diffusion steps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple math
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish convenience block."""

    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple module
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """
    Residual block with FiLM conditioning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple math
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_channels, 1)
        scale, bias = embed[:, 0, ...], embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """
    1D U-Net with diffusion-step embeddings and optional global conditioning.
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: Iterable[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
            ]
        )

        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                        ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                        ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size), nn.Conv1d(start_dim, input_dim, 1))

        param_count = sum(p.numel() for p in self.parameters())
        print(f"ConditionalUnet1D parameters: {param_count:e}")

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # (B, T, C) -> (B, C, T)
        x = sample.moveaxis(-1, -2)

        timesteps = timestep if torch.is_tensor(timestep) else torch.tensor([timestep], dtype=torch.long, device=sample.device)
        if timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        skip_connections = []
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            skip_connections.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet1, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule from https://arxiv.org/abs/2102.09672.
    """
    x = torch.linspace(0, timesteps, timesteps + 1) / timesteps
    alphas = torch.cos((x + s) / (1 + s) * torch.pi / 2) ** 2
    alphas = alphas / alphas[0]
    betas = 1 - (alphas[1:] / alphas[:-1])
    return torch.clip(betas, 0.0, 0.999)


def forward_diffusion(
    x0: torch.Tensor, t: torch.Tensor, sqrt_alpha_cumprod: torch.Tensor, sqrt_one_minus_alpha_cumprod: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add noise to the input trajectory at step t and return the noisy version and the actual noise.
    """
    noise = torch.randn_like(x0)
    noisy = sqrt_alpha_cumprod[t][:, None, None] * x0 + sqrt_one_minus_alpha_cumprod[t][:, None, None] * noise
    return noisy, noise

