"""Lightweight DDPM training & sampling wrapper around `diffusers`.

GPU experiments live in `experiments/`; this module provides only the
plumbing so the same code can run locally on CPU (slowly) or remotely on
GPU. The classes here are intentionally stateless wrappers — they don't
own training data, they just expose a clean API around the heavy bits
of `diffusers`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

try:
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
except ImportError as exc:  # pragma: no cover - optional dep
    raise ImportError(
        "eidetic.models.ddpm requires the `diffusion` extra: `pip install -e .[diffusion]`"
    ) from exc


@dataclass
class DDPMConfig:
    image_size: int = 32
    in_channels: int = 3
    base_channels: int = 64
    channel_mults: tuple[int, ...] = (1, 2, 2, 2)
    num_train_timesteps: int = 1000
    beta_schedule: str = "linear"


def build_unet(config: DDPMConfig) -> UNet2DModel:
    return UNet2DModel(
        sample_size=config.image_size,
        in_channels=config.in_channels,
        out_channels=config.in_channels,
        layers_per_block=2,
        block_out_channels=tuple(config.base_channels * m for m in config.channel_mults),
        down_block_types=tuple("DownBlock2D" for _ in config.channel_mults),
        up_block_types=tuple("UpBlock2D" for _ in config.channel_mults),
    )


def build_scheduler(config: DDPMConfig) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_schedule=config.beta_schedule,
    )


class DDPMSampler:
    """Generate `n_samples` images from a trained UNet + scheduler."""

    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: DDPMScheduler,
        device: str = "auto",
    ) -> None:
        self.device = self._resolve_device(device)
        self.unet = unet.to(self.device)
        self.scheduler = scheduler
        self.pipeline = DDPMPipeline(unet=self.unet, scheduler=self.scheduler)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @torch.no_grad()
    def sample(
        self, n_samples: int, *, generator: torch.Generator | None = None
    ) -> list[np.ndarray]:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        logger.info(f"sampling {n_samples} images on {self.device}")
        out = self.pipeline(batch_size=n_samples, generator=generator, output_type="np")
        return [np.asarray(img, dtype=np.float32) for img in out.images]


def diffusion_loss(
    image: torch.Tensor,
    unet: UNet2DModel,
    scheduler: DDPMScheduler,
    *,
    timestep: int,
    noise: torch.Tensor | None = None,
) -> float:
    """Per-image diffusion training loss at the supplied timestep.

    Used by `mia.strong_lira.estimate_expected_loss` when wired through
    a closure that captures `unet` and `scheduler`.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if noise is None:
        noise = torch.randn_like(image)
    t = torch.tensor([timestep], dtype=torch.long, device=image.device)
    noisy = scheduler.add_noise(image, noise, t)
    pred = unet(noisy, t).sample
    return float(torch.nn.functional.mse_loss(pred, noise).item())


def train_ddpm(
    images: Sequence[torch.Tensor],
    config: DDPMConfig,
    *,
    n_steps: int = 200_000,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    device: str = "auto",
) -> tuple[UNet2DModel, DDPMScheduler]:
    """Skeleton training loop — kept short so it's easy to specialise.

    For real CIFAR-10 reproductions paper §5 trains 16 such models;
    `experiments/train_cifar_shadow_models.py` orchestrates that.
    """
    if not images:
        raise ValueError("Need at least one training image.")

    unet = build_unet(config)
    scheduler = build_scheduler(config)
    target_device = DDPMSampler._resolve_device(device)
    unet = unet.to(target_device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
    stack = torch.stack([img.to(target_device) for img in images])
    n_samples = stack.shape[0]

    unet.train()
    for step in range(n_steps):
        idx = torch.randint(0, n_samples, (batch_size,), device=target_device)
        batch = stack[idx]
        noise = torch.randn_like(batch)
        t = torch.randint(
            0, scheduler.config.num_train_timesteps, (batch.shape[0],), device=target_device
        )
        noisy = scheduler.add_noise(batch, noise, t)
        pred = unet(noisy, t).sample
        loss = torch.nn.functional.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        if step % max(1, n_steps // 20) == 0:
            logger.info(f"step={step}/{n_steps} loss={loss.item():.4f}")

    unet.eval()
    return unet, scheduler
