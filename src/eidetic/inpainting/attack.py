"""Inpainting reconstruction attack (Carlini §5.3).

Mask out part of a candidate training image, ask a diffusion model to
inpaint the masked region many times, and pick the top-k reconstructions
by diffusion loss against the *original* image. The reconstruction loss
is reliably lower when the original was in the training set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

InpaintFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""Callable: `(masked_image, mask) -> inpainted_image`."""

LossFn = Callable[[np.ndarray, np.ndarray], float]
"""Callable: `(image, reference) -> diffusion_loss_against_reference`."""


def mask_left_half(image: np.ndarray) -> np.ndarray:
    """Boolean mask: True for pixels to be inpainted (left half)."""
    if image.ndim < 2:
        raise ValueError(f"Image must have spatial dims, got shape {image.shape}")
    h: int
    w: int
    if image.ndim == 2:
        h, w = int(image.shape[-2]), int(image.shape[-1])
    else:
        h, w = int(image.shape[-3]), int(image.shape[-2])
    mask = np.zeros((h, w), dtype=bool)
    mask[:, : w // 2] = True
    return mask


def mask_random_half(image: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    if rng.random() < 0.5:
        return mask_left_half(image)
    flipped = mask_left_half(image)
    return np.flip(flipped, axis=-1).copy()


@dataclass
class InpaintingResult:
    target: np.ndarray
    reconstructions: list[np.ndarray]
    losses: list[float]
    top_k: int

    @property
    def best(self) -> np.ndarray:
        if not self.reconstructions:
            raise ValueError("No reconstructions available.")
        return self.reconstructions[int(np.argmin(self.losses))]

    @property
    def top_losses(self) -> list[float]:
        order = np.argsort(self.losses)
        return [float(self.losses[i]) for i in order[: self.top_k]]


def inpaint_attack(
    target: np.ndarray,
    *,
    inpaint_fn: InpaintFn,
    loss_fn: LossFn,
    n_samples: int = 5000,
    top_k: int = 10,
    mask_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    progress: bool = True,
) -> InpaintingResult:
    """Run the inpainting attack and return the top-k reconstructions."""
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if top_k <= 0 or top_k > n_samples:
        raise ValueError(f"top_k must be in (0, n_samples], got {top_k}")

    mask = (mask_fn or mask_left_half)(target)

    reconstructions: list[np.ndarray] = []
    losses: list[float] = []
    iterator = tqdm(range(n_samples), disable=not progress)
    for _ in iterator:
        recon = inpaint_fn(target, mask)
        reconstructions.append(recon)
        losses.append(loss_fn(recon, target))

    order = np.argsort(losses)[:top_k]
    logger.info(f"inpainting attack: best loss = {losses[int(order[0])]:.4f}")
    return InpaintingResult(
        target=target,
        reconstructions=[reconstructions[int(i)] for i in order],
        losses=[float(losses[int(i)]) for i in order],
        top_k=top_k,
    )
