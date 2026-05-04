"""Strong LiRA — Monte Carlo loss estimation + augmentation averaging.

Carlini §5.2.2. For each candidate sample:

1. Estimate the *expected* diffusion loss `E_eps[L(x, t, eps)]` by
   averaging over `n_mc` random noise samples.
2. Average over a small set of pixel-space augmentations (paper uses
   horizontal flip).

Both reduce variance in the loss estimate and push TPR @ 0.1% FPR from
~7% (vanilla LiRA) to ~44% on CIFAR-10.

This file holds the orchestration — the actual diffusion loss per
(sample, timestep, noise) tuple is computed by a `LossFn` callable
supplied by the user (typically wrapping a `diffusers` pipeline).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from eidetic.mia.base import MembershipInferenceAttack
from eidetic.mia.lira import LiRADistributions, gaussian_log_pdf

DiffusionLossFn = Callable[[np.ndarray, int, np.ndarray], float]
"""Callable: `(image, timestep, noise) -> scalar loss`."""

Augmentation = Callable[[np.ndarray], np.ndarray]


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    return np.flip(image, axis=-2 if image.ndim >= 3 else -1).copy()


def estimate_expected_loss(
    image: np.ndarray,
    *,
    loss_fn: DiffusionLossFn,
    timestep: int = 100,
    n_mc: int = 20,
    augmentations: Sequence[Augmentation] | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo + augmentation average of the diffusion loss.

    Augmentations default to identity-only (`augmentations=None`); pass
    `[horizontal_flip]` (or anything similar) to recover paper §5.2.2.
    """
    if n_mc <= 0:
        raise ValueError(f"n_mc must be positive, got {n_mc}")
    rng = rng or np.random.default_rng()
    augs: Sequence[Augmentation] = augmentations if augmentations else (lambda x: x,)

    samples: list[float] = []
    for aug in augs:
        view = aug(image)
        for _ in range(n_mc):
            noise = rng.standard_normal(view.shape).astype(view.dtype)
            samples.append(loss_fn(view, timestep, noise))
    return float(np.mean(samples))


@dataclass
class StrongLiRAAttack(MembershipInferenceAttack):
    """LiRA scoring on top of Monte-Carlo + augmentation-averaged losses."""

    distributions: LiRADistributions

    name: ClassVar[str] = "strong_lira"

    def score(self, sample_loss: float, **kwargs: object) -> float:
        index = kwargs.get("sample_index")
        if not isinstance(index, int):
            raise ValueError("StrongLiRAAttack.score requires `sample_index=int`.")
        in_lp = gaussian_log_pdf(
            sample_loss,
            float(self.distributions.in_mean[index]),
            float(self.distributions.in_std[index]),
        )
        out_lp = gaussian_log_pdf(
            sample_loss,
            float(self.distributions.out_mean[index]),
            float(self.distributions.out_std[index]),
        )
        return float(in_lp - out_lp)


def goldilocks_timestep_search(
    image: np.ndarray,
    *,
    loss_fn: DiffusionLossFn,
    timesteps: Sequence[int] = (50, 100, 200, 300),
    n_mc: int = 20,
    rng: np.random.Generator | None = None,
) -> dict[int, float]:
    """Sweep `timesteps` and return the average loss at each — paper Fig. 9.

    The paper finds `t in [50, 300]` carries most of the signal; we
    expose this as a utility so callers can pick their `timestep`
    empirically per dataset.
    """
    return {
        t: estimate_expected_loss(image, loss_fn=loss_fn, timestep=t, n_mc=n_mc, rng=rng)
        for t in timesteps
    }
