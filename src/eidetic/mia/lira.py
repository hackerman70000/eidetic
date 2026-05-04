"""Likelihood Ratio Attack (Carlini et al. 2022) for diffusion models.

Shadow models trained on random subsets of the training set provide two
loss distributions for each candidate sample: `IN` (when the sample was
in training) and `OUT` (when it wasn't). The attack thresholds the
log-likelihood ratio between Gaussian fits of these distributions.

This module implements the *math* of LiRA — the heavy part is producing
the shadow-model loss matrices, which lives in `experiments/`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from eidetic.mia.base import MembershipInferenceAttack


@dataclass(frozen=True)
class LiRADistributions:
    """Per-sample Gaussian fits of `IN` and `OUT` loss distributions."""

    in_mean: np.ndarray  # shape: (n_samples,)
    in_std: np.ndarray
    out_mean: np.ndarray
    out_std: np.ndarray

    @classmethod
    def fit(
        cls,
        in_losses: np.ndarray,
        out_losses: np.ndarray,
        *,
        eps: float = 1e-6,
    ) -> LiRADistributions:
        """Fit per-sample Gaussians from shadow-model loss matrices.

        Both input arrays have shape `(n_samples, n_shadow_models)`. NaNs
        are interpreted as "this sample was missing from this shadow
        model's loss measurement" and ignored.
        """
        if in_losses.shape[0] != out_losses.shape[0]:
            raise ValueError(
                f"in_losses and out_losses must share axis 0; "
                f"got {in_losses.shape[0]} vs {out_losses.shape[0]}"
            )
        return cls(
            in_mean=np.nanmean(in_losses, axis=1),
            in_std=np.maximum(np.nanstd(in_losses, axis=1), eps),
            out_mean=np.nanmean(out_losses, axis=1),
            out_std=np.maximum(np.nanstd(out_losses, axis=1), eps),
        )


def gaussian_log_pdf(x: float, mean: float, std: float) -> float:
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}")
    return -0.5 * math.log(2 * math.pi * std * std) - 0.5 * ((x - mean) ** 2) / (std * std)


@dataclass
class LiRAAttack(MembershipInferenceAttack):
    """Likelihood-ratio score for a single candidate sample.

    The score is `log P(loss | IN) - log P(loss | OUT)` under per-sample
    Gaussian fits — higher means the observed loss looks more like the
    `IN` distribution (sample was in training).
    """

    distributions: LiRADistributions

    name: ClassVar[str] = "lira"

    def score(self, sample_loss: float, **kwargs: object) -> float:
        index = kwargs.get("sample_index")
        if not isinstance(index, int):
            raise ValueError("LiRAAttack.score requires `sample_index=int`.")
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
