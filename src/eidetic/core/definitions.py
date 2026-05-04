"""Carlini et al. memorization definitions, encoded as dataclasses."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExtractionResult:
    """Definition 1: (l, delta)-Diffusion Extraction.

    A training example x is *extractable* if there exists an algorithm A
    (without access to x) producing x_hat with l(x, x_hat) <= delta.
    """

    target: np.ndarray
    extracted: np.ndarray
    distance: float
    delta: float

    @property
    def is_extracted(self) -> bool:
        return self.distance <= self.delta


@dataclass(frozen=True)
class EideticMemorizationResult:
    """Definition 2: (k, l, delta)-Eidetic Memorization.

    `target` is k-eidetic memorized if it is extractable AND the training
    set contains at most k examples within delta of it. Low k means a
    near-unique copy was memorized; high k indicates duplicate-driven
    memorization.
    """

    target: np.ndarray
    extracted: np.ndarray
    distance: float
    delta: float
    duplicate_count: int
    k: int

    @property
    def is_eidetic_memorized(self) -> bool:
        return self.distance <= self.delta and self.duplicate_count <= self.k


def count_neighbors_within_delta(
    target: np.ndarray,
    pool: Sequence[np.ndarray],
    *,
    delta: float,
    distance_fn: Callable[[np.ndarray, np.ndarray], float],
) -> int:
    """Number of pool entries with `distance_fn(target, p) <= delta`.

    Used to compute the `k` in (k, l, delta)-Eidetic Memorization.
    """
    return sum(1 for p in pool if distance_fn(target, p) <= delta)
