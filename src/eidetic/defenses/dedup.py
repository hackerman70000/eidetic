"""Deduplication probe — Carlini §7.1.

Carlini shows that simple bit-for-bit dedup misses the relevant kind of
redundancy: near-duplicates that differ in a few pixels but share style,
captions, or framing dominate the memorized set. The recommended audit
embeds every training image with a vision encoder (CLIP) and removes
samples whose nearest-neighbor cosine similarity exceeds a threshold.

This module implements the embedding-space audit; the actual CLIP
encoding is delegated to a user-supplied callable so we can mock it in
tests.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

EmbedFn = Callable[[np.ndarray], np.ndarray]


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    normed = embeddings / norms
    return normed @ normed.T


@dataclass
class DedupReport:
    n_total: int
    n_duplicate: int
    duplicate_indices: list[int]
    threshold: float

    @property
    def duplicate_fraction(self) -> float:
        if self.n_total == 0:
            return 0.0
        return self.n_duplicate / self.n_total


def audit_duplicates(
    images: Sequence[np.ndarray],
    *,
    embed_fn: EmbedFn,
    threshold: float = 0.85,
) -> DedupReport:
    """Flag images whose top-1 (excluding self) cosine similarity exceeds threshold."""
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    embeddings = np.stack([embed_fn(img) for img in images], axis=0)
    sim = cosine_similarity_matrix(embeddings)
    np.fill_diagonal(sim, -np.inf)

    nearest = sim.max(axis=1)
    duplicate_idx = [int(i) for i, s in enumerate(nearest) if s >= threshold]

    return DedupReport(
        n_total=len(images),
        n_duplicate=len(duplicate_idx),
        duplicate_indices=duplicate_idx,
        threshold=threshold,
    )
