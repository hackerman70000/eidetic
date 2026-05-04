"""Image-to-image distance measures used in Carlini et al. 2023.

Standard pixel-space l2 (Definition 1 default), the *tiled* l2 used to
discount memorized images that share global colour with many training
samples (paper §4.2.1), and the adaptive nearest-neighbor normalization
from §5.1 that controls false positives in untargeted extraction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _as_array(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Empty image array.")
    return arr


def l2(a: np.ndarray, b: np.ndarray) -> float:
    """Per-element l2 distance: sqrt(sum((a-b)^2) / d).

    Matches the paper's metric: a normalized euclidean distance so the
    threshold delta is comparable across image resolutions.
    """
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch: {arr_a.shape} vs {arr_b.shape}")
    diff = arr_a - arr_b
    return float(np.sqrt(np.mean(diff * diff)))


def tiled_l2(
    a: np.ndarray,
    b: np.ndarray,
    *,
    tile: int = 128,
    grid: int = 4,
) -> float:
    """Maximum l2 over a `grid x grid` partition of `tile x tile` patches.

    Carlini §4.2.1 uses 16 non-overlapping 128x128 tiles on 512x512
    Stable Diffusion outputs and reports the maximum tile distance — this
    eliminates false positives from images with similar global colour.
    """
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch: {arr_a.shape} vs {arr_b.shape}")
    if arr_a.ndim < 2:
        raise ValueError(f"Need at least 2 spatial dims, got {arr_a.shape}")

    h: int
    w: int
    if arr_a.ndim == 2:
        h, w = int(arr_a.shape[-2]), int(arr_a.shape[-1])
    else:
        h, w = int(arr_a.shape[-3]), int(arr_a.shape[-2])

    if h < tile * grid or w < tile * grid:
        raise ValueError(f"Image {h}x{w} too small for {grid}x{grid} grid of {tile}x{tile} tiles.")

    distances: list[float] = []
    for i in range(grid):
        for j in range(grid):
            slicer: tuple[slice, ...]
            if arr_a.ndim == 2:
                slicer = (slice(i * tile, (i + 1) * tile), slice(j * tile, (j + 1) * tile))
            else:
                slicer = (
                    slice(i * tile, (i + 1) * tile),
                    slice(j * tile, (j + 1) * tile),
                    slice(None),
                )
            distances.append(l2(arr_a[slicer], arr_b[slicer]))
    return max(distances)


def adaptive_distance(
    candidate: np.ndarray,
    target: np.ndarray,
    neighbors: Sequence[np.ndarray],
    *,
    alpha: float = 0.5,
) -> float:
    """Carlini §5.1 adaptive metric: l2 to target / (alpha * mean l2 to neighbors).

    Score is small only when `candidate` is meaningfully *closer* to
    `target` than to its k nearest neighbors in the training set —
    suppressing false positives from images with many similar peers.
    """
    if not neighbors:
        raise ValueError("Need at least one neighbor for adaptive distance.")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    target_distance = l2(candidate, target)
    neighbor_distances = [l2(candidate, n) for n in neighbors]
    denom = alpha * float(np.mean(neighbor_distances))
    if denom == 0:
        return float("inf") if target_distance > 0 else 0.0
    return target_distance / denom
