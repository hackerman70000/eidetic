from __future__ import annotations

import numpy as np
import pytest

from eidetic.core.distances import adaptive_distance, l2, tiled_l2


def test_l2_zero_for_identical_arrays():
    a = np.ones((4, 4, 3), dtype=np.float32)
    assert l2(a, a) == pytest.approx(0.0)


def test_l2_normalised_by_dimensionality():
    a = np.zeros((10, 10, 3), dtype=np.float32)
    b = np.ones((10, 10, 3), dtype=np.float32)
    assert l2(a, b) == pytest.approx(1.0)


def test_l2_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        l2(np.zeros((4, 4)), np.zeros((5, 4)))


def test_tiled_l2_picks_max_tile():
    a = np.zeros((512, 512, 3), dtype=np.float32)
    b = np.zeros((512, 512, 3), dtype=np.float32)
    b[:128, :128] = 1.0
    score = tiled_l2(a, b)
    assert score == pytest.approx(1.0)


def test_tiled_l2_rejects_too_small_image():
    with pytest.raises(ValueError, match="too small"):
        tiled_l2(np.zeros((64, 64, 3)), np.zeros((64, 64, 3)))


def test_adaptive_distance_below_one_when_target_closer_than_neighbors():
    target = np.array([1.0, 0.0, 0.0])
    candidate = np.array([0.95, 0.0, 0.0])
    neighbors = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    score = adaptive_distance(candidate, target, neighbors, alpha=0.5)
    assert score < 1.0


def test_adaptive_distance_rejects_empty_neighbors():
    with pytest.raises(ValueError, match="neighbor"):
        adaptive_distance(np.zeros(3), np.zeros(3), [], alpha=0.5)


def test_adaptive_distance_rejects_non_positive_alpha():
    with pytest.raises(ValueError, match="alpha"):
        adaptive_distance(np.zeros(3), np.zeros(3), [np.ones(3)], alpha=0.0)
