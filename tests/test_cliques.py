from __future__ import annotations

import numpy as np

from eidetic.core.cliques import (
    build_similarity_graph,
    detect_memorization,
    find_largest_clique,
)


def _fixed_distance(threshold_groups: list[list[int]]):
    """Distance is 0 if both indices live in the same group, else 1."""
    membership: dict[int, int] = {}
    for group_id, group in enumerate(threshold_groups):
        for idx in group:
            membership[idx] = group_id

    def fn(a: np.ndarray, b: np.ndarray) -> float:
        return 0.0 if membership[int(a[0])] == membership[int(b[0])] else 1.0

    return fn


def test_build_similarity_graph_uses_threshold():
    samples = [np.array([i]) for i in range(4)]
    distance_fn = _fixed_distance([[0, 1, 2], [3]])
    adj = build_similarity_graph(samples, distance_threshold=0.5, distance_fn=distance_fn)
    assert adj[0, 1]
    assert adj[1, 2]
    assert adj[0, 2]
    assert not adj[0, 3]


def test_find_largest_clique_returns_full_group():
    samples = [np.array([i]) for i in range(5)]
    distance_fn = _fixed_distance([[0, 1, 2, 3], [4]])
    adj = build_similarity_graph(samples, distance_threshold=0.5, distance_fn=distance_fn)
    clique = find_largest_clique(adj)
    assert set(clique) == {0, 1, 2, 3}


def test_detect_memorization_flags_above_clique_size():
    samples = [np.array([i]) for i in range(10)]
    distance_fn = _fixed_distance([list(range(8)), [8, 9]])
    is_mem, clique = detect_memorization(
        samples,
        distance_threshold=0.5,
        distance_fn=distance_fn,
        clique_size=8,
    )
    assert is_mem is True
    assert len(clique) == 8


def test_detect_memorization_clears_when_no_clique():
    samples = [np.array([i]) for i in range(6)]
    distance_fn = _fixed_distance([[i] for i in range(6)])  # no edges
    is_mem, clique = detect_memorization(
        samples,
        distance_threshold=0.5,
        distance_fn=distance_fn,
        clique_size=2,
    )
    assert is_mem is False
    assert len(clique) <= 1
