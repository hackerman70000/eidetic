"""Clique-based memorization detection from Carlini et al. §4.2.1.

For each prompt the attacker generates N candidate images, builds a graph
where edges connect pairs whose distance is below a threshold, and flags
the prompt as memorized if a clique of size >= `clique_size` exists.

The greedy algorithm below is sufficient because the graphs are small
(N ~ 500) and we only need a single sufficiently-large clique.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def build_similarity_graph(
    samples: Sequence[np.ndarray],
    *,
    distance_threshold: float,
    distance_fn: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Boolean adjacency matrix; True iff `distance_fn(i, j) <= threshold`."""
    n = len(samples)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if distance_fn(samples[i], samples[j]) <= distance_threshold:
                adj[i, j] = True
                adj[j, i] = True
    return adj


def find_largest_clique(adjacency: np.ndarray) -> list[int]:
    """Return indices of a maximal clique discovered by greedy expansion.

    Greedy: pick the highest-degree node, intersect candidates with its
    neighbors, repeat. Approximate, but matches paper usage where any
    clique above the threshold suffices.
    """
    n = adjacency.shape[0]
    if n == 0:
        return []

    degrees = adjacency.sum(axis=1)
    seed = int(np.argmax(degrees))

    clique = [seed]
    candidates = set(int(j) for j in np.where(adjacency[seed])[0])

    while candidates:
        best = max(candidates, key=lambda v: int(adjacency[v, list(candidates)].sum()))
        clique.append(best)
        candidates &= set(int(j) for j in np.where(adjacency[best])[0])

    return sorted(clique)


def detect_memorization(
    samples: Sequence[np.ndarray],
    *,
    distance_threshold: float,
    distance_fn: Callable[[np.ndarray, np.ndarray], float],
    clique_size: int = 10,
) -> tuple[bool, list[int]]:
    """Memorization is flagged when the largest clique has size >= `clique_size`."""
    if clique_size <= 1:
        raise ValueError(f"clique_size must be >= 2, got {clique_size}")
    adj = build_similarity_graph(
        samples, distance_threshold=distance_threshold, distance_fn=distance_fn
    )
    clique = find_largest_clique(adj)
    return len(clique) >= clique_size, clique
