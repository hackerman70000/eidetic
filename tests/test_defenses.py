from __future__ import annotations

import numpy as np

from eidetic.defenses.dedup import audit_duplicates, cosine_similarity_matrix


def test_cosine_similarity_matrix_has_one_on_diagonal_for_self():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    sim = cosine_similarity_matrix(embeddings)
    assert sim.shape == (3, 3)
    np.testing.assert_allclose(np.diagonal(sim), [1.0, 1.0, 1.0])


def test_cosine_similarity_handles_zero_vectors():
    embeddings = np.array([[0.0, 0.0], [1.0, 0.0]])
    sim = cosine_similarity_matrix(embeddings)
    assert sim.shape == (2, 2)
    assert np.isfinite(sim).all()


def test_audit_duplicates_finds_pair():
    images = [np.array([[1.0]]), np.array([[2.0]]), np.array([[3.0]]), np.array([[4.0]])]
    embeddings = {
        id(images[0]): np.array([1.0, 0.0]),
        id(images[1]): np.array([0.99, 0.01]),
        id(images[2]): np.array([0.0, 1.0]),
        id(images[3]): np.array([0.0, -1.0]),
    }

    def embed(img: np.ndarray) -> np.ndarray:
        return embeddings[id(img)]

    report = audit_duplicates(images, embed_fn=embed, threshold=0.95)
    assert report.duplicate_indices == [0, 1]
    assert report.n_duplicate == 2


def test_audit_duplicates_clears_when_below_threshold():
    images = [np.array([[1.0]]), np.array([[2.0]])]
    e = {id(images[0]): np.array([1.0, 0.0]), id(images[1]): np.array([0.0, 1.0])}
    report = audit_duplicates(images, embed_fn=lambda img: e[id(img)], threshold=0.85)
    assert report.duplicate_indices == []
