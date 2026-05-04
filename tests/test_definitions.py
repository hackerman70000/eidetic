from __future__ import annotations

import numpy as np

from eidetic.core.definitions import (
    EideticMemorizationResult,
    ExtractionResult,
    count_neighbors_within_delta,
)
from eidetic.core.distances import l2


def test_extraction_result_threshold():
    extracted = ExtractionResult(
        target=np.zeros((4, 4)),
        extracted=np.zeros((4, 4)) + 0.1,
        distance=0.05,
        delta=0.1,
    )
    assert extracted.is_extracted is True

    too_far = ExtractionResult(
        target=np.zeros((4, 4)),
        extracted=np.ones((4, 4)),
        distance=1.0,
        delta=0.1,
    )
    assert too_far.is_extracted is False


def test_eidetic_requires_both_extraction_and_low_k():
    image = np.zeros((4, 4))
    eidetic = EideticMemorizationResult(
        target=image,
        extracted=image,
        distance=0.05,
        delta=0.1,
        duplicate_count=2,
        k=10,
    )
    assert eidetic.is_eidetic_memorized is True

    too_many_duplicates = EideticMemorizationResult(
        target=image,
        extracted=image,
        distance=0.05,
        delta=0.1,
        duplicate_count=15,
        k=10,
    )
    assert too_many_duplicates.is_eidetic_memorized is False


def test_count_neighbors_within_delta():
    target = np.zeros((4, 4))
    pool = [
        np.zeros((4, 4)),
        np.zeros((4, 4)) + 0.05,
        np.zeros((4, 4)) + 0.5,
        np.zeros((4, 4)) + 1.0,
    ]
    n = count_neighbors_within_delta(target, pool, delta=0.1, distance_fn=l2)
    assert n == 2
