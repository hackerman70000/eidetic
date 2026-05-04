from __future__ import annotations

import numpy as np

from eidetic.core.distances import l2
from eidetic.extraction.black_box import (
    ExtractionConfig,
    extract_from_prompt,
    extract_from_prompts,
)


def _fake_generator(memorized_prompt: str, memorized_image: np.ndarray, n_unique: int = 50):
    rng = np.random.default_rng(0)

    def gen(prompt: str, n: int) -> list[np.ndarray]:
        if prompt == memorized_prompt:
            return [
                memorized_image + rng.normal(0, 0.001, size=memorized_image.shape) for _ in range(n)
            ]
        return [rng.normal(0, 1, size=memorized_image.shape) for _ in range(n)]

    return gen


def test_extract_from_prompt_flags_memorization():
    target = np.zeros((8, 8, 3), dtype=np.float32)
    cfg = ExtractionConfig(
        n_candidates=20,
        distance_threshold=0.05,
        clique_size=10,
        distance_fn=l2,
    )
    result = extract_from_prompt(
        "memorized prompt",
        generator=_fake_generator("memorized prompt", target),
        config=cfg,
    )
    assert result.is_memorized is True
    assert len(result.clique_indices) >= 10


def test_extract_from_prompt_clears_random_generator():
    target = np.zeros((8, 8, 3), dtype=np.float32)
    cfg = ExtractionConfig(
        n_candidates=20,
        distance_threshold=0.05,
        clique_size=10,
        distance_fn=l2,
    )
    result = extract_from_prompt(
        "novel prompt",
        generator=_fake_generator("memorized prompt", target),
        config=cfg,
    )
    assert result.is_memorized is False
    assert result.clique_indices == []


def test_extract_from_prompts_iterates():
    target = np.zeros((8, 8, 3), dtype=np.float32)
    cfg = ExtractionConfig(
        n_candidates=20,
        distance_threshold=0.05,
        clique_size=10,
        distance_fn=l2,
    )
    results = extract_from_prompts(
        ["memorized prompt", "novel prompt"],
        generator=_fake_generator("memorized prompt", target),
        config=cfg,
        progress=False,
    )
    assert len(results) == 2
    assert results[0].is_memorized is True
    assert results[1].is_memorized is False
