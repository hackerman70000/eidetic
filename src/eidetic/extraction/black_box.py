"""Black-box clique extraction pipeline (Carlini §4.2).

For each prompt:

1. Sample N candidate images from the model.
2. Build the pairwise similarity graph under the configured distance.
3. If the largest clique exceeds `clique_size`, flag the prompt as
   memorized. The clique members are the extracted candidates.

Compatible with any callable `generate(prompt, n) -> list[ndarray]`,
which lets us swap real diffusion pipelines for mocks during testing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from eidetic.core.cliques import detect_memorization
from eidetic.core.distances import tiled_l2

GeneratorFn = Callable[[str, int], list[np.ndarray]]
DistanceFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class ExtractionConfig:
    n_candidates: int = 500
    distance_threshold: float = 0.15
    clique_size: int = 10
    distance_fn: DistanceFn = tiled_l2


@dataclass
class PromptExtractionResult:
    prompt: str
    is_memorized: bool
    candidates: list[np.ndarray] = field(default_factory=list)
    clique_indices: list[int] = field(default_factory=list)

    @property
    def extracted(self) -> list[np.ndarray]:
        return [self.candidates[i] for i in self.clique_indices]


def extract_from_prompt(
    prompt: str,
    *,
    generator: GeneratorFn,
    config: ExtractionConfig | None = None,
) -> PromptExtractionResult:
    cfg = config or ExtractionConfig()
    candidates = generator(prompt, cfg.n_candidates)
    if len(candidates) == 0:
        return PromptExtractionResult(prompt=prompt, is_memorized=False)

    is_mem, clique = detect_memorization(
        candidates,
        distance_threshold=cfg.distance_threshold,
        distance_fn=cfg.distance_fn,
        clique_size=cfg.clique_size,
    )
    return PromptExtractionResult(
        prompt=prompt,
        is_memorized=is_mem,
        candidates=candidates,
        clique_indices=clique if is_mem else [],
    )


def extract_from_prompts(
    prompts: Iterable[str],
    *,
    generator: GeneratorFn,
    config: ExtractionConfig | None = None,
    progress: bool = True,
) -> list[PromptExtractionResult]:
    cfg = config or ExtractionConfig()
    prompts_list = list(prompts)
    iterator: Sequence[str] = tqdm(prompts_list, disable=not progress) if progress else prompts_list

    results: list[PromptExtractionResult] = []
    for prompt in iterator:
        result = extract_from_prompt(prompt, generator=generator, config=cfg)
        results.append(result)
        if result.is_memorized:
            logger.info(f"memorized: '{prompt[:60]}' (clique={len(result.clique_indices)})")
    return results
