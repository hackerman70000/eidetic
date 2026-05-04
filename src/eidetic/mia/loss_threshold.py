"""Loss-threshold MIA (Yeom et al. 2018) — paper §5.2 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from eidetic.mia.base import MembershipInferenceAttack


@dataclass
class LossThresholdAttack(MembershipInferenceAttack):
    """Predict `member` when the diffusion loss falls below `threshold`.

    We return `-loss` as the membership score so that higher = more
    likely member (consistent with the rest of the MIA suite).
    """

    name: ClassVar[str] = "loss_threshold"

    def score(self, sample_loss: float, **kwargs: object) -> float:
        del kwargs
        return -float(sample_loss)
