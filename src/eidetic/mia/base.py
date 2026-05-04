"""Base interface for membership inference attacks on diffusion models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class MIAResult:
    """Sample-level membership inference output.

    `score` is calibrated so that higher = more confident the sample is
    a *member* of the training set. The threshold for a binary verdict
    is selected externally (e.g. to fix a target false-positive rate).
    """

    score: float
    is_member_predicted: bool
    threshold: float


class MembershipInferenceAttack(ABC):
    name: ClassVar[str]

    @abstractmethod
    def score(self, sample_loss: float, **kwargs: object) -> float:
        """Higher score = more likely member."""

    def predict(self, sample_loss: float, threshold: float, **kwargs: object) -> MIAResult:
        s = self.score(sample_loss, **kwargs)
        return MIAResult(score=s, is_member_predicted=s >= threshold, threshold=threshold)


def auc_log_log(
    member_scores: np.ndarray | list[float],
    nonmember_scores: np.ndarray | list[float],
) -> float:
    """Plain ROC AUC computed across the supplied score arrays.

    Carlini's MIA papers report attack performance as TPR @ low FPR
    (e.g. 0.1% / 1%) plotted on log-log axes; the underlying ranking
    metric is just the standard ROC AUC, which we use here.
    """
    members = np.asarray(member_scores, dtype=float)
    nonmembers = np.asarray(nonmember_scores, dtype=float)
    if members.size == 0 or nonmembers.size == 0:
        raise ValueError("Need at least one member and one non-member score.")
    y_true = np.concatenate([np.ones_like(members), np.zeros_like(nonmembers)])
    y_score = np.concatenate([members, nonmembers])
    return float(roc_auc_score(y_true, y_score))


def tpr_at_fpr(
    member_scores: np.ndarray | list[float],
    nonmember_scores: np.ndarray | list[float],
    target_fpr: float,
) -> float:
    """True positive rate at a fixed false-positive rate (paper Fig. 10)."""
    if not 0 < target_fpr < 1:
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    members = np.asarray(member_scores, dtype=float)
    nonmembers = np.asarray(nonmember_scores, dtype=float)
    threshold = float(np.quantile(nonmembers, 1 - target_fpr))
    return float(np.mean(members >= threshold))
