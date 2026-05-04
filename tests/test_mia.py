from __future__ import annotations

import numpy as np
import pytest

from eidetic.mia.base import auc_log_log, tpr_at_fpr
from eidetic.mia.lira import LiRAAttack, LiRADistributions, gaussian_log_pdf
from eidetic.mia.loss_threshold import LossThresholdAttack
from eidetic.mia.strong_lira import (
    StrongLiRAAttack,
    estimate_expected_loss,
    goldilocks_timestep_search,
    horizontal_flip,
)


def test_loss_threshold_inverts_loss():
    attack = LossThresholdAttack()
    assert attack.score(2.5) == pytest.approx(-2.5)


def test_lira_score_higher_when_loss_matches_in_distribution():
    n_samples = 5
    in_losses = np.full((n_samples, 100), 1.0) + np.random.default_rng(0).normal(
        0,
        0.05,
        size=(n_samples, 100),
    )
    out_losses = np.full((n_samples, 100), 3.0) + np.random.default_rng(1).normal(
        0,
        0.05,
        size=(n_samples, 100),
    )
    dists = LiRADistributions.fit(in_losses, out_losses)
    attack = LiRAAttack(distributions=dists)

    member_score = attack.score(1.0, sample_index=0)
    nonmember_score = attack.score(3.0, sample_index=0)
    assert member_score > nonmember_score


def test_lira_requires_sample_index():
    dists = LiRADistributions.fit(
        np.zeros((1, 5)) + 1.0,
        np.zeros((1, 5)) + 3.0,
    )
    attack = LiRAAttack(distributions=dists)
    with pytest.raises(ValueError, match="sample_index"):
        attack.score(1.5)


def test_gaussian_log_pdf_peaks_at_mean():
    at_peak = gaussian_log_pdf(0.0, 0.0, 1.0)
    away = gaussian_log_pdf(2.0, 0.0, 1.0)
    assert at_peak > away


def test_strong_lira_uses_same_score_path():
    in_losses = np.full((1, 50), 1.0) + np.random.default_rng(0).normal(0, 0.1, size=(1, 50))
    out_losses = np.full((1, 50), 3.0) + np.random.default_rng(1).normal(0, 0.1, size=(1, 50))
    dists = LiRADistributions.fit(in_losses, out_losses)
    attack = StrongLiRAAttack(distributions=dists)
    score = attack.score(1.0, sample_index=0)
    assert isinstance(score, float)


def test_estimate_expected_loss_averages_mc_samples():
    calls: list[tuple[int, np.ndarray]] = []

    def loss_fn(_image: np.ndarray, timestep: int, noise: np.ndarray) -> float:
        calls.append((timestep, noise))
        return 0.5

    image = np.zeros((3, 8, 8), dtype=np.float32)
    avg = estimate_expected_loss(image, loss_fn=loss_fn, timestep=100, n_mc=4)
    assert avg == pytest.approx(0.5)
    assert len(calls) == 4
    assert all(t == 100 for t, _ in calls)


def test_estimate_expected_loss_with_augmentation_doubles_samples():
    counts = []

    def loss_fn(_image: np.ndarray, _t: int, _n: np.ndarray) -> float:
        counts.append(1)
        return 1.0

    image = np.zeros((3, 8, 8), dtype=np.float32)
    estimate_expected_loss(
        image,
        loss_fn=loss_fn,
        n_mc=3,
        augmentations=[lambda x: x, horizontal_flip],
    )
    assert len(counts) == 6


def test_goldilocks_search_returns_one_entry_per_timestep():
    def loss_fn(_image: np.ndarray, t: int, _noise: np.ndarray) -> float:
        return float(t)

    image = np.zeros((3, 4, 4), dtype=np.float32)
    result = goldilocks_timestep_search(image, loss_fn=loss_fn, timesteps=(50, 100), n_mc=2)
    assert set(result.keys()) == {50, 100}
    assert result[50] < result[100]


def test_auc_perfect_separation():
    members = [0.9, 0.8, 0.7]
    nonmembers = [0.1, 0.2, 0.3]
    assert auc_log_log(members, nonmembers) == pytest.approx(1.0)


def test_tpr_at_fpr_recovers_clean_split():
    members = [0.9, 0.8, 0.7, 0.6]
    nonmembers = [0.1, 0.2, 0.3, 0.4]
    assert tpr_at_fpr(members, nonmembers, target_fpr=0.25) == pytest.approx(1.0)


def test_tpr_at_fpr_rejects_out_of_range():
    with pytest.raises(ValueError, match="target_fpr"):
        tpr_at_fpr([1.0], [0.0], target_fpr=0.0)
