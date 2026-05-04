from __future__ import annotations

import numpy as np

from eidetic.inpainting.attack import inpaint_attack, mask_left_half, mask_random_half


def test_mask_left_half_covers_left_columns():
    image = np.zeros((10, 10, 3))
    mask = mask_left_half(image)
    assert mask[:, :5].all()
    assert not mask[:, 5:].any()


def test_mask_random_half_returns_half_image():
    rng = np.random.default_rng(0)
    image = np.zeros((10, 10, 3))
    for _ in range(10):
        mask = mask_random_half(image, rng=rng)
        assert mask.sum() == 50


def test_inpaint_attack_picks_lowest_loss_reconstructions():
    target = np.ones((4, 4, 3), dtype=np.float32)

    def inpaint(_target: np.ndarray, _mask: np.ndarray) -> np.ndarray:
        return target + np.random.default_rng().normal(0, 0.01, size=target.shape)

    def loss(rec: np.ndarray, ref: np.ndarray) -> float:
        return float(np.mean((rec - ref) ** 2))

    result = inpaint_attack(
        target,
        inpaint_fn=inpaint,
        loss_fn=loss,
        n_samples=20,
        top_k=3,
        progress=False,
    )
    assert len(result.reconstructions) == 3
    assert all(result.losses[i] <= result.losses[i + 1] for i in range(len(result.losses) - 1))
