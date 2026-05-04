"""Visualisation helpers for extraction and MIA results.

Lazy-imports matplotlib so the core attack code stays light.
Install the optional extra with `uv sync --extra plotting`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _lazy_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("matplotlib not installed. Run `uv sync --extra plotting`.") from exc
    return plt


def plot_distance_histogram(
    distances: Sequence[float],
    *,
    threshold: float | None = None,
    title: str = "Distances",
    bins: int = 50,
    save_path: Path | str | None = None,
) -> None:
    """Histogram of pairwise / target distances. Use to pick `delta`."""
    plt = _lazy_pyplot()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(list(distances), bins=bins, color="steelblue", edgecolor="white")
    if threshold is not None:
        ax.axvline(threshold, color="crimson", linestyle="--", label=f"delta = {threshold}")
        ax.legend()
    ax.set_xlabel("distance")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_lira_roc(
    member_scores: Sequence[float],
    nonmember_scores: Sequence[float],
    *,
    title: str = "LiRA ROC",
    save_path: Path | str | None = None,
    log_scale: bool = True,
) -> float:
    """Plot LiRA ROC (log-log axes by default per Carlini paper) and return AUC."""
    import numpy as np
    from sklearn.metrics import auc, roc_curve

    plt = _lazy_pyplot()
    y_true = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    y_score = np.concatenate([np.asarray(member_scores), np.asarray(nonmember_scores)])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([1e-4, 1], [1e-4, 1], color="grey", lw=1, linestyle="--")
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(1e-4, 1)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)
    return roc_auc
