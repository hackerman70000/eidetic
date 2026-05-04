from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from loguru import logger

from eidetic.core.cliques import detect_memorization
from eidetic.core.distances import l2, tiled_l2
from eidetic.mia.base import auc_log_log, tpr_at_fpr

app = typer.Typer(
    help="eidetic — extraction & membership inference for diffusion models.",
    no_args_is_help=True,
)


@app.command(name="check-memorization")
def check_memorization(
    npz_path: Path = typer.Argument(..., help="Path to .npz with key 'samples' (n, h, w, c)."),
    distance: str = typer.Option("tiled_l2", "--distance", help="Either 'l2' or 'tiled_l2'."),
    threshold: float = typer.Option(0.15, "--threshold"),
    clique_size: int = typer.Option(10, "--clique-size"),
) -> None:
    """Run the clique extraction check on a saved batch of generated images."""
    if not npz_path.exists():
        raise typer.BadParameter(f"File not found: {npz_path}")

    arr = np.load(npz_path)
    if "samples" not in arr.files:
        raise typer.BadParameter("Expected key 'samples' in npz file.")
    samples = list(arr["samples"])

    fn = l2 if distance == "l2" else tiled_l2
    is_mem, clique = detect_memorization(
        samples,
        distance_threshold=threshold,
        distance_fn=fn,
        clique_size=clique_size,
    )
    typer.secho(
        f"\n{'MEMORIZED' if is_mem else 'no memorization'}: clique size {len(clique)}",
        fg=typer.colors.RED if is_mem else typer.colors.GREEN,
    )
    if clique:
        typer.echo(f"clique indices: {clique}")


@app.command(name="mia-summary")
def mia_summary(
    members_path: Path = typer.Argument(..., help=".npy file with member scores."),
    nonmembers_path: Path = typer.Argument(..., help=".npy file with non-member scores."),
    target_fpr: float = typer.Option(0.01, "--target-fpr"),
) -> None:
    """Compute AUC and TPR@FPR from precomputed score arrays."""
    members = np.load(members_path)
    nonmembers = np.load(nonmembers_path)

    auc = auc_log_log(members, nonmembers)
    tpr = tpr_at_fpr(members, nonmembers, target_fpr=target_fpr)

    typer.echo(f"AUC = {auc:.4f}")
    typer.echo(f"TPR @ FPR={target_fpr} = {tpr:.4f}")
    logger.info("done")


if __name__ == "__main__":
    app()
