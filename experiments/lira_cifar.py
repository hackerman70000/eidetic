"""Paper §5.2: compute LiRA scores on CIFAR-10 from pre-trained shadows.

Loads the shadow models produced by `train_cifar_shadow_models.py`,
computes Monte-Carlo + flip-augmented diffusion losses on every CIFAR-10
sample under every shadow, fits per-sample IN/OUT Gaussians, and reports
TPR @ FPR=1% / 0.1% (paper Fig. 10).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torchvision import datasets, transforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow-dir", type=Path, default=Path("checkpoints/shadow"))
    parser.add_argument("--out", type=Path, default=Path("results/lira"))
    parser.add_argument("--timestep", type=int, default=100, help="Goldilocks default.")
    parser.add_argument("--n-mc", type=int, default=20)
    parser.add_argument("--use-flip", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    from eidetic.mia.lira import LiRAAttack, LiRADistributions
    from eidetic.mia.strong_lira import horizontal_flip
    from eidetic.models.ddpm import DDPMConfig, build_scheduler, build_unet, diffusion_loss

    transform = transforms.Compose([transforms.ToTensor()])
    cifar = datasets.CIFAR10(
        root=str(args.data_root), train=True, download=True, transform=transform
    )
    images = torch.stack([img for img, _ in cifar])
    n_total = images.shape[0]

    shadow_files = sorted(args.shadow_dir.glob("shadow_*.pt"))
    if not shadow_files:
        raise SystemExit(f"No shadow checkpoints in {args.shadow_dir}")

    config = DDPMConfig()
    in_losses = np.full((n_total, len(shadow_files)), np.nan)
    out_losses = np.full((n_total, len(shadow_files)), np.nan)

    for shadow_idx, ckpt_path in enumerate(shadow_files):
        logger.info(f"scoring with {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        unet = build_unet(config)
        unet.load_state_dict(ckpt["unet"])
        unet.eval()
        scheduler = build_scheduler(config)

        in_mask = np.array(ckpt["in_mask"], dtype=bool)
        for i in range(n_total):
            sample = images[i]
            losses = []
            for view in (sample, horizontal_flip(sample.numpy())) if args.use_flip else (sample,):
                view_tensor = view if isinstance(view, torch.Tensor) else torch.from_numpy(view)
                for _ in range(args.n_mc):
                    losses.append(
                        diffusion_loss(view_tensor, unet, scheduler, timestep=args.timestep)
                    )
            mean_loss = float(np.mean(losses))
            (in_losses if in_mask[i] else out_losses)[i, shadow_idx] = mean_loss

    np.save(args.out / "in_losses.npy", in_losses)
    np.save(args.out / "out_losses.npy", out_losses)

    distributions = LiRADistributions.fit(in_losses, out_losses)
    attack = LiRAAttack(distributions=distributions)

    member_scores: list[float] = []
    nonmember_scores: list[float] = []
    for i in range(n_total):
        for shadow_idx in range(len(shadow_files)):
            if not np.isnan(in_losses[i, shadow_idx]):
                member_scores.append(attack.score(in_losses[i, shadow_idx], sample_index=i))
            if not np.isnan(out_losses[i, shadow_idx]):
                nonmember_scores.append(attack.score(out_losses[i, shadow_idx], sample_index=i))

    np.save(args.out / "member_scores.npy", np.array(member_scores))
    np.save(args.out / "nonmember_scores.npy", np.array(nonmember_scores))
    logger.info(
        f"computed {len(member_scores)} member and {len(nonmember_scores)} non-member scores"
    )


if __name__ == "__main__":
    main()
