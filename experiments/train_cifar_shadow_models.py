"""Paper §5: train N unconditional DDPMs on random 50% splits of CIFAR-10.

Each shadow model sees a freshly sampled half of the training set; the
out-of-set examples become its `OUT` set for LiRA. Designed for remote
GPU execution — wall-clock for one 25M-param DDPM on CIFAR-10 is in the
tens of GPU-hours, so adjust `--n-shadows` accordingly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torchvision import datasets, transforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/shadow"))
    parser.add_argument("--n-shadows", type=int, default=16, help="Paper uses 16.")
    parser.add_argument("--n-steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    transform = transforms.Compose([transforms.ToTensor()])
    cifar = datasets.CIFAR10(
        root=str(args.data_root), train=True, download=True, transform=transform
    )
    images = torch.stack([img for img, _ in cifar])  # (50000, 3, 32, 32)
    n_total = images.shape[0]

    from eidetic.models.ddpm import DDPMConfig, train_ddpm

    config = DDPMConfig()
    membership_log: list[dict] = []

    for shadow_id in range(args.n_shadows):
        in_idx = rng.choice(n_total, size=n_total // 2, replace=False)
        in_mask = np.zeros(n_total, dtype=bool)
        in_mask[in_idx] = True

        logger.info(f"Training shadow {shadow_id + 1}/{args.n_shadows} (|IN|={in_mask.sum()})")
        unet, scheduler = train_ddpm(
            list(images[in_idx]),
            config,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

        ckpt_path = args.output_dir / f"shadow_{shadow_id}.pt"
        torch.save(
            {
                "shadow_id": shadow_id,
                "unet": unet.state_dict(),
                "scheduler_config": scheduler.config,
                "in_mask": in_mask.tolist(),
                "config": vars(config),
            },
            ckpt_path,
        )
        membership_log.append({"shadow_id": shadow_id, "checkpoint": str(ckpt_path)})

    (args.output_dir / "shadows.json").write_text(json.dumps(membership_log, indent=2))
    logger.info(f"saved {args.n_shadows} shadow checkpoints to {args.output_dir}")


if __name__ == "__main__":
    main()
