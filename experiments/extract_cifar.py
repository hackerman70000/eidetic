"""Paper §5.1: untargeted extraction from a CIFAR-10 DDPM.

Generate 2^16 unconditional samples, search for cliques whose tiled-l2
distance falls below the adaptive threshold, report what fraction of the
training set is recovered.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=2**16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("results/extracted/samples.npz"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    from eidetic.models.ddpm import DDPMConfig, DDPMSampler, build_scheduler, build_unet

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = DDPMConfig()
    unet = build_unet(config)
    unet.load_state_dict(ckpt["unet"])
    scheduler = build_scheduler(config)
    sampler = DDPMSampler(unet, scheduler)

    all_samples: list[np.ndarray] = []
    remaining = args.n_samples
    while remaining > 0:
        batch = min(args.batch_size, remaining)
        all_samples.extend(sampler.sample(batch))
        remaining -= batch
        logger.info(f"sampled {len(all_samples)}/{args.n_samples}")

    np.savez_compressed(args.output, samples=np.stack(all_samples))
    logger.info(f"saved {len(all_samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
