"""Diffusion model wrappers (optional `diffusion` extra)."""

from __future__ import annotations

__all__: list[str] = []

try:
    from eidetic.models.ddpm import (  # noqa: F401
        DDPMConfig,
        DDPMSampler,
        build_scheduler,
        build_unet,
        diffusion_loss,
        train_ddpm,
    )

    __all__.extend(
        [
            "DDPMConfig",
            "DDPMSampler",
            "build_scheduler",
            "build_unet",
            "diffusion_loss",
            "train_ddpm",
        ]
    )
except ImportError:  # pragma: no cover
    pass
