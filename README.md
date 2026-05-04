# eidetic

Training-data extraction and membership-inference for diffusion models.
Implements the toolkit from Carlini et al. (USENIX Security 2023):
black-box clique extraction, three membership-inference attacks
(loss-threshold, LiRA, Strong LiRA with Monte-Carlo + augmentation),
inpainting reconstruction, and the deduplication defence probe.

## Why

Diffusion models verbatim regenerate training images when prompted with
their captions. eidetic gives you the building blocks to measure that
leakage on your own model and to audit the impact of common defences.

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
# add the diffusion extra when you want to actually train / sample DDPMs:
uv pip install -e ".[dev,diffusion]"
pre-commit install
```

The `diffusion` extra pulls `diffusers`, `transformers`, `accelerate`,
`datasets`. The core attack code (distances, cliques, MIA scoring) does
not require it — useful for tests and notebook prototyping on CPU.

## Quickstart

```python
import numpy as np
from eidetic import detect_memorization, tiled_l2

# 500 candidate images for one prompt (numpy arrays HxWxC).
candidates = [...]
is_memorized, clique = detect_memorization(
    candidates, distance_threshold=0.15, distance_fn=tiled_l2, clique_size=10,
)
```

Membership inference once you have shadow-model losses:

```python
import numpy as np
from eidetic import LiRAAttack, LiRADistributions

in_losses  = np.load("results/lira/in_losses.npy")    # (n_samples, n_shadows)
out_losses = np.load("results/lira/out_losses.npy")
attack = LiRAAttack(distributions=LiRADistributions.fit(in_losses, out_losses))
score = attack.score(observed_loss, sample_index=42)
```

CLI:

```bash
eidetic check-memorization path/to/samples.npz --threshold 0.15 --clique-size 10
eidetic mia-summary results/lira/member_scores.npy results/lira/nonmember_scores.npy --target-fpr 0.01
```

## Live test

End-to-end smoke on a single GPU box:

1. `experiments/train_cifar_shadow_models.py --n-shadows 2 --n-steps 5000`
   — two undertrained DDPMs, ~30 min total.
2. `experiments/lira_cifar.py --use-flip --n-mc 5 --limit 500`
   — fits IN/OUT Gaussians and scores 500 CIFAR samples.

| Stage              | Result                      |
|--------------------|-----------------------------|
| Shadow training    | converges (loss 1.08 → 0.03)|
| 4-image sampling   | shape (32, 32, 3), pixel range [0.59, 0.95] |
| Valid sample mask  | 259/500 observed in both IN and OUT |
| LiRA scoring       | 259 member / 259 non-member scores produced |
| AUC / TPR @ FPR=1% | 1.000 / 1.000 (degenerate — see note) |

Note: with only 2 shadows each "valid" sample has exactly one IN
loss and one OUT loss. The per-sample Gaussian fit collapses to a
delta peaked at the single observation, so scoring that very same
loss against it is trivially perfect. This is the reason the paper
trains 16 shadows: every sample needs ≥ 4 observations on each side
for the Gaussian to carry meaningful variance. Pipeline is functional;
real numerical match requires the full shadow set.

## Layout

```
src/eidetic/
    core/         distances (l2, tiled_l2, adaptive), Definitions 1+2, clique detection
    extraction/   black-box clique attack pipeline
    mia/          loss-threshold, LiRA, Strong LiRA (MC + augmentation), Goldilocks search
    inpainting/   paper section 5.3 reconstruction attack
    defenses/     CLIP-based dedup probe
    models/       DDPM trainer + sampler + diffusion-loss helper (requires `diffusion` extra)
    cli/          typer entry points
tests/            pytest suite (CPU; uses mock generators / loss fns)
experiments/      train_cifar_shadow_models.py / lira_cifar.py / extract_cifar.py
docs/             METHOD.md — formal algorithm reference
```

## Workflow on a remote GPU

```bash
# on the GPU box, after cloning + uv sync --extra dev --extra diffusion
python experiments/train_cifar_shadow_models.py --n-shadows 16 --n-steps 200000
python experiments/lira_cifar.py --shadow-dir checkpoints/shadow --use-flip
python experiments/extract_cifar.py --checkpoint checkpoints/shadow/shadow_0.pt --n-samples 65536
```

## Development

```bash
pytest                  # tests (CPU, no GPU needed)
ruff check . && ruff format .
mypy src
pre-commit run --all
```

## References

- Carlini et al. *Extracting Training Data from Diffusion Models.* USENIX 2023.
- Yeom et al. *Privacy Risk in Machine Learning.* 2018 — loss-threshold MIA.
- Carlini et al. *Membership Inference Attacks From First Principles.* IEEE SP 2022 — LiRA.

See `docs/METHOD.md` for the formal algorithm reference.
