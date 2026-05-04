# Method reference

## Definitions (Carlini et al. §4.1)

### Definition 1 — (l, delta)-Diffusion Extraction

A training example x is _extractable_ from a diffusion model if there
exists an efficient algorithm A (without access to x) producing
`x_hat = A(f_theta)` such that `l(x, x_hat) <= delta`.

### Definition 2 — (k, l, delta)-Eidetic Memorization

x is `(k, l, delta)`-eidetic memorized if it is extractable AND there
are at most k training examples within delta of it. Low k = near-unique
copy; high k = duplicate-driven memorization.

## Distances

- `l2(a, b) = sqrt(mean((a - b)^2))` — normalised so the same delta
  applies across resolutions.
- `tiled_l2` — partition each image into `4 x 4` non-overlapping
  `128 x 128` patches and report the _maximum_ tile l2 (paper §4.2.1).
  Removes false positives from images that share global colour but
  differ in detail.
- `adaptive_distance(candidate, target, neighbors, alpha)` —
  `l2(candidate, target) / (alpha * mean l2(candidate, neighbors))`.
  Score is small only when the candidate is meaningfully closer to the
  target than to its k nearest training neighbors (paper §5.1, alpha=0.5,
  k=50).

## Black-box clique extraction (paper §4.2)

For each prompt:

1. Generate N candidate images (paper uses 500 per prompt).
2. Build a similarity graph: edge `(i, j)` iff `tiled_l2(x_i, x_j) <= delta`.
3. Find a clique of size `>= clique_size` (paper uses 10).
4. If found, the prompt is flagged as memorized and the clique members
   are the recovered training images.

## Membership inference (paper §5.2)

Three attacks of increasing strength:

1. Loss threshold (Yeom et al. 2018) — predict `member` when the
   diffusion loss falls below tau. Cheap baseline.
2. LiRA (Carlini et al. 2022) — train shadow models on random 50%
   splits, fit per-sample IN/OUT Gaussians, score each candidate by
   the log-likelihood ratio.
3. Strong LiRA — LiRA + Monte-Carlo loss estimation
   `E_eps[L(x, t, eps)]` (paper uses ~20 samples) + horizontal-flip
   augmentation. On CIFAR-10 this improves TPR @ 0.1% FPR from ~7% to
   ~44%.

### Goldilocks zone

The diffusion loss is most informative for membership inference at
intermediate timesteps. Paper Fig. 9 sweeps `t in [1, T]` and finds
`t in [50, 300]` produces the strongest signal — too small ⇒ noise is
too gentle to discriminate, too large ⇒ noise dominates the signal.
Paper uses `t = 100` for all subsequent experiments.

## Inpainting attack (paper §5.3)

When the adversary knows a portion of the candidate image:

1. Mask half (left or right).
2. Inpaint the masked region 5,000 times.
3. Score reconstructions by diffusion loss against the _original_; take
   the top-10.

Reconstruction loss is consistently lower when the original was in
training, even on non-duplicated CIFAR-10 examples.

## Diffusion vs GANs (paper Tab. 1)

| Architecture | Extracted | FID |
| ------------ | --------- | --- |
| OpenAI-DDPM  | 301       | 2.9 |
| DDPM (Ho)    | 232       | 3.2 |
| StyleGAN-ADA | 150       | 2.9 |
| DiffBigGAN   | 57        | 4.6 |

DDPM memorizes ~2x more than the strongest comparable GAN at the same
FID. Better generative quality correlates with stronger leakage:
TPR @ 1% FPR climbs from 7% to nearly 100% as FID drops from 8 to 3.5.

## Defenses (paper §7)

- Deduplication — embed every training image with CLIP, drop those
  whose nearest-neighbor cosine similarity exceeds 0.85. On CIFAR-10
  this removes 5,275 / 50,000 images and drops extraction from 1,280
  to 986. Necessary but not sufficient.
- DP-SGD — gold-standard defence in theory. In practice the paper
  reports training divergence even at `epsilon >= 50`; gradient clipping
  alone breaks the diffusion training. Open problem.
- Canary auditing — insert random images into the training set and
  measure exposure. Paper §7.3 shows two duplicates suffice to reach
  maximum exposure.

## References

- Carlini, Hayes, Nasr, Jagielski, Sehwag, Tramèr, Balle, Ippolito,
  Wallace. _Extracting Training Data from Diffusion Models._ USENIX
  Security 2023.
