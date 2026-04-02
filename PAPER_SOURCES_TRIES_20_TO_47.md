# Paper Sources and Motivations: Tries 20 to 47

## Decoder and multiscale supervision

### Try 20 / Try 22

The decoder change back to bilinear upsampling was motivated by the checkerboard-artifact literature:

- Augustus Odena, Vincent Dumoulin, Chris Olah, "Deconvolution and Checkerboard Artifacts", Distill, 2016  
  [https://distill.pub/2016/deconv-checkerboard/](https://distill.pub/2016/deconv-checkerboard/)

The multiscale supervision idea follows the broader multiscale dense-prediction literature:

- Mathieu et al., "Deep Multi-Scale Video Prediction Beyond Mean Square Error", 2016  
  [https://arxiv.org/abs/1511.05440](https://arxiv.org/abs/1511.05440)

The project adaptation was:

- apply a cleaner decoder,
- and explicitly supervise path loss at more than one spatial scale.

## Prior + residual learning

### Try 41 / Try 42 / Try 47

These tries are motivated by the fact that large-scale propagation is highly structured and should not need to be relearned from scratch by the network.

The specific physical prior family was built from:

- hybrid two-ray / FSPL style large-scale propagation
- COST231-Hata-style urban attenuation
- and later A2G-inspired `NLoS` strengthening

The train-only calibrated prior pipeline is documented in:

- [FORMULA_PRIOR_CALIBRATION_SYSTEM.md](/C:/TFG/TFGpractice/FORMULA_PRIOR_CALIBRATION_SYSTEM.md)

## A2G LoS/NLoS formulas

The most important A2G source used for the enhanced `NLoS` prior family is the supervisor paper:

- "Spatially Consistent Air-to-Ground Channel Modeling and Simulation via 3D Shadow Projections"

Within that family, the relevant ideas were taken from:

- `Eq. (1)`:
  - deterministic large-scale loss plus excess-loss structure
- `Eq. (9)`:
  - base reference term
- `Eq. (10)`:
  - `LoS` angle-dependent behavior
- `Eq. (11)`:
  - `NLoS` angle-dependent excess loss
- `Eq. (12)`:
  - shadow-variability / sigma behavior

The project does **not** copy the paper as a full stochastic simulator.
Instead, it uses the deterministic structural parts of those equations to build a prior that can then be calibrated train-only and corrected by the network.

## PMNet family

### Try 42 / Try 43 / Try 44

The PMNet line was motivated by the PMNet / radiomap literature suggesting that a stronger multiscale context model could help path-loss prediction.

Relevant references:

- PMNet repository:
  [https://github.com/abman23/pmnet](https://github.com/abman23/pmnet)
- PMNet paper/materials referenced in `TFG_Proto1`

What we learned from this family:

- PMNet with prior can help,
- PMNet without prior is not convincing enough,
- and a more faithful PMNet-v3-style control still did not outperform the prior-aware branch.

That is why `Try 47` moves away from PMNet as the main backbone.

## Mixture-of-experts

### Try 45 / Try 46 / Try 47

The MoE idea is motivated by the fact that one residual head seems too blunt for highly heterogeneous regimes.

Relevant inspiration:

- MLoRE: "Multi-Task Dense Prediction via Mixture of Low-Rank Experts"  
  [https://arxiv.org/abs/2403.17749](https://arxiv.org/abs/2403.17749)
- "Heterogeneous Mixture of Experts for Remote Sensing Image Super-Resolution"  
  [https://arxiv.org/abs/2502.09654](https://arxiv.org/abs/2502.09654)

Project adaptation:

- no giant transformer MoE,
- just a small expert system where specialization is needed most,
- specifically in the `NLoS` correction branch.

## Why Try 47 is the current synthesis

`Try 47` explicitly combines three source lines:

1. The strong practical decoder lessons from the `Try 22` family
2. The calibrated prior + residual formulation from the `Try 42` family
3. The regime-specialized `NLoS` handling / MoE motivation from `Try 46`

Its key novelty relative to the recent PMNet branches is:

- keep the newer prior-aware physics,
- but return to the U-Net-style spatial backbone that empirically behaved better in this project.

## Try 48 prior refinement

`Try 48` keeps the `Try 42` learning idea (`prior + residual`, then light GAN refinement), but the current source-backed conclusion is that the prior itself still needs improvement.

### Why the old two-ray prior is too weak

The `two_ray_ground` prior used in code is a breakpoint-style large-scale model:

- free-space below crossover,
- asymptotic `40 log10(d) - 20 log10(h_tx) - 20 log10(h_rx)` above crossover.

This captures the large-scale slope change of two-ray propagation, but it is **not** a coherent interference model. Therefore it cannot naturally reproduce the fine concentric ring structure visible in the dataset.

### Source for a better LoS prior

- N. H. Ranchagoda, K. Sithamparanathan, M. Ding, A. Al-Hourani, and K. M. Gomez, "Elevation-angle based two-ray path loss model for Air-to-Ground wireless channels", *Vehicular Communications*, 2021  
  [https://www.sciencedirect.com/science/article/pii/S2214209621000620](https://www.sciencedirect.com/science/article/pii/S2214209621000620)

Why it matters:

- it treats A2G path loss as direct plus ground-reflected propagation,
- it explicitly discusses destructive-interference down-fades,
- and it is much closer to the ring-like LoS behavior seen in the maps.

Project implication:

- a stronger `Try 48` prior should use a **coherent two-ray LoS model**,
- not only a smooth crossover approximation.

### Sources for a better NLoS prior

- E. Vinogradov, A. Saboor, Z. Cui, and A. Fakhreddine, "Spatially Consistent Air-to-Ground Channel Modeling with Probabilistic LOS/NLOS Segmentation"  
  [https://arxiv.org/abs/2506.12794](https://arxiv.org/abs/2506.12794)
- local `TFG_Proto1` note `2511.10763v1` ("Millimeter-Wave UAV Channel Model with Height-Dependent Path Loss and Shadowing in Urban Scenarios")

Why they matter:

- they support explicit LoS/NLoS segmentation,
- they support a distinct NLoS large-scale law,
- and they support height-dependent NLoS behavior instead of treating NLoS as a single additive excess-loss patch.

### Shadow / segmentation support

- supervisor paper local markdown `2511.15412v1`

What remains useful from that family:

- deterministic shadow-based LoS/NLoS segmentation,
- `Eq. (12)` as a shadow-severity or uncertainty cue,
- but not necessarily the literal excess-loss term alone as the final NLoS mean law.

### Additional geometric motivation

- M. Pang et al., "Geometry-Based Stochastic Probability Models for the LoS and NLoS Paths of A2G Channels under Urban Scenario"  
  [https://arxiv.org/abs/2205.09399](https://arxiv.org/abs/2205.09399)

Why it matters:

- it supports the idea that NLoS is not a single homogeneous propagation regime,
- and that geometry controls whether links are dominated by LoS, ground-specular, or richer scattering behavior.

### Practical interpretation for this project

Because the project uses a fixed carrier frequency and varying UAV height, the next gain is expected to come mainly from:

- improving the **form** of the LoS prior,
- improving the **form** of the NLoS mean law,
- and only then calibrating lightly on top.

The key lesson is:

- the project should spend less effort on ultra-precise coefficient tuning,
- and more effort on choosing the right physical prior family.
