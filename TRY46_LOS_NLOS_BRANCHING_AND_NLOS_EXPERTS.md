# Try 46: Explicit LoS/NLoS Branching with NLoS Experts

`Try 46` is the first branch in this project that treats `LoS` and `NLoS` as structurally different prediction problems inside the network itself.

## Why this branch exists

Recent evidence showed a consistent pattern:

- `LoS` is already relatively well explained by the prior
- `NLoS` remains the dominant error source
- changing only the backbone (`U-Net` vs `PMNet`) does not solve that gap

So `Try 46` changes the problem formulation rather than just increasing capacity again.

## Core design

The model keeps:

- a calibrated physical prior
- a PMNet-style shared trunk
- path-loss residual learning

But it replaces the single residual head with:

1. a lightweight `LoS` residual head
2. a stronger `NLoS`-only MoE residual head

The final residual is blended by the explicit `LoS` map:

- `LoS` pixels use mostly the `LoS` head
- `NLoS` pixels use mostly the `NLoS` experts

## Why the NLoS head uses experts

Inside `NLoS`, the project likely still contains multiple sub-regimes such as:

- shallow `NLoS`
- deep shadow / deep `NLoS`
- dense-urban `NLoS`
- low-antenna difficult `NLoS`

So `Try 46` does not use one single strong `NLoS` head.

It uses:

- one shared trunk
- one `NLoS` gating module
- several small `NLoS` experts

This keeps the specialization where it matters without exploding the whole model size.

## Training changes

`Try 46` also changes the loss structure:

- standard final reconstruction loss on the full valid mask
- residual loss relative to the calibrated prior
- multiscale path-loss loss
- branch-specific supervision:
  - `LoS` branch is supervised only on `LoS` valid pixels
  - `NLoS` branch is supervised only on `NLoS` valid pixels
- expert-balance regularization for the `NLoS` MoE head

This is important because architecture-only branching would not be enough if the loss still encouraged both heads to learn the same average solution.

## Extra NLoS diagnostics

`Try 46` also expands the validation JSON so that `NLoS` is not reported as one single bucket only.

It now reports:

- overall `NLoS`
- `NLoS` by `city type`
- `NLoS` by antenna-height bin
- `NLoS` by shadow-depth proxy:
  - `shallow_shadow`
  - `medium_shadow`
  - `deep_shadow`

This makes it easier to see whether the experts are failing at:

- mild shadow edges
- medium shadow interiors
- or deep blocked regions

## Sources behind the idea

### LoS / NLoS segmentation logic

- [Spatially Consistent Air-to-Ground Channel Modeling and Simulation via 3D Shadow Projections](https://arxiv.org/abs/2511.15412)
- [Spatially Consistent Air-to-Ground Channel Modeling with Probabilistic LOS/NLOS Segmentation](https://arxiv.org/abs/2506.12794)

These motivate the idea that:

- visibility/shadow structure should be explicit,
- and `LoS`/`NLoS` should not be treated as one homogeneous regime.

### MoE for heterogeneous dense prediction

- [Multi-Task Dense Prediction via Mixture of Low-Rank Experts (MLoRE)](https://arxiv.org/abs/2403.17749)
- [Heterogeneous Mixture of Experts for Remote Sensing Image Super-Resolution](https://arxiv.org/abs/2502.09654)

These do not solve our exact problem, but they support the engineering choice that heterogeneous spatial regimes are a good use case for expert specialization.

## Current launch policy

`Try 46` can be launched before `Try 45` because it does not depend on the prior-only `NLoS < 20 dB` gate in the same way.

The rationale is:

- `Try 45` is a stronger version of the same prior-centered line
- `Try 46` is already a formulation change inside the model itself

So `Try 46` is a valid exploratory branch even while the prior-only `Try 45` gate is still unmet.
