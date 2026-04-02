# Try 48 Two-Ray and NLoS Prior Refinement

## Current lesson

The first structural prior refresh for `Try 48` did **not** improve the prior-only RMSE enough. This is useful negative evidence.

It suggests that:

- the project is not bottlenecked by coefficient precision,
- the bottleneck is the **functional form** of the prior,
- especially for:
  - coherent LoS ring structure,
  - and masked ground-valid `NLoS`.

## Why fixed frequency changes the strategy

In this project, the carrier frequency is fixed while the UAV height changes from sample to sample.

That means the important physics should be driven mainly by:

- elevation angle `theta`,
- direct/reflected path-length difference,
- reflection coefficient behavior,
- LoS/NLoS segmentation,
- and obstruction severity / geometry.

It is therefore not especially useful to spend effort on extremely precise frequency-dependent coefficient fitting.

The prior should instead be improved by changing the **model family**:

- better LoS model,
- better NLoS model,
- then a light train-only calibration on top.

## What the sources say

### 1. LoS should be a coherent two-ray problem, not just a breakpoint slope change

Source:

- N. H. Ranchagoda et al., ["Elevation-angle based two-ray path loss model for Air-to-Ground wireless channels"](https://www.sciencedirect.com/science/article/pii/S2214209621000620), *Vehicular Communications*, 2021.

Most relevant idea:

- in A2G links, the direct and ground-reflected rays create elevation-dependent down-fades,
- so a smooth breakpoint-only `two_ray_ground` model is too weak if the target maps show ring-like interference structure.

Implication:

- the LoS prior should use a coherent two-ray mean-power expression,
- ideally with a physically interpretable reflection coefficient,
- and with optional damping rather than replacing it with a pure asymptotic law.

### 2. NLoS should not be a single excess-loss patch on top of a LoS-like prior

Sources:

- E. Vinogradov et al., ["Spatially Consistent Air-to-Ground Channel Modeling with Probabilistic LOS/NLOS Segmentation"](https://arxiv.org/abs/2506.12794)
- A. Saboor and E. Vinogradov, local markdown `2511.10763v1`

Most relevant idea:

- the channel should be structured through LoS/NLoS segmentation first,
- and the NLoS branch should have a distinct path-loss law,
- especially with height-dependent behavior.

Implication:

- the NLoS prior should look more like a dedicated large-scale fading law,
- not just like a LoS prior plus a scalar excess term.

### 3. Shadow geometry is still important, but should support the NLoS law rather than replace it

Source:

- supervisor paper local markdown `2511.15412v1`

Most relevant idea:

- shadow projection is useful for deterministic LoS/NLoS region construction,
- `Eq. (12)` is valuable as an elevation-dependent shadow-severity cue,
- but the literal excess-loss equation alone is not strong enough for our dataset.

Implication:

- use shadow-derived features for regime support,
- but do not expect them to solve NLoS without a stronger mean attenuation law.

### 4. Multi-path regime diversity matters beyond a binary LoS/NLoS split

Source:

- M. Pang et al., ["Geometry-Based Stochastic Probability Models for the LoS and NLoS Paths of A2G Channels under Urban Scenario"](https://arxiv.org/abs/2205.09399)

Most relevant idea:

- ground-specular and building-scattering behavior vary with altitude and geometry,
- so a richer prior should eventually distinguish different NLoS mechanisms.

Implication:

- future prior refinements may need:
  - shallow-shadow / deep-shadow distinction,
  - ground-specular-aware modulation,
  - or building-scatter severity cues.

## Refined next-step plan for the prior

### Stage 1: get the LoS prior right

- implement a more faithful coherent two-ray law,
- keep frequency fixed,
- vary only geometry-dependent quantities:
  - `theta`,
  - path-length difference,
  - reflection behavior,
  - transmitter height.

This stage is about reproducing the concentric/radial structure better.

### Stage 2: get the NLoS mean law right

- use LoS/NLoS segmentation,
- use a height-dependent NLoS PLE family,
- optionally vary coarsely by city morphology,
- keep coefficients low-precision and interpretable.

This stage is about lowering the masked ground-valid `NLoS` RMSE before asking the network to fix everything.

### Stage 3: only then use the network for high-frequency correction

Once the prior is better:

- keep `Try 42`-style `prior + residual`,
- add the light PatchGAN in `Try 48`,
- and let the network recover high-frequency local structure.

## Practical project rule

For the next prior iterations:

- do not launch `Try 48` until the LoS prior family reproduces the concentric/radial structure more faithfully;
- do not assume that a more "physical-looking" formula is automatically better;
- treat prior refinements as accepted only if they beat the current best prior-only RMSE baseline of `23.5746 dB`.

## Current completed result

The first refined `Try 48` prior iteration has now finished with:

- `raw_prior`: `67.9603 dB`
- best calibrated system: `city_type_los_ant_quadratic`
- best validation RMSE: `24.1777 dB`

This is worse than the earlier `Try 47` prior (`23.5746 dB`), so the current refined prior should be considered a rejected candidate rather than an improvement.

- do **not** spend time on overly fine coefficient searches,
- do spend time on:
  - a more faithful coherent two-ray LoS law,
  - a more physically distinct NLoS mean law,
  - and source-backed regime structure.
