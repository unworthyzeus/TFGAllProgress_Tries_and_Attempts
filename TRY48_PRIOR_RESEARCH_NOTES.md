# Try 48 Prior Research Notes

## Why the current prior misses the concentric pattern

The current `two_ray_ground` implementation used in the recent tries is **not** a coherent two-ray interference model. In code, it behaves as:

- `FSPL` below the crossover distance,
- the asymptotic `40 log10(d) - 20 log10(h_tx) - 20 log10(h_rx)` law above crossover.

That model captures the **large-scale slope change** of two-ray propagation, but it does **not** model the direct-plus-reflected phase interaction that produces alternating constructive and destructive rings. Therefore, the prior can reproduce the broad radial trend, but it cannot reproduce the fine concentric oscillations observed in the ground-truth path-loss maps.

This explains why:

- LoS RMSE can already be reasonably low with the current prior,
- but the prior still looks too smooth,
- and the network then has to invent the high-frequency radial structure by itself.

## Source-backed direction for LoS prior

### 1. Replace the current asymptotic two-ray with a coherent elevation-angle-aware two-ray prior

The paper below is the strongest source for this part:

- N. H. Ranchagoda, K. Sithamparanathan, M. Ding, A. Al-Hourani, and K. M. Gomez, ["Elevation-angle based two-ray path loss model for Air-to-Ground wireless channels"](https://www.sciencedirect.com/science/article/pii/S2214209621000620), *Vehicular Communications*, 2021.

What matters for us:

- it explicitly treats A2G path loss as a **direct + ground-reflected** problem,
- it highlights that destructive-interference down-fades appear at specific elevation angles,
- it argues that a modified two-ray model with reflection coefficient is a better A2G mean model than a plain asymptotic transition.

Practical implication for `Try 48`:

- the LoS prior should be upgraded from a smooth crossover model to a **coherent two-ray mean-power model**,
- using path-length difference and a reflection coefficient,
- optionally with damping/clipping so that the prior remains numerically stable.

This should be the only realistic way for the prior itself to start reproducing at least part of the concentric ring structure.

## Source-backed direction for NLoS prior

The recent experiments show that NLoS remains the bottleneck even when LoS is already quite good. The current obstruction-aware train-only calibration reduced the overall prior RMSE, but the NLoS component remained far from acceptable. That suggests that the current NLoS prior is structurally too weak, not just miscalibrated.

### 2. Use Vinogradov-style probabilistic LoS/NLoS segmentation as the backbone of the NLoS prior

Primary source:

- E. Vinogradov, A. Saboor, Z. Cui, and A. Fakhreddine, ["Spatially Consistent Air-to-Ground Channel Modeling with Probabilistic LOS/NLOS Segmentation"](https://arxiv.org/abs/2506.12794), VTC-Spring 2025 workshop paper.

Key idea:

- the path loss should be parameterized through **LoS/NLoS segmentation**, not only through a single blended path-loss curve,
- and the segmentation can depend on environment-aware geometry while remaining much cheaper than full ray tracing.

Why this matters for us:

- our current `los_mask` is already a very strong signal,
- but the prior still treats NLoS mostly as a scalar correction over a LoS-oriented base formula,
- whereas the Vinogradov line of work suggests that the model should first decide whether the point belongs to a LoS or NLoS regime and only then apply the corresponding loss law.

### 3. Use height-dependent NLoS path-loss exponents instead of a single excess-loss formula

Primary source:

- A. Saboor and E. Vinogradov, local `TFG_Proto1` markdown `2511.10763v1`, "Millimeter-Wave UAV Channel Model with Height-Dependent Path Loss and Shadowing in Urban Scenarios".

This paper is especially relevant because it gives a compact, physically structured NLoS model:

- the A2G mean attenuation is written as a LoS/NLoS weighted combination,
- LoS and NLoS have different path-loss exponents,
- the NLoS exponent becomes **height-dependent**,
- and shadow-fading standard deviation is also **height-dependent**.

The core equations from this paper are useful for a prior even without mmWave-specific details:

- `P_LoS(theta)` as a sigmoid,
- `Lambda(d,h) = P_LoS * Lambda_LoS + (1-P_LoS) * Lambda_NLoS`,
- `Lambda_x = Lambda_0 + 10 n_x(h) log10(d/d0) + Psi_x`,
- `n_x(h)` and `sigma_x(h)` modeled with exponential height dependence.

Why this is more promising than the current NLoS prior:

- it makes the NLoS branch structurally different from LoS,
- it gives the NLoS slope freedom to change with altitude,
- and it matches the empirical observation that NLoS becomes less harsh at higher ABS heights.

### 4. Use the supervisor's segmentation/shadow paper as a geometry prior, not just a noise term

Primary source:

- ["Spatially Consistent Air-to-Ground Channel Modeling and Simulation via 3D Shadow Projections"](https://arxiv.org/abs/2511.15412) and local markdown `2511.15412v1`.

What the paper gives us cleanly:

- deterministic LoS/NLoS segmentation through shadow projection,
- `Eq. (10)` for LoS excess loss,
- `Eq. (11)` for NLoS excess loss,
- `Eq. (12)` for elevation-dependent shadow-fading standard deviation.

What we learned from our own calibration attempts:

- the plain `Eq. (10)/(11)`-style NLoS excess-loss term is still too weak on our dataset,
- but `Eq. (12)` is still useful as a **shadow severity feature** or uncertainty prior,
- and the shadow-projection/segmentation viewpoint is likely more valuable than the literal NLoS excess-loss constants.

## Additional source that supports a richer NLoS prior

- M. Pang et al., ["Geometry-Based Stochastic Probability Models for the LoS and NLoS Paths of A2G Channels under Urban Scenario"](https://arxiv.org/abs/2205.09399), *IEEE Internet of Things Journal*, 2023.

This paper does not directly provide the final path-loss law we want, but it does support a useful idea:

- NLoS should not be treated as a single monolithic mode,
- because the occurrence of LoS, ground-specular, and building-scattering paths depends on geometry, altitude, and Fresnel-zone behavior.

For our project this supports using extra obstruction-aware inputs such as:

- shadow depth,
- distance since LoS break,
- blocker count,
- max blocker height,

## Current negative result

The latest `Try 48` prior-only calibration with the refined Fresnel/coherent two-ray LoS component and the hybrid `COST231 + A2G NLoS` branch has now completed.

Final validation result:

- best train-only calibrated system: `city_type_los_ant_quadratic`
- validation RMSE: `24.1777 dB`

This is worse than the previous obstruction-aware prior used for the `Try 47` family:

- previous best prior-only RMSE: `23.5746 dB`

So this specific refinement should be treated as useful negative evidence:

- changing the LoS prior toward a more coherent/Fresnel-like form is not sufficient by itself,
- the current implementation still does not reproduce the concentric structure well enough,
- and this exact prior variant should not be used as the default prior for the next training run.

The practical lesson is that the project still needs a better physically structured LoS prior before launching `Try 48`.

## Current Try 48 training configuration

The currently launched `Try 48` does **not** use the rejected structural-prior candidate above.

Instead, the active training configuration uses:

- the exact train-only calibrated prior exported from the successful `Try 47` prior family,
- embedded locally inside `Try 48` as:
  - `prior_calibration/regime_obstruction_train_only_from_try47.json`
- the `Try 48` light PatchGAN training setup,
- and no `MoE`.

This is intentional:

- the structural prior remains a research branch,
- but the training run should use the strongest available prior baseline until a better concentric/ring-aware LoS prior is found.

## Current data-pipeline change for training stability

The latest `Try 48` update also changes how the prior is fed into the model.

Instead of:

- computing the prior on the fly for every sample,
- or caching thousands of per-sample `.pt` files,

the new setup writes the calibrated prior input into:

- one single HDF5 file under `Try 48/precomputed/`

and the dataset loader reads the formula channel directly from that HDF5.

This is meant to reduce:

- irregular training throughput,
- file-system overhead,
- and fragile checkpoint/cache behavior during cluster runs.
- and possibly future ground-specular indicators.

## Recommended prior upgrade for Try 48

`Try 48` should keep the **generator-side model idea** simple:

- `Try 42` backbone,
- calibrated prior input,
- residual learning,
- light PatchGAN for high-frequency correction.

But the **prior itself** should be upgraded in two different ways for LoS and NLoS.

### Proposed LoS prior

- Replace the current asymptotic `two_ray_ground` with a **coherent A2G two-ray prior**.
- Use direct-plus-reflected mean power with elevation-angle dependence.
- Keep it clipped/stabilized to avoid pathological oscillations.
- Keep the existing train-only calibration layer on top.

### Proposed NLoS prior

- Stop treating NLoS as only a scalar correction over the same LoS-shaped prior.
- Build the NLoS mean prior from:
  - LoS/NLoS segmentation,
  - a height-dependent NLoS PLE model,
  - optional environment grouping (`open_lowrise`, `mixed_midrise`, `dense_highrise`),
  - obstruction-derived features already available from the newer tries.
- Use `Eq. (12)`-style shadow-sigma only as an additional severity feature or uncertainty proxy, not as the sole NLoS correction mechanism.

## Concrete implementation order

1. Upgrade the LoS formula input from asymptotic two-ray to coherent two-ray.
2. Replace the current NLoS mean term with a height-dependent log-distance NLoS model.
3. Re-run train-only calibration over the new prior family.
4. Use that improved prior in `Try 48` with the `Try 42 + PatchGAN` generator/discriminator setup.

## Implemented structural prior for Try 48

The current `Try 48` codebase has been updated to use a new structural prior family:

- formula mode: `hybrid_coherent_two_ray_vinogradov_nlos`
- no old `Try 47` regime-calibration JSON is reused by default
- the first calibration pass is intended to be re-generated on top of this new prior family

### Implemented LoS prior

- a coherent two-ray mean-power model is used instead of the old smooth asymptotic crossover
- the reflected path uses a coarse elevation-dependent reflection magnitude
- the final LoS prior is lightly stabilized by blending it with direct-path FSPL

### Implemented NLoS prior

- a height-dependent log-distance NLoS model is used
- the path-loss exponent is chosen coarsely by urban morphology class:
  - `open_lowrise`
  - `mixed_midrise`
  - `dense_highrise`
- the coefficients are intentionally low-precision:
  - `open_lowrise`: `n0=4.3`, `n_inf=2.8`, `h0=25`
  - `mixed_midrise`: `n0=4.7`, `n_inf=2.8`, `h0=40`
  - `dense_highrise`: `n0=4.7`, `n_inf=2.8`, `h0=120`
- an `Eq. (12)`-style elevation-dependent shadow-sigma term is used only as a deterministic severity proxy, not as a random fading draw

This is intentional: the expected gain should come from the **model structure**, not from over-tuning coefficient precision.

## Calibration workflow

The updated prior should be calibrated again using train-only data. The local calibration workflow now supports:

- `directml`
- visible progress logs (`--log-every`)
- writing outputs under `Try 48/prior_calibration`

This calibration is now supposed to produce a **new** calibration JSON that is consistent with the structural prior above, instead of reusing the old `Try 47` calibration file.

## Important conclusion

The main issue is not only that the network underlearns the concentric pattern. The prior itself currently removes most of that structure before learning even starts. Therefore, improving `Try 48` should focus first on:

- a better LoS prior for concentric/radial structure,
- a more physically distinct NLoS prior,
- and only then the adversarial high-frequency refinement.
