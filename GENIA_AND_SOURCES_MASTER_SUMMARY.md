# Master Summary for Genia + Sources

This file is meant to be the main long-form reference for the recent experimental phase.

It merges four things that had become too scattered across the repository:

- a supervisor-facing explanation of what the recent tries are trying to solve,
- the concrete experimental lessons from the strongest branches,
- the paper/source motivations behind those choices,
- and the current decision logic for what should happen next.

Older notes are intentionally preserved for traceability, but this file should be treated as the clearest current narrative.

## How to read this file

If the goal is quick orientation:

1. read the executive summary,
2. read the "What actually changed in the metric" section,
3. read the sections on `Try 22`, `Try 42`, and `Try 47`,
4. then use the source sections only if more justification is needed.

If the goal is research discussion with a supervisor:

1. read the executive summary,
2. read the family transitions,
3. read the source mapping,
4. read the current release logic and next-decision rules.

## Executive summary

The project is no longer bottlenecked by generic decoder tuning alone.

The main unresolved difficulty is now clearly:

- masked, ground-valid `NLoS` path-loss prediction

where:

- buildings are excluded from the metric,
- the easier `LoS` regime is already relatively strong,
- and old historical `NLoS` results were partially inflated by a less strict evaluation definition.

The main recent experimental lessons are:

- `Try 22` was the strongest clean U-Net path-loss baseline of the older family.
- `Try 42` showed that `calibrated prior + learned residual` is a real improvement, especially for `NLoS`.
- PMNet alone was not convincing enough as a replacement for the stronger U-Net spatial behavior.
- `Try 46` showed that explicit `LoS / NLoS` specialization is a reasonable direction, but not enough when combined with the PMNet trunk that underperformed in this project.
- `Try 47` is therefore the current synthesis:
  - return to the stronger `Try 22` U-Net family,
  - keep the calibrated prior from the `Try 42` line,
  - keep explicit `LoS / NLoS` specialization,
  - add `NLoS` experts and obstruction-oriented inputs.

The project has therefore moved from:

- "make the image-to-image pipeline work"

to:

- "use physics for the easy large-scale field, and use learning mainly for the hard correction regimes."

## Current prior lesson for Try 48

The latest prior work clarified a useful point:

- the next gain is not expected to come mainly from more precise coefficient fitting,
- it is expected to come from a better **physical prior family**.

The reason is that the project uses:

- fixed carrier frequency,
- but variable UAV height and therefore variable elevation angle and path geometry.

That means the prior should depend mainly on:

- direct/reflected path-length difference,
- elevation angle,
- reflection behavior,
- LoS/NLoS segmentation,
- and NLoS obstruction regime.

The practical consequence is:

- the old smooth breakpoint-style two-ray prior is too weak to explain the fine concentric LoS structure,
- and the current NLoS prior is still too weak structurally even before learning.

The latest completed `Try 48` prior-only refinement confirms this point:

- refined Fresnel/coherent-hybrid prior: `24.1777 dB`
- previous obstruction-aware prior baseline: `23.5746 dB`

So the first attempt at a more coherent LoS prior was not enough. The project still needs a better LoS prior family before launching the PatchGAN-based `Try 48`.

The currently active `Try 48` training run therefore uses a pragmatic fallback:

- keep the stronger prior from the `Try 47` calibration family,
- remove `MoE`,
- keep the light PatchGAN idea,
- and continue prior research separately until a better concentric LoS prior is found.

So for `Try 48`, the recommended logic is:

1. improve the LoS prior family,
2. improve the NLoS mean law,
3. recalibrate train-only,
4. then let the `Try 42`-style model and light PatchGAN handle the residual high-frequency correction.

## The most important correction: what changed in the metric

One of the biggest sources of confusion in the project was not architectural, but evaluative.

Older strong numbers, especially for `NLoS`, were not fully comparable to the newer masked setup.

The modern rule is:

- only pixels with `topology == 0` and a valid dataset mask count in the official error

That means:

- building pixels are excluded from training supervision,
- excluded from validation/test metrics,
- and excluded from visual error maps.

This is a stricter and more meaningful definition, because those building pixels are not valid ground receiver locations.

The key consequence is:

- old `NLoS` values looked better partly because they were evaluated on a more favorable set of pixels.

So when recent tries seem "worse", that does not always mean the models are intrinsically worse.
It often means the evaluation is more honest.

## Why the old `Try 14 NLoS` result was misleading

Historically, a result around `23 dB` in `Try 14 NLoS` looked like strong evidence that the project had once solved `NLoS` much better.

That turned out to be misleading for two reasons:

1. the old `los_only` / `nlos_only` splits were sample-dominant splits, not pure pixel-level regime splits;
2. building pixels were not excluded with the modern strict mask.

After reevaluation:

- `Try 14 NLoS` under the legacy definition:
  - overall about `23.08 dB`
  - `NLoS` about `26.62 dB`
- `Try 14 NLoS` under the modern ground-only mask:
  - overall about `28.81 dB`
  - `NLoS` about `37.46 dB`

So the apparent old `NLoS` strength does not survive the stricter definition.

This was not "malicious" or fraudulent. It was simply a looser experimental definition that became problematic only after the project started asking regime-level questions more precisely.

## Family transitions: what the project learned over time

## Tries 20-22: the strongest clean U-Net path-loss family

`Try 20` and `Try 21` were useful because they isolated two different hypotheses.

### Try 20

`Try 20` tested the decoder hypothesis:

- use bilinear upsampling instead of transposed-convolution-heavy decoding

Main motivation:

- reduce checkerboard artifacts,
- improve spatial smoothness,
- stabilize image reconstruction.

### Try 21

`Try 21` tested the supervision hypothesis:

- supervise path loss at multiple scales, not only at the full-resolution map

Main motivation:

- preserve global field structure,
- reduce the tendency to get local texture right while missing the larger field shape.

### Try 22

`Try 22` combined the two previous ideas and became the strongest clean path-loss baseline of this family.

What it means:

- a cleaner decoder,
- plus supervision that explicitly checks the field at several spatial scales.

What worked:

- this became the best recent clean `path_loss` branch before the prior-based line;
- it established the baseline that later path-loss tries had to beat;
- it remained especially strong in `LoS`.

What did not solve the whole problem:

- even this model still underlearned the radial propagation pattern around the transmitter;
- and under the modern building mask, its `NLoS` error remains much worse than its global RMSE suggests.

Modern masked reevaluation gives approximately:

- overall RMSE: `19.94 dB`
- `LoS`: `3.78 dB`
- `NLoS`: `34.43 dB`

This is still a very important baseline because it tells us the U-Net family was not the main weakness in `LoS`; the real weakness is the hard masked `NLoS` regime.

## Tries 23-30: attempts to refine path loss and spreads within the same broad family

This group of tries did not change the basic paradigm as much as later branches, but it was still useful.

Main roles:

- adapt the structural improvements of `Try 22` to spread targets,
- test whether more global context was missing,
- test topology-aware weighting,
- test more physically targeted losses.

These branches clarified an important negative result:

- adding complexity on top of the same broad decoder/backbone family was not enough to solve the real remaining `NLoS` bottleneck.

That is why later branches stopped focusing mainly on decoder tuning and started changing the problem formulation itself.

## Tries 31-36: paradigm shift toward priors, residuals, support maps, and stricter masking

These tries introduced several important ideas:

- `path_loss = prior + residual`
- spread prediction as support plus amplitude
- building-mask exclusion during training and evaluation

Conceptually, this was a major improvement because it recognized:

- the network should not need to rediscover the full large-scale propagation law from scratch,
- and building pixels should not contribute to the metric or the loss.

These tries did not yet solve the problem, but they changed the project in the correct direction.

## Tries 37-40: same masked family on the harder new dataset

When the project moved to `CKM_Dataset_270326.h5`, the task became harder in practice.

This mattered because:

- more cities were present,
- morphology was more diverse,
- and the priors/backbones that looked acceptable before no longer generalized as comfortably.

The main lesson of this family was:

- dataset difficulty and evaluation strictness matter as much as network size.

## Try 41: physical prior + learned residual becomes central

`Try 41` was the first branch to put the new main idea at the center:

- `prediction = calibrated_physical_prior + learned_residual`

This changed the role of learning.

Instead of learning:

- the whole path-loss field from topology and geometry,

the network now mainly learns:

- how reality deviates from a structured prior.

This was a very important step because it made the project explicitly physics-guided rather than only physics-conditioned.

## Try 42: PMNet-style residual model over the calibrated prior

`Try 42` kept the calibrated prior idea but changed the backbone.

The intention was:

- use a more modern, stronger context model,
- keep the prior anchor,
- and let the network learn a residual on top of it.

Best observed result:

- overall RMSE about `19.17 dB`
- `LoS` about `3.84 dB`
- `NLoS` about `33.40 dB`

This matters because it means:

- `Try 42` slightly improves over masked `Try 22` overall,
- and improves over `Try 22` in `NLoS`,
- even though it is not as elegant spatially as the older U-Net family.

In other words:

- the prior is helping,
- but PMNet is not obviously the ideal backbone for this project.

## Try 43 and Try 44: PMNet controls without the prior

These controls mattered because they answered an important research question:

- is the improvement coming from the prior,
- or simply from changing the backbone?

The answer was:

- PMNet without the prior was not convincing enough.

So the correct lesson was not:

- "PMNet is the answer"

but rather:

- "the prior is doing real work, and a good spatial backbone still matters."

## Try 46: explicit `LoS / NLoS` specialization

`Try 46` tried to address the obvious remaining issue directly:

- `LoS` was already comparatively good,
- `NLoS` was still bad,
- so the network should specialize.

That led to:

- explicit `LoS / NLoS` branching,
- a dedicated `NLoS` MoE branch,
- and regime-specific metrics.

This idea was correct in spirit.

The main problem was that the PMNet-style trunk still did not behave as convincingly as the older spatial U-Net family.

So the project learned:

- regime specialization is worth keeping,
- but PMNet should not automatically remain the trunk.

## Try 47: current synthesis

`Try 47` is the current active synthesis because it combines the strongest parts of three earlier lines:

### From Try 22

- bilinear U-Net decoder
- group normalization
- FiLM with antenna height
- distance-map channel
- multiscale path-loss supervision

### From Try 42

- calibrated prior
- explicit residual learning
- the idea that physics should do the coarse large-scale work first

### From Try 46

- explicit `LoS / NLoS` specialization
- `NLoS`-only experts
- regime-aware losses

### New additions in Try 47

- obstruction proxy channels:
  - shadow depth
  - distance since LoS break
  - maximum blocker height
  - blocker count
- stronger combination losses for hard `NLoS` subsets
- separate small `LoS` residual head

The central research hypothesis is:

- perhaps the project did not need to abandon the `Try 22` spatial family,
- it only needed to add the newer physical-prior and regime-specialization ideas on top of it.

## Why `LoS` is no longer the main problem

One of the clearest cross-try findings is that `LoS` is already relatively controlled.

Examples:

- `Try 22` masked `LoS`: about `3.78 dB`
- `Try 42` `LoS`: about `3.84 dB`
- some prior-only analyses already reach values in that same ballpark

That means the project target is not blocked primarily by `LoS`.

The real challenge is:

- masked ground-valid `NLoS`
- especially in harder city morphologies
- especially with lower antennas

That is why the project increasingly moved toward:

- better priors,
- better regime decomposition,
- better `NLoS` specialization,
- and better regime diagnostics.

## Why the prior still matters

The prior is still central for three reasons.

### 1. It already solves much of `LoS`

The prior gives the model a structured large-scale field from the beginning.

That means the network does not need to waste capacity rediscovering:

- distance decay,
- radial field structure,
- and the basic low-complexity propagation law.

### 2. It improved the project more than PMNet without prior

The PMNet controls without prior did not show enough benefit.

So the important distinction is not simply:

- U-Net vs PMNet

but:

- no prior vs calibrated prior + residual.

### 3. It makes `NLoS` learning more targeted

If the prior already solves the easy large-scale component, then the `NLoS` expert only needs to learn:

- obstruction effects,
- shadow severity,
- regime-specific excess loss.

That is a much more plausible job than asking the same branch to reconstruct the whole propagation field from scratch.

## What sources motivated the recent direction

## 1. Bilinear decoding and multiscale supervision

Main references:

- Augustus Odena, Vincent Dumoulin, Chris Olah, "Deconvolution and Checkerboard Artifacts", Distill, 2016  
  [https://distill.pub/2016/deconv-checkerboard/](https://distill.pub/2016/deconv-checkerboard/)
- Mathieu et al., "Deep Multi-Scale Video Prediction Beyond Mean Square Error", 2016  
  [https://arxiv.org/abs/1511.05440](https://arxiv.org/abs/1511.05440)

Project use:

- support the move away from transposed-convolution-heavy decoders,
- justify multiscale path-loss supervision,
- and remain part of the `Try 47` base through the `Try 22` architectural inheritance.

## 2. Radio-map backbones

Relevant references:

- RadioUNet  
  [https://arxiv.org/abs/1911.09002](https://arxiv.org/abs/1911.09002)
- RadioNet  
  [https://arxiv.org/abs/2105.07158](https://arxiv.org/abs/2105.07158)
- PMNet repository  
  [https://github.com/abman23/pmnet](https://github.com/abman23/pmnet)

Project use:

- motivate stronger context-aware image-to-image backbones,
- justify the PMNet family tries,
- and provide negative evidence when the PMNet controls without prior were not strong enough.

That negative evidence is itself useful, because it helps justify the decision to return to the `Try 22` U-Net family for `Try 47`.

## 3. A2G LoS/NLoS physics from the supervisor paper line

Main source family:

- "Spatially Consistent Air-to-Ground Channel Modeling and Simulation via 3D Shadow Projections"
- and the related LoS/NLoS segmentation paper line

Project interpretation:

- `Eq. (1)` gives the large-scale decomposition view
- `Eq. (9)` gives the reference/base term
- `Eq. (10)` gives the `LoS` behavior
- `Eq. (11)` gives the `NLoS` excess-loss behavior
- `Eq. (12)` motivates shadow-variability-style features

Important nuance:

- the project does not copy the full paper as a stochastic simulator;
- it extracts the deterministic structural parts that are useful for building a practical prior.

This is why the prior family is described as:

- hybrid
- calibrated
- and train-only fitted

rather than:

- exact closed-form simulator output.

## 4. Mixture-of-experts for heterogeneous regimes

Main references:

- MLoRE: "Multi-Task Dense Prediction via Mixture of Low-Rank Experts"  
  [https://arxiv.org/abs/2403.17749](https://arxiv.org/abs/2403.17749)
- "Heterogeneous Mixture of Experts for Remote Sensing Image Super-Resolution"  
  [https://arxiv.org/abs/2502.09654](https://arxiv.org/abs/2502.09654)

Project use:

- not to build a giant transformer-style MoE,
- but to justify a small expert system exactly where heterogeneity is highest,
- namely the masked `NLoS` correction branch.

This is why the project direction became:

- keep one simpler path for the already-structured `LoS` case,
- but allow multiple experts for the much more heterogeneous `NLoS` case.

## Why the calibration JSON is part of the experiment

One important experimental point is that the calibration is not just a side artifact.

The JSON used for the prior is part of the actual experiment definition.

That means:

- the training run depends on a particular calibration file,
- that file must be stored inside the try folder,
- and it must be reproducible from training-only fitting.

This is especially important for generalization claims:

- the prior should not be tuned using validation or test information,
- and a new dataset should trigger a new train-only calibration refresh.

That is why the cluster workflow for `Try 47` runs:

1. calibration job
2. training job with dependency

instead of assuming that an older calibration remains valid forever.

## Current project logic

The current logic is no longer:

- "make the model larger"

or:

- "swap backbones until something works"

It is now:

1. keep the strict ground-only metric
2. keep the calibrated prior
3. keep residual learning
4. keep regime diagnostics
5. treat `NLoS` as the main unresolved problem
6. combine the stronger older spatial backbone with the newer prior/regime logic

That is exactly why `Try 47` exists.

## What success for Try 47 would mean

`Try 47` does not need to magically solve the whole project in one step to be valuable.

A meaningful success would be:

- match or improve the strong `LoS` behavior of `Try 22` / `Try 42`
- and improve masked `NLoS` beyond the current prior-aware PMNet line

If that happens, the project will have strong evidence that:

- the prior was worth keeping,
- the PMNet trunk was not the real long-term answer,
- and the `Try 22` family was the right spatial base after all.

## Current bottom line

The current strongest reading is:

- the project was right to move toward prior-guided residual learning;
- it was right to make the evaluation stricter by excluding buildings;
- it was right to add regime-level diagnostics;
- and it is now reasonable to test whether the old strong U-Net family, when combined with prior and `NLoS` specialization, outperforms the PMNet branch.

That is the role of `Try 47`.
