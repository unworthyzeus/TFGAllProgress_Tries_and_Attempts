# Summary for Genia: Tries 20-42

This note summarizes the recent experimental evolution from `Try 20` to `Try 42`, with emphasis on what each family was meant to test, what actually worked, and why the project has now moved away from the old U-Net/cGAN line for path-loss prediction.

## Executive summary

The project progressed in three major stages:

1. clean up and stabilize the older image-to-image baseline;
2. inject stronger physical structure into path-loss prediction;
3. replace the old path-loss backbone with a more suitable residual architecture.

The strongest stable conclusions before the latest architecture change were:

- `Try 22` was the strongest clean `path_loss` baseline in the older dataset family;
- `Try 26` was the strongest clean `delay_spread + angular_spread` baseline in the older dataset family.

However, these were still far from the target level of performance, especially for `path_loss`.

That is why the project moved toward:

- explicit physically calibrated priors,
- residual learning over those priors,
- and finally a backbone change in `Try 42`.

## Stage 1: make the old baseline cleaner and more interpretable

### Tries 20-22

These tries asked a simple question:

- were the remaining `path_loss` errors mostly a decoder/coarse-structure problem?

What changed:

- `Try 20`: bilinear decoder instead of transposed convolution;
- `Try 21`: multiscale path-loss supervision;
- `Try 22`: combine both.

What was learned:

- the combination mattered more than either piece in isolation;
- `Try 22` became the best clean path-loss baseline of that family;
- but it still plateaued far above the project target.

### Tries 23-30

These tries explored more targeted corrections after visual inspection:

- extend the cleaner recipe to spread outputs;
- add bottleneck attention;
- add edge-aware or hotspot-aware losses;
- add radial losses for path-loss.

What was learned:

- these ideas were useful diagnostically;
- some improved early behavior or local structure;
- but none of them fundamentally changed the overall plateau.

The lesson was:

- the bottleneck was no longer a small decoder tweak or one more auxiliary loss.

## Stage 2: change the formulation, not just the loss

### Tries 31-32

These were the first tries that changed the formulation itself.

- `Try 31` moved to:
  - `path_loss = physical_prior + learned_residual`
- `Try 32` decomposed spread prediction into:
  - support
  - amplitude

What they meant:

- stop predicting everything from scratch if part of the structure is already known physically;
- separate subproblems that should not be forced into one single output map.

What was learned:

- these were conceptually stronger than the earlier tweaks;
- but they still did not become clear winners in the existing architecture family.

## Stage 3: clean masking and new dataset

### Tries 33-40

At this point, a more basic modeling issue became important:

- pixels where `topology != 0` should not be treated as valid receiver points.

So the next family enforced that building pixels are excluded from:

- training loss,
- metric computation,
- exported error visualizations.

The newer dataset branch then recreated masked versions of earlier ideas on:

- `CKM_Dataset_270326.h5`

What was learned:

- the new dataset appears significantly harder;
- the clean masked rerun of the old path-loss recipe dropped much more than expected;
- the branch using an explicit physical formula input was more resilient than the branch learning mostly from topology and distance alone.

This pushed the project toward making the physical prior central rather than optional.

## Try 41: make the prior the main scaffold

`Try 41` formalized the following statement:

- `prediction = calibrated_physical_prior + learned_residual`

What changed:

- the hybrid `two_ray_ground + COST231` path-loss map became an explicit input;
- the same map became the additive baseline for the prediction;
- a train-only regime-aware quadratic calibration was added on top of the raw prior.

That calibration is split by:

- city type,
- LoS / NLoS,
- antenna-height tertile.

What was learned:

- the prior itself is genuinely useful;
- train-only regime-aware calibration reduced prior-only RMSE dramatically compared to the raw formula;
- but the old U-Net-based family still improved only modestly beyond that calibrated prior.

So `Try 41` taught an important lesson:

- the next bottleneck is not only the prior;
- it is also the backbone that has to learn the residual.

## Try 42: replace the path-loss backbone

`Try 42` is the first recent path-loss branch that truly leaves the old family behind.

What changes:

- keep the calibrated prior from `Try 41`;
- keep residual learning;
- remove the discriminator path;
- stop using the U-Net backbone;
- replace it with a PMNet-inspired residual regressor.

The new architecture uses:

- a residual encoder;
- a multi-branch dilated context module;
- top-down feature fusion;
- a direct regression head for the residual map.

So `Try 42` is not another small variant.

It is the first attempt to answer this question directly:

- if the prior is already strong, is the remaining failure mainly a backbone limitation?

## Why the current direction is reasonable

The current direction is scientifically defensible because it is based on three consistent observations:

1. the old baseline family can be improved, but only up to a point;
2. the physical prior helps significantly, especially on the harder newer dataset;
3. once the prior is reasonably calibrated, the remaining challenge is how to model the residual corrections.

That makes `Try 42` a natural next step rather than a random architecture change.

## Practical interpretation for supervision

The recent work should not be described as random trial-and-error.

It is better described as:

- baseline stabilization,
- failure-mode diagnosis,
- physically informed reformulation,
- and then architecture replacement once the formulation itself was clearer.
