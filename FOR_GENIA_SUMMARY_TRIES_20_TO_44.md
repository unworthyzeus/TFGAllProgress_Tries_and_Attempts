# Summary for Genia: Tries 20-44

This note extends the previous summary to include the current PMNet control branches and the first conclusions drawn from them.

## Executive summary

From `Try 20` onward, the project has moved through four clear stages:

1. make the old path-loss baseline cleaner and more stable;
2. add physically motivated supervision and masking fixes;
3. make the physical prior central;
4. test whether the remaining bottleneck is the backbone itself.

The strongest conclusions so far are:

- `Try 22` remained the strongest clean older `path_loss` baseline;
- `Try 26` remained the strongest clean older spread baseline;
- the newer dataset is harder than the old one;
- the physical prior helps, but it is not enough by itself;
- the main remaining failure is concentrated in difficult `NLoS` urban regimes.

## What the recent PMNet family is testing

### Try 42

`Try 42` keeps the calibrated prior from `Try 41` and changes the backbone:

- remove the old U-Net / cGAN family,
- keep `prediction = calibrated_prior + learned_residual`,
- replace the path-loss network with a PMNet-inspired residual regressor.

The goal is to answer:

- if the prior already explains the easy physical carrier, is the remaining bottleneck now the backbone?

### Try 43

`Try 43` is the no-prior PMNet control:

- same masked supervision,
- same LoS / distance / antenna context,
- same regime-level metrics,
- but no physical prior input and no prior-residual formulation.

It predicts `path_loss` directly.

This try exists to separate:

- the contribution of the prior,
- from the contribution of PMNet itself.

### Try 44

`Try 44` is a stricter PMNet control.

It still has no physical prior, but it uses a more faithful PMNet-v3-style architecture:

- bottleneck-style encoder,
- ASPP context aggregation,
- decoder structure closer to the original PMNet repository.

So `Try 44` is the cleaner test of:

- whether the PMNet family itself is a better path-loss architecture for this project,
- before mixing it again with the physical prior.

## Why the first Try 42 result matters

The first received `Try 42` epoch showed:

- global RMSE around `23.19 dB`

which is still far from the project target.

However, the regime breakdown is much more informative than the global number:

- `LoS`: about `4.36 dB`
- `NLoS`: about `40.46 dB`

This is a strong and useful result because it shows:

- the system already handles the easy physical carrier reasonably well;
- the real problem is the residual urban correction in hard `NLoS` cases.

The same pattern also appears by morphology and antenna height:

- `open_lowrise` is much easier than `dense_highrise`;
- high antennas are much easier than low antennas.

So the project is not failing uniformly.
It is failing in a concentrated and physically meaningful subset of regimes.

## Interpretation of the prior issue

There was an important debugging step around the prior metrics.

At one point, a prior-only figure around `24.16 dB` looked much better than the channel that was actually entering the model.
After rechecking the calibration pipeline, the conclusion was:

- the raw prior is indeed very poor on its own (`~67.23 dB`);
- train-only regime-aware calibration improves it strongly;
- but the operational question is not whether the prior is good enough alone;
- it is whether the network can exploit it better than a direct model without prior.

That is exactly why the comparison now focuses on:

- `Try 42`: PMNet + prior
- `Try 44`: PMNet-v3-style without prior

## Practical meaning for supervision

The recent work should be described as:

- physically informed reformulation,
- then architecture control,
- then stricter diagnosis by regime.

It should not be described as random trial-and-error.

The current logic is:

1. verify whether the prior helps;
2. verify whether PMNet helps without the prior;
3. only then decide whether a future `Try 45` should combine the more faithful PMNet branch with the prior again.

## Current decision logic for the next branch

The next sensible question is not automatically "add the prior again".

The cleaner decision rule is:

- if `Try 44` does better than `Try 43`, then the more faithful PMNet backbone is probably worth keeping;
- if `Try 44` also shows a useful gain over difficult regimes, then a future `Try 45 = Try 44 + calibrated prior` is justified;
- if `Try 44` does not improve, then the PMNet family itself may not be the right next direction for this dataset.

So `Try 44` is a necessary control experiment, not an unnecessary detour.
