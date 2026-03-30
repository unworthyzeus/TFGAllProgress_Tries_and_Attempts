# Supervisor Summary: Tries 20 to 45

This summary keeps the recent path-loss story compact and supervisor-friendly.

## What worked best before PMNet

The strongest clean path-loss baseline before the PMNet family was:

- `Try 22`

Its value was not novelty for novelty's sake. It simply combined the two changes that consistently helped:

- bilinear decoder/fusion instead of artifact-prone transposed-convolution-heavy decoding
- multiscale path-loss supervision

That branch remained the best clean baseline for a long time.

## Why the project moved beyond Try 22

Even though `Try 22` was the strongest baseline, it was still far from the target error level.

Visual review and regime metrics showed:

- the model was already relatively good in easier `LoS` conditions
- the largest remaining error was concentrated in harder `NLoS` regimes
- denser city types and lower antenna heights were especially difficult

That is why later branches focused less on generic tuning and more on:

- physical priors
- residual learning
- regime-aware diagnostics

## What Tries 41-45 are testing

### Try 41

`Try 41` introduced the main formulation shift:

- `prediction = physical_prior + learned_residual`

This was the first serious attempt to stop asking the network to rediscover the entire large-scale propagation law from scratch.

### Try 42

`Try 42` kept the calibrated prior idea from `Try 41` but replaced the old U-Net/cGAN family with a PMNet-inspired residual regressor.

This branch was important because it showed:

- the prior was still helping
- but the dominant remaining issue was still `NLoS`

### Try 43

`Try 43` was the PMNet control branch without prior.

It existed to answer a simple question:

- is PMNet strong enough by itself?

### Try 44

`Try 44` was a more faithful PMNet-v3-style control branch, still without prior.

It was intended to test whether the previous PMNet branch was underperforming because the implementation was too simplified.

### Try 45

`Try 45` builds on `Try 42`, not on the no-prior controls.

It keeps:

- PMNet-style residual learning
- the calibrated-prior formulation
- regime-level metrics

and adds:

- a stronger train-only prior calibration designed to be more informative in `NLoS`
- a lightweight spatial mixture-of-experts residual head

The idea is that:

- the prior should explain more of the easy-to-model large-scale structure
- and the residual branch should specialize where the data remain hardest

## Main lesson so far

The recent evidence does **not** suggest that simply replacing U-Net with PMNet solves the project.

The stronger lesson is:

- the physical prior matters
- direct no-prior PMNet branches are not convincing enough
- the remaining challenge is regime-specific, especially `NLoS`

So the next serious experiments are the ones that combine:

- stronger priors
- residual learning
- regime-aware specialization

rather than generic architecture inflation.
