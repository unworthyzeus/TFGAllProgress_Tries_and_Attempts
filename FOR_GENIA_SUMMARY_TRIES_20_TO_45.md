# Supervisor Summary: Tries 20 to 46

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

## What Tries 41-46 are testing

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

## What the current numbers say

The current message from the data is already fairly clear.

### Try 42

`Try 42` shows that:

- the calibrated prior is useful,
- PMNet can improve over that prior,
- but the hard part is still `NLoS`.

Reference validation numbers:

- overall RMSE: about `19.78 dB`
- prior-only RMSE: about `24.16 dB`
- `LoS`: about `3.86 dB`
- `NLoS`: about `34.47 dB`

So `Try 42` is not solving the project, but it does show that the prior is helping.

### Try 44

`Try 44` is useful because it shows what happens when PMNet is made more faithful to the original repository structure but the prior is removed.

Reference validation numbers:

- overall RMSE: about `22.45 dB`
- `LoS`: about `4.74 dB`
- `NLoS`: about `39.06 dB`

So the more faithful PMNet control does **not** beat the prior-based branch.

### Practical interpretation

This means the current bottleneck is not simply:

- "U-Net vs PMNet"

The deeper issue is:

- the prior is still too weak in `NLoS`,
- and that weakness propagates into the learned residual problem.

That is why `Try 45` is currently treated as a gated branch rather than the next automatic cluster run.

## Current release rule for Try 45

`Try 45` should only be launched when the **prior-only** validation result reaches:

- `NLoS RMSE < 20 dB`

Until that happens, the work should stay focused on improving the prior itself.

Current best prepared `Try 45` prior-only calibration:

- overall RMSE: about `23.57 dB`
- `LoS`: about `3.81 dB`
- `NLoS`: about `41.27 dB`

### Try 46

`Try 46` is the first branch that changes the network structure around the regime split itself:

- one lightweight `LoS` residual head
- one stronger `NLoS`-only MoE residual head
- one shared PMNet-style trunk
- and extra `NLoS` diagnostics split by shadow-depth proxy

The idea is that `LoS` and `NLoS` should no longer be forced through one shared residual head.

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
