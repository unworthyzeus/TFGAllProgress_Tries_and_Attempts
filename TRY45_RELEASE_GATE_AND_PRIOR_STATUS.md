# Try 45 Release Gate and Prior Status

This note records the current release rule for `Try 45` and the reason behind it.

## Release rule

`Try 45` must **not** be launched to the cluster until the prior-only validation result satisfies:

- `NLoS RMSE < 20 dB`

The metric must be computed:

- on the validation split only,
- with train-only fitted calibration,
- and only on valid ground pixels:
  - `topology == 0`
  - plus the dataset validity mask.

## Why this gate exists

`Try 45` is not just another architecture tweak. It is a more expensive branch that depends on:

- a calibrated physical prior,
- a PMNet-style residual network,
- and a lightweight spatial mixture-of-experts residual head.

If the prior itself is still too weak in `NLoS`, then the experiment risks spending cluster time correcting a poor anchor instead of testing the MoE idea properly.

So the gating logic is:

1. improve the prior first,
2. verify the prior-only `NLoS` score,
3. only then launch `Try 45`.

## Current status

Current best stored `Try 45` prior-only calibration:

- overall RMSE: about `23.57 dB`
- `LoS` RMSE: about `3.81 dB`
- `NLoS` RMSE: about `41.27 dB`

That means:

- `LoS` is already strong enough to act as a useful anchor,
- but `NLoS` is still far from the required threshold.

So `Try 45` remains in **hold** status for cluster launch.

## Current comparison context

Latest model-side reference points:

- `Try 42` (`PMNet + calibrated prior`):
  - overall RMSE: about `19.78 dB`
  - `LoS`: about `3.86 dB`
  - `NLoS`: about `34.47 dB`
- `Try 44` (`PMNet-v3-style control without prior`):
  - overall RMSE: about `22.45 dB`
  - `LoS`: about `4.74 dB`
  - `NLoS`: about `39.06 dB`

These numbers suggest:

- the prior is helping,
- PMNet without prior is not convincing enough,
- and the main unresolved bottleneck is still `NLoS`.


## Required artifacts

When `Try 45` is eventually launched, the following must be archived together:

- YAML config
- checkpoint
- validation JSONs
- prior calibration JSON
- prior calibration Markdown note

Otherwise future inference would not reproduce the same experiment definition.
