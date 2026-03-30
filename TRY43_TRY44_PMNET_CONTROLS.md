# Try 43 and Try 44: PMNet control logic

This note records why `Try 43` and `Try 44` were added after `Try 42`.

## Why Try 42 alone was not enough

`Try 42` tested:

- calibrated physical prior
- plus learned residual
- with a PMNet-inspired backbone

That was useful, but it mixed two different questions:

1. is the calibrated prior helping?
2. is PMNet a better path-loss backbone than the older U-Net family?

So a control branch was required.

## Try 43

`Try 43` is the direct PMNet control without the physical prior.

It keeps:

- building-mask exclusion
- LoS input
- distance-map input
- antenna-height channel
- the PMNet family

It removes:

- the physical prior input
- the prior-residual decomposition

So `Try 43` predicts `path_loss` directly.

The purpose is simple:

- measure how much PMNet can do on its own
- before giving it the calibrated prior.

## Try 44

`Try 44` is a stricter PMNet control branch.

It still has no physical prior, but it replaces the earlier simplified PMNet with a more faithful PMNet-v3-style implementation:

- bottleneck residual encoder
- ASPP context aggregation
- decoder path closer to the official repository

So `Try 44` is the cleaner answer to:

- is the PMNet family itself a good fit for this project,
- if implemented more faithfully?

## Why both should still expose regime metrics

Even without a physical prior, these branches still need the richer diagnostics introduced in `Try 42`.

That means:

- `LoS / NLoS` RMSE
- RMSE by `city type`
- RMSE by `antenna-height bin`
- RMSE by the combined calibration-style regime

These metrics are important because the global RMSE hides where the failure is concentrated.

For example, the first `Try 42` result already showed a strong asymmetry:

- `LoS` was much easier
- `NLoS` remained the dominant error source

So even the no-prior controls must keep the same regime breakdown.

## Recommended interpretation path

The current comparison should be read in this order:

1. `Try 42`
   - PMNet-inspired backbone
   - calibrated prior
2. `Try 43`
   - PMNet control without prior
3. `Try 44`
   - more faithful PMNet-v3-style control without prior

Then the decision for a future `Try 45` is:

- if `Try 44 > Try 43`, the more faithful PMNet backbone is worth keeping;
- if `Try 44` is also competitive enough overall, then `Try 45 = Try 44 + calibrated prior` becomes the logical next step.
