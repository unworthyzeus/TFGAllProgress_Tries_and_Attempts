# Supervisor Summary: Tries 20 to 47

## Main message

The recent experiments show that the project is no longer bottlenecked by generic decoder tuning alone.

The dominant remaining difficulty is the masked ground-valid `NLoS` regime.

That is why the current path-loss line has moved from:

- plain U-Net tuning,
- to physical priors,
- to residual learning,
- to explicit `LoS / NLoS` specialization,
- and now to a hybrid branch that combines the strongest parts of the old U-Net family and the newer prior-aware family.

## What Try 22 taught us

`Try 22` remained important for a long time because it combined two changes that consistently helped:

- bilinear decoder/fusion
- multiscale path-loss supervision

It became the strongest clean path-loss baseline of that family.

After masked reevaluation, its modern reference numbers are approximately:

- overall RMSE: `19.94 dB`
- `LoS`: `3.78 dB`
- `NLoS`: `34.43 dB`

So `Try 22` is still very good in `LoS`, but its `NLoS` error remains high under the stricter modern masking.

## What Try 42 taught us

`Try 42` introduced the more important conceptual shift:

- `prediction = calibrated_physical_prior + learned_residual`

That branch improved the balance relative to `Try 22`, especially in the harder masked `NLoS` setting.

Best observed numbers from the current saved results are approximately:

- overall RMSE: `19.17 dB`
- `LoS`: `3.84 dB`
- `NLoS`: `33.40 dB`

So `Try 42` is slightly worse than `Try 22` in `LoS`, but better overall and better in `NLoS`.

## What Try 46 taught us

`Try 46` tested whether explicit `LoS / NLoS` branching plus a dedicated `NLoS` MoE head could solve the hard regime.

The idea was correct in spirit, but the PMNet-style backbone still did not look convincing enough.

That suggested the problem might not be "prior vs no prior", but rather:

- keeping the prior,
- while returning to a stronger spatial backbone.

## Why Try 47 exists

`Try 47` is the direct answer to that finding.

It combines:

- the `Try 22` U-Net family:
  - bilinear decoder
  - group norm
  - scalar FiLM
  - distance-map channel
- the calibrated prior/residual formulation from the `Try 42` family
- explicit `LoS / NLoS` specialization
- a small `LoS` head
- an `NLoS`-only MoE head
- extra obstruction proxy channels:
  - shadow depth
  - distance since LoS break
  - maximum blocker height
  - blocker count
- stronger `NLoS` combo losses for difficult subsets

So `Try 47` is not "another PMNet try".

It is a deliberate return to the stronger old spatial family, now augmented with the newer physical-prior and regime-specialization ideas.

## Why this is the right next experiment

The current evidence suggests:

- `LoS` is already relatively well controlled once the prior is available,
- the real bottleneck is masked ground-valid `NLoS`,
- and PMNet alone has not been convincing enough as the main path-loss backbone.

That makes `Try 47` a more principled next step than:

- simply making PMNet larger,
- or adding more generic experts without better spatial grounding.

## Calibration note

`Try 47` is trained only after a separate prior-calibration job refreshes:

- `prior_calibration/regime_obstruction_train_only.json`

This calibration is train-only and is part of the experiment definition.

The training job is submitted with a dependency on the calibration job so that the network always uses the intended prior.
