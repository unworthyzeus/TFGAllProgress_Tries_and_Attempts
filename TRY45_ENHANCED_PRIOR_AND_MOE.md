# Try 45: Enhanced Prior + Spatial MoE Residual

`Try 45` is the next branch after `Try 42`.

## Core idea

Keep what was useful in `Try 42`:

- calibrated physical prior
- PMNet-style residual prediction
- regime metrics

Then add two things:

1. a stronger `NLoS`-aware prior
2. a lightweight spatial mixture-of-experts residual head

So the model still predicts:

`path_loss = calibrated_prior + learned_residual`

but both the prior and the residual branch become more specialized.

## Why this branch exists

`Try 42` showed a clear pattern:

- `LoS` was already relatively good
- `NLoS` remained the dominant source of error

That suggests the next gain is not likely to come from a larger generic backbone alone.

Instead, we want:

- a better prior where `NLoS` is physically harder
- and a residual branch that can specialize by regime

## Prior side

The prior now combines:

- the previous hybrid deterministic path-loss prior
- train-only calibration by regime
- multiscale local topology descriptors
- local shadow-support proxies from the LOS map

This keeps the prior more structured and makes it more informative before the learned model starts correcting it.

## Residual side

The residual branch uses a spatial MoE head:

- shared PMNet-style encoder/context backbone
- several small residual experts
- per-pixel gating maps
- expert balance regularization

This is intended to let different experts focus on:

- easier LoS corrections
- dense urban NLoS
- low-antenna difficult regimes
- fallback general corrections

without forcing one residual head to explain every case in the same way.

## Metrics

`Try 45` keeps regime-level validation diagnostics:

- overall `path_loss` RMSE
- prior-only RMSE
- `LoS / NLoS`
- `city type`
- `antenna-height bin`
- combined calibration regime

This is important because the global RMSE alone hides where the gains or failures occur.

## Files

Main files:

- [model_pmnet.py](C:/TFG/TFGpractice/TFGFortyFifthTry45/model_pmnet.py)
- [train_pmnet_residual.py](C:/TFG/TFGpractice/TFGFortyFifthTry45/train_pmnet_residual.py)
- [data_utils.py](C:/TFG/TFGpractice/TFGFortyFifthTry45/data_utils.py)
- [fortyfifthtry45_pmnet_moe_enhanced_prior.yaml](C:/TFG/TFGpractice/TFGFortyFifthTry45/experiments/fortyfifthtry45_pmnet_moe_enhanced_prior/fortyfifthtry45_pmnet_moe_enhanced_prior.yaml)
- [PRIOR_FORMULAS_AND_CALIBRATION.md](C:/TFG/TFGpractice/TFGFortyFifthTry45/PRIOR_FORMULAS_AND_CALIBRATION.md)

Calibration artifacts:

- [regime_obstruction_train_only.json](C:/TFG/TFGpractice/TFGFortyFifthTry45/prior_calibration/regime_obstruction_train_only.json)
- [regime_obstruction_train_only.md](C:/TFG/TFGpractice/TFGFortyFifthTry45/prior_calibration/regime_obstruction_train_only.md)

Utilities:

- [fit_formula_prior_obstruction_calibration.py](C:/TFG/TFGpractice/scripts/fit_formula_prior_obstruction_calibration.py)
- [export_try42_try45_predictions.py](C:/TFG/TFGpractice/scripts/export_try42_try45_predictions.py)
