# Formula Prior Calibration System

This document explains the calibrated physical-prior system currently used by `Try 41`.

## Goal

The raw physical prior is useful, but not accurate enough by itself on the newer dataset.

So the system keeps the physical formula, but calibrates it before the network uses it.

The current idea is:

- start from a hybrid `two_ray_ground + COST231` path-loss prior,
- then apply a regime-aware quadratic calibration learned only from the training split,
- and finally use that calibrated prior as both:
  - an input channel,
  - and the anchor for residual learning.

In short:

- `final_prediction = calibrated_physical_prior + learned_residual`

## Official metric scope

The official RMSE used for this calibration analysis is computed only where:

- `topology == 0`
- and the dataset mask is valid.

So building pixels are not part of the main calibration score.

## Regime definition

The current calibration uses three pieces of information:

- `city type`
- `LoS / NLoS`
- `antenna-height tertile`

### City type

City type is defined from the training split using city-level averages of:

- building density
- building-height proxy

The current labels are:

- `open_lowrise`
- `mixed_midrise`
- `dense_highrise`

If the city is already known from the training split, the stored city label is reused.

If a new city appears later, the fallback is:

- compute its density and height proxy from topology,
- then classify it using the saved training thresholds.

### LoS / NLoS

The calibration uses the LoS mask to distinguish:

- `LoS`
- `NLoS`

and applies different coefficients to each region.

### Antenna-height bin

Antenna height is split into train-defined tertiles:

- `low_ant`
- `mid_ant`
- `high_ant`

## Calibration form

Inside each regime, the prior is calibrated with a quadratic mapping:

- `calibrated = a2 * raw^2 + a1 * raw + a0`

This is still simple enough to stay interpretable, but flexible enough to improve over a single global affine fit.

## Stored calibration file

The current train-only calibration used by `Try 41` is stored at:

- `TFGFortyFirstTry41/prior_calibration/regime_quadratic_train_only.json`

That JSON contains:

- the dataset used,
- split settings,
- the official metric mask definition,
- train-derived thresholds,
- city-type assignments for known training cities,
- and the quadratic coefficients for each regime.

## Why this is safer than the earlier quick checks

The calibration was fitted only on `train` and evaluated only on `val`.

So it avoids direct validation leakage during calibration.

However, an important caveat remains:

- the current split is still sample-based, not city-held-out.

That means the calibration is train/validation-safe, but not yet the strongest possible unseen-city test.

## What to do for a new dataset

The calibration should not be assumed to transfer automatically to a new dataset.

For a genuinely new dataset, the correct workflow is:

1. build the new train/val split,
2. refit the calibration on the new training split,
3. validate it on the new validation split,
4. only then use it as the prior anchor for residual learning.

## Script

The current calibration analysis and fitting workflow is documented and reproducible with:

- `scripts/fit_formula_prior_calibration.py`
- `scripts/analyze_formula_prior_generalization.py`

This script:

- fits train-only calibration variants,
- evaluates them on validation,
- reports which regime-aware prior works best,
- and writes a reproducible JSON/Markdown summary.

If needed later, it can be extended into a dedicated city-held-out calibration pipeline.

## How Try 42 uses this prior

`Try 42` keeps this same calibrated prior system, but changes the network family on top of it.

Instead of using the prior only as a conditioning channel inside the old U-Net/cGAN line, `Try 42` uses it as:

- an explicit input channel;
- the additive baseline for reconstruction;
- and the reference for the residual target.

So for `Try 42` the practical rule is:

- do not ask the network to predict the whole path-loss map from scratch;
- ask it to predict only the correction to this calibrated prior.
