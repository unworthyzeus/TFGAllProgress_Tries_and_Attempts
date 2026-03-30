# Try 45 Prior Formulas and Calibration

This note documents exactly which formulas feed `Try 45`, where they come from, how they are used, and why the calibration JSON must be stored together with the model.

## Why this matters

`Try 45` is **not** a generic direct predictor. It is a `prior + learned residual` model.

That means the network input and the target decomposition depend on a specific calibrated prior map. If that calibration JSON changes, the effective experiment changes as well.

So the calibration artifact is part of the experiment definition, just like the checkpoint and the YAML.

## Stored calibration artifact

The train-only calibration used by `Try 45` is stored in:

- [regime_obstruction_train_only.json](C:/TFG/TFGpractice/TFGFortyFifthTry45/prior_calibration/regime_obstruction_train_only.json)

This file must be archived together with:

- the YAML config
- the checkpoint
- the validation JSONs

Otherwise future inference would not reproduce the same prior.

## Formula layers in Try 45

`Try 45` uses three layers:

1. base deterministic propagation prior
2. train-only regime-aware calibration
3. learned residual correction from the PMNet-MoE model

So the final prediction is:

`prediction = calibrated_prior + learned_residual`

## Base deterministic prior

The base prior is not a single textbook formula. It is a hybrid.

### 1. Two-ray / free-space LoS branch

Source family:

- classical free-space path loss
- two-ray ground reflection approximation

This branch captures the basic radial structure and the fact that clear-LoS propagation should be much easier than blocked propagation.

In code this is part of:

- [data_utils.py](C:/TFG/TFGpractice/TFGFortyFifthTry45/data_utils.py)

### 2. COST231-Hata urban branch

Source family:

- COST231-Hata urban macro empirical path-loss model

Why it is used:

- the raw two-ray / free-space branch is too optimistic in dense urban settings
- COST231 adds a stronger urban attenuation baseline

In code this is part of:

- [data_utils.py](C:/TFG/TFGpractice/TFGFortyFifthTry45/data_utils.py)

### 3. A2G LoS/NLoS excess-loss branch

Additional source used for `Try 45`:

- [2511.15412v1.md](C:/TFG/TFGpractice/TFG_Proto1/docs/markdown/2511.15412v1/2511.15412v1.md)

The most relevant equations are:

- Eq. (1): total large-scale loss as reference term + excess loss + shadow fading
- Eq. (9): reference attenuation term
- Eq. (11): NLoS excess loss as a function of elevation angle

How we use it:

- we keep the deterministic large-scale part only
- we do **not** inject the stochastic shadow-fading term into the prior itself
- the A2G elevation-angle NLoS term is used to make the prior more realistic in blocked regions

Why this helps:

- our regime diagnostics show that the dominant error is still `NLoS`
- the earlier prior was already reasonable in `LoS`
- so the paper is mainly used to strengthen the `NLoS` branch, not to replace everything

## Why the prior is calibrated again after the formulas

Even a better deterministic formula is still not enough.

The dataset contains:

- multiple city morphologies
- different antenna-height regimes
- strong `LoS / NLoS` asymmetry
- local topology effects that a closed-form formula does not fully capture

So the formula is followed by a **train-only calibration**.

## Train-only calibration structure

The calibration is:

- fitted only on the training split
- evaluated only on the validation split
- masked to official valid pixels only:
  - `topology == 0`
  - dataset mask valid

The calibration groups data by:

- `city type`
- `LoS / NLoS`
- `antenna-height bin`

It then uses a richer feature set built from the sample itself:

- `prior_db_squared`
- `prior_db`
- `log(1 + distance)`
- local building density at a near scale
- local building density at a broader scale
- local mean building-height proxy at a near scale
- local mean building-height proxy at a broader scale
- broad-scale building-density × distance interaction
- local NLoS support at a near scale
- local NLoS support at a broader scale
- broad-scale NLoS support × distance interaction

This is why the calibration is called:

- `regime_obstruction_train_only`

The calibration fitting code lives in:

- [fit_formula_prior_obstruction_calibration.py](C:/TFG/TFGpractice/scripts/fit_formula_prior_obstruction_calibration.py)

## Current stored calibration and validation result

The currently stored train-only calibration is:

- [regime_obstruction_train_only.json](C:/TFG/TFGpractice/TFGFortyFifthTry45/prior_calibration/regime_obstruction_train_only.json)

Its current validation-only prior result is approximately:

- overall RMSE: `23.60 dB`
- LoS RMSE: `3.81 dB`
- NLoS RMSE: `41.32 dB`

That means the enhanced prior is still useful as a structured anchor, but it is **not** enough by itself to solve the hard `NLoS` part of the problem.

This is exactly why `Try 45` still keeps the learned residual branch and now adds a spatial MoE head on top.

## Relation to Try 22

`Try 45` does not copy the old U-Net from `Try 22`, but it still keeps the ideas from `Try 22` that continued to help:

- bilinear resizing/fusion instead of transposed-convolution-heavy decoding
- multiscale path-loss supervision
- group norm with `batch_size = 1`

So `Try 45` is best read as:

- `Try 42` backbone family
- plus `Try 22` training lessons
- plus a stronger NLoS-aware prior
- plus a spatial mixture-of-experts residual head

## Reproducibility note

If `Try 45` is retrained later, the following must remain aligned:

- YAML formula mode
- calibration JSON
- checkpoint
- validation JSONs

If one of these changes, the experiment is no longer the same experiment.
