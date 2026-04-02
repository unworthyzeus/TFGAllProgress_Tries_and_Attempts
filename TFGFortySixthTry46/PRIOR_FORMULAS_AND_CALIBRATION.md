# Try 45 Prior Formulas and Calibration

This note documents exactly which formulas feed `Try 45`, where they come from, how they are used, and why the calibration JSON must be stored together with the model.

## Why this matters

`Try 45` is **not** a generic direct predictor. It is a `prior + learned residual` model.

That means the network input and the target decomposition depend on a specific calibrated prior map. If that calibration JSON changes, the effective experiment changes as well.

So the calibration artifact is part of the experiment definition, just like the checkpoint and the YAML.

## Stored calibration artifact

The train-only calibration currently prepared for `Try 45` is stored in:

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
- Eq. (10): LoS excess loss as a function of elevation angle
- Eq. (11): NLoS excess loss as a function of elevation angle
- Eq. (12): shadow-fading standard deviation as a function of elevation angle and regime

How we use them:

- we keep the deterministic large-scale part only
- we do **not** inject the stochastic shadow-fading term into the prior itself
- the A2G elevation-angle `LoS` and `NLoS` terms help make the prior more realistic in difficult regimes
- the `Eq. 12` shadow-variability term is treated as a structured side feature, not as direct additive mean loss

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
- broad-scale building-density x distance interaction
- local `NLoS` support at a near scale
- local `NLoS` support at a broader scale
- broad-scale `NLoS` support x distance interaction

This is why the current calibration is called:

- `regime_obstruction_train_only`

The calibration fitting code lives in:

- [fit_formula_prior_obstruction_calibration.py](C:/TFG/TFGpractice/scripts/fit_formula_prior_obstruction_calibration.py)

## Current stored calibration and validation result

The currently stored train-only calibration is:

- [regime_obstruction_train_only.json](C:/TFG/TFGpractice/TFGFortyFifthTry45/prior_calibration/regime_obstruction_train_only.json)

Its current validation-only prior result is approximately:

- overall RMSE: `23.57 dB`
- LoS RMSE: `3.81 dB`
- NLoS RMSE: `41.27 dB`

That means the enhanced prior is still useful as a structured anchor, but it is **not** enough by itself to solve the hard `NLoS` part of the problem.

This is exactly why `Try 45` still keeps the learned residual branch and now adds a spatial MoE head on top.

## Release gate for Try 45

`Try 45` should only be launched when the prior-only validation score reaches:

- `NLoS RMSE < 20 dB`

Until then, the correct focus is still:

- improving the prior family,
- improving the train-only calibration,
- and preserving reproducibility of the calibration artifact.

## Calibration artifact and reproducibility

The calibration JSON is part of the experiment definition.

That means this file must be archived together with:

- the YAML config
- the model checkpoint
- the validation JSONs
- and any future export or inference bundle

If a different calibration JSON is used later, that is a **different experiment**.

## Relation to Try 22

`Try 45` does not copy the old U-Net from `Try 22`, but it still keeps the ideas from `Try 22` that continued to help:

- bilinear resizing/fusion instead of transposed-convolution-heavy decoding
- multiscale path-loss supervision
- group norm with `batch_size = 1`

So `Try 45` is best read as:

- `Try 42` backbone family
- plus `Try 22` training lessons
- plus a stronger `NLoS`-aware prior
- plus a spatial mixture-of-experts residual head

## Current practical reading

At the moment, the experimental evidence supports the following conclusions:

- `LoS` is already relatively well explained by the current physical family
- the unresolved problem is still `NLoS`
- PMNet without prior is not good enough by itself
- therefore the next gains should come from improving the prior and then letting the MoE residual specialize on top of it

## Additional search utilities

Useful scripts for this prior line:

- [fit_formula_prior_obstruction_calibration.py](C:/TFG/TFGpractice/scripts/fit_formula_prior_obstruction_calibration.py)
- [search_try45_a2g_parameter_grid.py](C:/TFG/TFGpractice/scripts/search_try45_a2g_parameter_grid.py)

The current best stored calibration comes from:

- hybrid deterministic prior with A2G-aware `NLoS` strengthening
- train-only obstruction-aware calibration
- explicit `theta` and `Eq. 12`-style sigma side features
