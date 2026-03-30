# Try 41 and Regime-Aware Error Analysis

This note records the next step after observing that the masked reruns on the newer dataset did not simply transfer the old behavior.

## Main diagnosis

The newer dataset appears harder than the old one:

- more cities,
- more variability,
- and weaker transfer from the older masked baselines.

In particular:

- the clean masked rerun of the old path-loss recipe did not hold up well enough,
- while the branch using an explicit physical formula input was more resilient.

That suggested that the project should stop treating the physical prior as an optional side idea and make it the main formulation.

## Try 41

`Try 41` makes the following statement explicit:

- `prediction = physical_prior + learned_residual`

The prior is not only a conditioning hint anymore. It is the reference field that the network corrects.

Concretely, the try uses:

- the newer dataset: `CKM_Dataset_270326.h5`
- building-mask exclusion
- hybrid `two_ray_ground + COST231` formula input
- multiscale path-loss supervision
- and residual supervision relative to that formula prior

The intended benefit is simple:

- the model should not waste capacity rediscovering the transmitter-centered radial carrier;
- it should focus on the deviations introduced by topology, LoS/NLoS structure, and urban effects.

## Why the model size was kept contained

At this stage, bigger backbones do not look like the main missing ingredient.

So `Try 41` keeps a contained backbone rather than scaling channels up again, in order to:

- isolate the formulation change,
- train faster,
- and avoid mixing "more physics" with "just more capacity".

## Regime-aware error analysis

A new analysis script was added:

- `scripts/analyze_pathloss_regime_errors.py`

It evaluates a path-loss checkpoint by regimes rather than only with one global RMSE.

The script produces:

- a JSON summary,
- a Markdown report,
- and a per-sample CSV table.

It currently splits error using:

- pixel-level distance bins from the transmitter,
- pixel-level LoS vs NLoS,
- sample-level building density bins,
- sample-level building-height proxy bins based on non-zero topology values,
- sample-level antenna-height bins.

This is important because a single global RMSE can hide whether the main failure comes from:

- far-range prediction,
- NLoS propagation,
- dense urban scenes,
- taller building environments,
- or low / high UAV heights.

## Why this matters for training later

The regime analysis is not only for reporting. It is also the basis for the next training decisions.

If the worst error concentrates in specific regimes, the project can later move toward:

- curriculum training,
- balanced sampling,
- or regime-aware fine-tuning.

So the immediate sequence is:

1. evaluate `Try 41`,
2. run the regime analysis,
3. then decide whether the next change should be architectural, data-sampling based, or regime-specific.

## Prior-only calibration without validation leakage

To check how far the physical prior can go on its own, a second script was added:

- `scripts/analyze_formula_prior_generalization.py`

This script is stricter than the earlier quick checks:

- it fits every calibration only on the `train` split,
- it evaluates only on the `val` split,
- and its official RMSE only uses pixels where `topology == 0`.

So the reported numbers do not include building pixels in the main metric.

### Main finding

The raw physical prior is far too crude by itself on this dataset:

- raw prior on validation: about `67.23 dB` RMSE

But train-only calibration helps a lot:

- one global affine calibration: about `26.97 dB`
- regime-aware affine calibration by city type + LoS/NLoS + antenna-height tertile: about `24.17 dB`
- the best tested prior-only system so far, a regime-aware quadratic calibration by city type + LoS/NLoS + antenna-height tertile: about `24.16 dB`

### Important caveat

This is leakage-safe with respect to `train` vs `val`, but it is still a sample-level split, not a city-held-out split.

That means:

- it is a valid train/validation calibration check,
- but it is not yet the hardest possible generalization test for unseen cities.

If stronger generalization guarantees become critical, the next evaluation step should be a city-held-out split.

### Practical conclusion

The physical prior is clearly useful, but prior-only calibration still does not reach the hoped-for `20 dB` level.

So the most reasonable interpretation is:

- the prior should stay in the system,
- but as a scaffold that the network corrects,
- not as a nearly complete replacement for learned prediction.

## Current calibrated prior used by the rerun

The updated rerun of `Try 41` uses the best train-only prior calibration found so far:

- raw formula: hybrid `two_ray_ground + COST231`
- regime split: `city type + LoS/NLoS + antenna-height tertile`
- calibration form: quadratic

The fitted calibration is stored in:

- `TFGFortyFirstTry41/prior_calibration/regime_quadratic_train_only.json`

and the system overview is described in:

- `FORMULA_PRIOR_CALIBRATION_SYSTEM.md`

This calibrated prior is then used:

- as the formula input channel,
- and as the anchor for residual path-loss learning.
