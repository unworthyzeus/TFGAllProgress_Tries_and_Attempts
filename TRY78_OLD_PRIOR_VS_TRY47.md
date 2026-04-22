# Old Try 78 prior vs Try 47

This note explains what changed between:

- `TFGpractice/TFGFortySeventhTry47/prior_calibration/regime_obstruction_train_only.json`
- `TFGpractice/TFGSeventyEighthTry78/old_try_78_with_nlos/prior_try78.py`

and why the old Try 78 prior gets much better NLoS calibration.

## Short version

Old Try 78 is not a completely new prior family. It is mostly a cleaner and more self-consistent reimplementation of the same regime-aware obstruction calibration idea from Try 47:

- same general regime split: `city_type x LoS/NLoS x antenna-height bin`
- same core multiscale obstruction features
- same shadow-sigma and elevation-angle features
- same ridge-style linear calibration per regime

The big improvement comes from removing mismatches that were hurting Try 47, especially on NLoS pixels.

## What stayed the same

Both versions use almost the same calibrated feature set:

1. quadratic term on the raw prior
2. linear term on the raw prior
3. `log(1 + distance)`
4. local building density at two scales
5. local mean building height at two scales
6. local NLoS support at two scales
7. interaction terms with distance and angle
8. empirical A2G shadow-sigma
9. one model per `city_type x LoS/NLoS x antenna bin`

So the gain is not "Try 78 discovered a magical new feature". The gain is mostly from making the whole pipeline agree with itself.

## Main differences

### 1. The base formula and the fitted calibration now match

Try 47 calibration was fitted for:

- `hybrid_two_ray_cost231_a2g_nlos`

But the old Try 47 training config used:

- `hybrid_two_ray_cost231`

as the prior input channel.

That means the runtime prior and the fitted calibration were not perfectly aligned. This hurts NLoS most, because NLoS is exactly where the A2G-NLoS branch matters.

Old Try 78 fixes this by computing the intended prior directly inside `prior_try78.py` with:

- LoS: FSPL / two-ray blended with A2G-LoS
- NLoS: `max(COST-231, A2G-NLoS)`

So the model is fitted and evaluated on the same prior definition.

### 2. Try 78 removes the feature-scale mismatch from Try 47

This is the most important improvement.

In Try 47, the calibration fit script and the runtime application path did not build local-height features in exactly the same scale:

- the fit script built local height from `topology_norm * 255`
- the runtime path applied the calibration on already normalized tensors and then divided by `height_scale` again

In practice, the fitted model saw one version of the height features, while inference used a much smaller version of those same features.

This especially destabilized NLoS regimes, because NLoS depends much more strongly on local obstruction morphology.

Old Try 78 avoids this by:

- loading raw HDF5 arrays directly
- computing features directly from those arrays
- using the same feature definition during fitting and evaluation
- using an explicit height normalization (`HEIGHT_SCALE = 90.0`) consistently

This is a major reason the NLoS coefficients in Try 78 are much more moderate and stable.

### 3. Try 78 computes regimes directly from the sample, not mainly from a city lookup

Try 47 inherited a `city_type_by_city` mapping from an older calibration and used that when available.

Old Try 78 instead derives the regime from each sample's own morphology:

- building density
- mean building height
- antenna-height bin

This keeps the calibration more tied to the actual sample being evaluated instead of relying on a fixed city label.

That is a cleaner choice for cross-city generalization.

### 4. Try 78 uses a direct city-holdout evaluation loop

Try 47 used the old dataset split pipeline with random sample-level splits (`val_ratio`, `test_ratio`).

Old Try 78 explicitly builds a city-holdout split:

- fit on one set of cities
- evaluate on different cities

This makes the prior study cleaner and better aligned with the thesis goal of city generalization.

### 5. Try 78 is prior-only and fully self-contained

Try 47 calibration lived inside the larger training/data pipeline:

- normalized tensors
- dataset helpers
- config-driven formula input
- runtime application inside `data_utils.py`

Old Try 78 strips all of that away and keeps only:

- raw HDF5 loading
- prior computation
- feature computation
- regime-wise ridge fit
- city-holdout evaluation

That makes it much easier to verify that the calibration itself is correct.

## Why NLoS improved so much

The NLoS gain is mainly explained by these factors together:

1. the base prior now really is the intended `hybrid_two_ray_cost231_a2g_nlos`
2. the fitted features and inference-time features are now on the same scale
3. morphology-sensitive features are computed consistently from raw maps
4. regime assignment is cleaner
5. the coefficients no longer need to compensate for pipeline bugs

In Try 47, some NLoS regime weights became extremely large, which is a strong sign that the linear calibration was compensating for inconsistencies instead of learning a stable correction.

In old Try 78, the NLoS weights are still meaningful, but much less pathological.

## Evidence from the outputs

Try 47 prior calibration reported:

- LoS RMSE: about `3.81 dB`
- NLoS RMSE: about `41.27 dB`
- overall RMSE: about `23.57 dB`

Old Try 78 reports:

- raw prior LoS RMSE: about `3.95 dB`
- raw prior NLoS RMSE: about `25.15 dB`
- calibrated LoS RMSE: about `3.95 dB`
- calibrated NLoS RMSE: about `3.40 dB`
- calibrated overall RMSE: about `3.92 dB`

The key pattern is:

- LoS was already decent before
- the calibration barely changes LoS
- NLoS is where almost the entire improvement happens

That is exactly what we wanted from an obstruction-aware prior.

## Practical conclusion

Old Try 78 improves over Try 47 mostly by making the calibration pipeline internally consistent.

The main lesson is:

- the old idea from Try 47 was already good
- the implementation around it was not fully aligned
- once the prior formula, feature scaling, regime logic, and evaluation split were cleaned up, the NLoS calibration started working properly

## Recommended takeaway for the thesis

If this comparison is described in the thesis, the safest wording is:

"Try 78 does not introduce a radically different obstruction-aware calibration family. Instead, it reimplements the Try 47 regime-aware prior in a cleaner prior-only pipeline, removing formula/feature inconsistencies and evaluating it under explicit city holdout. The resulting gain appears mainly in NLoS, where the original implementation was most sensitive to calibration mismatch."

## Verdict on the old no-prior models

My blunt call: the old no-prior path-loss models were genuinely bad at NLoS, and Try 76 was needed. There is a shared normalization / scaling contract across the older branches, so a small scale mismatch may have amplified the problem, but I do not see evidence that one hidden normalization bug explains the whole NLoS collapse once the prior is removed.

What the evidence does show is a split story:

- Try 47 had a real prior-pipeline mismatch. The fitted calibration and the runtime prior were not aligned, and that hurt NLoS hard.
- The older branches also share the same dB target scaling/clamp conventions, so a normalization issue could have made the regression harder.
- But Try 73 is explicitly no-prior, and the best full-resolution result in the Try 73 notes still reports NLoS at about 31.48 dB.
- Try 76 then shows the older family collapsing to an NLoS mean of about 36.5 dB against a target mean of about 107.1 dB, which is not the signature of a small implementation bug. It is a mode-collapse / wrong-distribution problem.

So the clean thesis reading is:

- there was a real bug in the prior branch,
- the no-prior direct models were still structurally weak in NLoS,
- and Try 76 was the needed architectural fix, not just a nicer reimplementation.

If you want one sentence to reuse in the thesis, use this:

"The older path-loss models were not merely suffering from a hidden code bug; they were also structurally mismatched to the NLoS distribution, and Try 76 was needed to resolve that failure mode."

## Early Try 5 normalization check

I also checked the early HDF5-era Try 5 branch, because that is the best place for a broad normalization bug to hide.

My read is:

- there is no clear evidence of a universal normalization error in Try 5 itself
- the HDF5 contract is internally consistent: `topology_map / 255`, `delay_spread / 1000`, `angular_spread / 180`, `path_loss / 180`, `los_mask` in `[0, 1]`
- the loader, training, evaluation, and prediction paths all use the same `target_metadata` round-trip for denormalization
- the main normalization smell is in the older cGAN branch documentation, which says the physical target scales were still placeholder defaults and needed alignment to the real encoding

So for Try 5, I would not call normalization the root cause.
At most, there were rough early conventions and some representation choices that could make training harder, but the evidence does not look like one broken normalization layer that explains the failure by itself.
