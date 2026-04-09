# Try 51 Prior Aggregation Note

This note explains why the `Try 51` prior appeared to improve by about `1 dB`
 in `overall RMSE`, even though the displayed `LoS` and `NLoS` prior numbers did
 not both improve.

## Files compared

- `Try 49` prior reference:
  - [validate_metrics_epoch_35_cgan.json](C:/TFG/TFGpractice/cluster_outputs/TFGFortyNinthTry49/fortyninthtry49_pmnet_prior_stage1_t49_stage1_w112_4gpu/validate_metrics_epoch_35_cgan.json)
- `Try 51` literature branch:
  - [validate_metrics_epoch_36_cgan.json](C:/TFG/TFGpractice/cluster_outputs/TFGFiftyFirstTry51/fiftyfirsttry51_pmnet_prior_stage1_literature_t51_stage1_w112_4gpu/validate_metrics_epoch_36_cgan.json)

## The apparent contradiction

The reported prior numbers are:

- `Try 49`
  - `prior overall RMSE = 23.5657 dB`
  - `prior LoS RMSE = 3.8139 dB`
  - `prior NLoS RMSE = 41.2481 dB`
- `Try 51`
  - `prior overall RMSE = 22.5223 dB`
  - `prior LoS RMSE = 3.7887 dB`
  - `prior NLoS RMSE = 41.4446 dB`

At first sight this looks inconsistent:

- `LoS` improved only a little
- `NLoS` got slightly worse
- yet `overall` became much better

## Why this happens

`overall RMSE` is **not** the average of `LoS RMSE` and `NLoS RMSE`.

It comes from the total pixelwise mean squared error:

```text
overall_mse = p_los * mse_los + (1 - p_los) * mse_nlos
overall_rmse = sqrt(overall_mse)
```

So two things matter:

1. the per-regime errors;
2. the fraction of `LoS` vs `NLoS` pixels in the validation split.

In `Try 51`, the split changed to:

- `data.split_mode = city_holdout`

and the prior routing also changed to:

- `prefer_threshold_city_type: true`

That means the comparison is **not** on exactly the same holdout mix as
`Try 49`.

## Estimated LoS share implied by the reported MSE

Using the reported `LoS`, `NLoS`, and `overall` prior MSE values, the implied
`LoS` share is:

- `Try 49`: about `67.94% LoS`
- `Try 51`: about `71.06% LoS`

So the `Try 51` holdout is effectively more `LoS`-heavy.

That alone can pull the `overall RMSE` down even if the `NLoS` prior got a bit
worse.

## Apples-to-apples cross-mix check

If we take the `Try 49` prior errors and evaluate them with the `Try 51`
`LoS/NLoS` mixture, the estimated result is:

- `Try 49 prior on Try 51 mix`: about `22.4207 dB`

If we instead force the `Try 51` prior onto the older `Try 49` mixture, the
estimated result is:

- `Try 51 prior on Try 49 mix`: about `23.6732 dB`

This is the key conclusion:

- the `22.52 dB` of `Try 51` does **not** prove that the new prior is really
  better than the old one;
- once the mix effect is removed, the `Try 51` prior is actually very slightly
  worse than the old `Try 49` prior.

## What really changed

The prior did change in practice, even though the formula family stayed the
same:

- same formula family:
  - `hybrid_two_ray_cost231_a2g_nlos`
- same calibration JSON:
  - [regime_obstruction_train_only_from_try47.json](C:/TFG/TFGpractice/TFGFiftyFirstTry51/prior_calibration/regime_obstruction_train_only_from_try47.json)
- but different routing:
  - `prefer_threshold_city_type: true`
- and different validation split:
  - `city_holdout`

So this is effectively:

- a different prior-routing setup
- measured on a different holdout composition

## Practical takeaway

For `Try 51`, we should **not** use `prior overall RMSE` alone to judge whether
the prior itself improved.

The safer rule is:

1. compare `prior LoS` and `prior NLoS`;
2. compare on the same split whenever possible;
3. treat cross-split `overall prior RMSE` as potentially misleading.

In short:

- the `Try 51` prior did **not** really gain a clean `1 dB`;
- the apparent gain is mostly an aggregation / split-composition effect.
