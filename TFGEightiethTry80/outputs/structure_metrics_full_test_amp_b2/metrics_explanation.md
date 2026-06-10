# Metrics Explanation

> Stale numbers: this explanation quotes values generated before the
> MapCorr/GradCorr metric fix. See `RERUN_REQUIRED_MAPCORR_GRADCORR_FIX.md`
> and rerun before using the quoted values in the thesis/report. Preferred
> combined script:
> `C:\TFG\TFGAllProgress_Tries_and_Attempts\TFGEightiethTry80\scripts\compare_prior_try80_all_quality_metrics.py`.

This folder compares the frozen calibrated priors against the final Try80 model
on the Try80 test split.

## Prediction Pairs

| Pair name | Meaning |
|---|---|
| `prior` | Ground truth vs frozen calibrated prior |
| `model` | Ground truth vs frozen calibrated prior + Try80 residual model |
| `prior_vs_model` | Frozen calibrated prior vs prior + Try80 residual model |

For thesis reporting, the key comparison is usually `model - prior`: negative is
better for error metrics, positive is better for similarity/correlation metrics.

## RMSE

Pixel-weighted root mean squared error over valid receiver pixels:

```text
RMSE = sqrt(mean((prediction - target)^2))
```

Lower is better. This is the primary accuracy metric.

## PSNR

Peak signal-to-noise ratio derived from RMSE:

```text
PSNR = 20 log10(task_range / RMSE)
```

Higher is better. The task ranges used here are:

| Output | Range |
|---|---:|
| PL | 180 dB |
| DS | 400 ns |
| AS | 90 deg |

PSNR is useful as an image-quality metric, but it is RMSE-derived, so it should
be reported as a companion to RMSE rather than independent evidence.

## NRMSE

Normalized RMSE:

```text
NRMSE = RMSE / task_range
```

Lower is better. It puts PL, DS, and AS on comparable unitless scales.

## SSIM

Structural similarity index, computed on the native output maps over the same
valid pixels used by RMSE.

Higher is better in principle, but for DS and AS it can be misleading because
spread maps are heavy-tailed and SSIM mixes luminance, contrast, and structure.
In this run, Try80 improves RMSE and visual/spatial structure for spreads while
native-domain SSIM decreases, so SSIM should be treated as a secondary
diagnostic rather than the main structure claim for DS/AS.

## MapCorr

`MapCorr` is the final full-run name for what was first tested as
`valid_z_corr`.

It is Pearson correlation between the target map and prediction map, computed
only on valid receiver pixels:

```text
MapCorr = corr(target_valid_pixels, prediction_valid_pixels)
```

Equivalently, it is correlation after subtracting each map's valid-pixel mean
and dividing by its valid-pixel standard deviation. This is why the diagnostic
name used `z_corr`: it measures whether the spatial pattern matches after
removing global bias and scale.

Higher is better. This is the preferred structure metric for PL/DS/AS because it
focuses on whether high/low regions appear in the right places.

### What Z-Score Means Here

A z-score is a standardized value:

```text
z = (value - mean) / standard_deviation
```

For MapCorr, this is done only over valid receiver pixels. In words:

1. Take the GT map and keep only valid pixels.
2. Subtract the valid-pixel mean of the GT map.
3. Divide by the valid-pixel standard deviation of the GT map.
4. Do the same independently for the prediction map.
5. Compute the average product of the two standardized maps.

So MapCorr does not ask whether the prediction has the exact same absolute
level. RMSE already measures that. MapCorr asks whether both maps go high and
low in the same places after removing offset and scale. This makes it a spatial
pattern metric.

This is closely related to Pearson correlation and zero-mean normalized
cross-correlation. In image/template matching literature, normalized
cross-correlation is used because subtracting the mean and normalizing by
standard deviation makes the comparison less sensitive to brightness and
contrast changes.

### Why MapCorr Can Look Numerically Low

MapCorr is deliberately strict. It is computed over all valid receiver pixels,
not over smoothed regions, cropped examples, or per-sample qualitative panels.
For DS and AS, the ground-truth maps are noisy, heavy-tailed, and often contain
very local scattering structures. A small spatial displacement, a slightly
smoother ridge, or a corrected absolute level can reduce pixelwise correlation
even when the visual structure is better.

Because each map is z-scored before comparison, MapCorr ignores global
bias/scale improvements and only rewards pixel-level spatial alignment. For
spread maps, values around `0.3` to `0.6` are therefore not "bad" in the way a
classification-style correlation might sound. The important result is that
Try80 beats the prior under this harsh criterion:

| Output | GT-prior MapCorr | GT-model MapCorr | Gain |
|---|---:|---:|---:|
| PL | 0.948640 | 0.962807 | +0.014167 |
| DS | 0.253268 | 0.374358 | +0.121090 |
| AS | 0.390245 | 0.596313 | +0.206068 |

For DS, moving from `0.25` to `0.37` is a large relative gain under this
pixel-level criterion. For AS, moving from `0.39` to `0.60` is stronger still.

## GradCorr

Gradient-magnitude correlation over valid receiver pixels:

```text
GradCorr = corr(|grad(target)|, |grad(prediction)|)
```

Higher is better. This measures whether the prediction preserves spatial
transitions, ridges, edges, and local variation patterns. It is useful when the
absolute values improve but we also want evidence that the map structure itself
is better.

### Why GradCorr Is Also Harsh

GradCorr is even more punishing than MapCorr because it compares local
derivative magnitude. If a ridge or transition is shifted by only a few pixels,
the gradient pattern can decorrelate even when the map is qualitatively right.
It also penalizes smoothing: a prediction may have better RMSE while having
weaker local gradients.

So absolute GradCorr values should be read as a strict edge/transition
alignment diagnostic. The important result is again the improvement over the
prior:

| Output | GT-prior GradCorr | GT-model GradCorr | Gain |
|---|---:|---:|---:|
| PL | 0.995352 | 0.996257 | +0.000905 |
| DS | 0.092910 | 0.145567 | +0.052658 |
| AS | 0.298502 | 0.444885 | +0.146383 |

For DS/AS, positive GradCorr gains mean Try80 recovers more of the local
spread-map structure than the prior, even though the absolute correlation values
remain modest because the metric is harsh.

## Literature Context

For a more focused explanation of only `MapCorr` and `GradCorr`, including
non-radio examples and source links, see
`mapcorr_gradcorr_literature_notes.md` in this folder.

### Radio Map Papers

Radio-map papers usually report RMSE/NMSE/SSIM/PSNR rather than MapCorr or
GradCorr directly.

- **RadioDiff** reports RMSE, NMSE, PSNR, and SSIM for radio-map construction.
  In its AFT ablation, SSIM improves from `0.9465` to `0.9691`, and PSNR from
  `31.62` to `35.13`.
- **R2Net / RadioUNet-style radio map estimation** also reports NMSE, RMSE,
  SSIM, and PSNR. In one table, benchmark SSIM values range from about `0.57`
  to `0.89`, while the best model reaches about `0.89` for indoor pathloss radio
  maps.
- **RadioDiff-3D** reports RMSE, NMSE, SSIM, and PSNR for ToA/DoA radio-map
  outputs, with no-sampling SSIM values around `0.86` to `0.97` depending on
  output type, and higher values under sampling.
- **The ICASSP Pathloss Radio Map Prediction Challenge** uses RMSE as the main
  challenge metric, which supports keeping RMSE as the primary accuracy number.

These examples are useful for justifying RMSE/PSNR/SSIM as standard radio-map
metrics. They do not provide a direct target threshold for MapCorr or GradCorr,
because those are stricter structure diagnostics added for this analysis.

### Correlation / Z-Normalized Similarity

MapCorr is a full-map version of a standard idea: compare centered and
standardized signals. Normalized cross-correlation is widely used in image
template matching and registration because it removes mean/contrast effects and
focuses on pattern agreement.

General correlation interpretation rules are only rough. Some statistics
guides classify Pearson `r >= 0.3` as moderate or large depending on the rule,
and `r >= 0.5` as large, but those thresholds are not universal. For our maps,
the fair interpretation is comparative:

```text
GT-model MapCorr > GT-prior MapCorr
```

That is what matters, because both scores are computed on the same samples,
masks, tasks, and pixels.

### Gradient Metrics

GradCorr is related in spirit to gradient-based full-reference image quality
assessment. The GMSD/GMSM family compares gradient magnitude maps because image
gradients are sensitive to local structural distortions. Our `GradCorr` is not
exactly GMSD, but it uses the same idea: if the target has a ridge, edge, or
rapid transition, the prediction should place a similar transition in the same
valid-pixel region.

There is no universal "good GradCorr" threshold for DS/AS radio spread maps.
The metric is harsher than MapCorr because even small spatial shifts can
decorrelate gradients. The important result is that Try80 improves it for all
three outputs.

## Recommended Reporting

Use:

- `RMSE` as the main accuracy metric.
- `PSNR` or `NRMSE` as normalized/image-quality companions.
- `MapCorr` as the main structure metric.
- `GradCorr` as a secondary structure/edge metric.

Mention SSIM only as a diagnostic: it improves for PL but is not well aligned
with the visually improved DS/AS spread structures in native scale.
