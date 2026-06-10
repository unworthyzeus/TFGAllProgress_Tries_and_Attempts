# MapCorr and GradCorr Literature Notes

> Stale numbers: this note quotes values generated before the MapCorr/GradCorr
> metric fix. See `RERUN_REQUIRED_MAPCORR_GRADCORR_FIX.md` and rerun before
> using the quoted values in the thesis/report.

This note focuses only on the two structure metrics added after the SSIM/RMSE
run: `MapCorr` and `GradCorr`.

## Citation And Provenance Note

The formulas for `MapCorr` and `GradCorr` are the formulas implemented in our
evaluation script. The external citations justify the statistical/image-analysis
ideas behind them:

- z-scoring: standard score definition from NIST.
- `MapCorr`: Pearson/PCC and normalized cross-correlation as image similarity.
- `GradCorr`: gradient correlation, edge preservation, and gradient-magnitude
  image-quality metrics.
- "Good score" ranges: not copied from a single paper. They are reporting
  anchors derived from general Pearson-correlation interpretation guides, then
  adjusted qualitatively for the fact that gradient correlation is stricter.
  They should be presented as practical guidance, not universal thresholds.

## Exact Metrics Used Here

`MapCorr` is the Pearson correlation coefficient between the ground-truth map
and the prediction map, computed over valid receiver pixels only:

```text
MapCorr = corr(GT_valid_pixels, prediction_valid_pixels)
```

Equivalently, each map is z-normalized over its valid pixels first:

```text
z = (value - valid_pixel_mean) / valid_pixel_standard_deviation
MapCorr = mean(z_GT * z_prediction)
```

So `MapCorr` does not measure absolute offset or scale. RMSE already does that.
`MapCorr` measures whether high and low regions appear in the same places.
This is the same statistical idea as PCC/Pearson correlation used as a
reference image-similarity metric in medical image synthesis, and as normalized
cross-correlation in image matching.

`GradCorr` is the Pearson correlation coefficient between gradient-magnitude
maps:

```text
GradCorr = corr(|grad(GT)|_valid_pixels, |grad(prediction)|_valid_pixels)
```

In the script, gradient magnitude is:

```text
|grad(x)| = sqrt((dx/drow)^2 + (dx/dcol)^2)
```

This makes `GradCorr` an edge/transition metric: it asks whether ridges, sharp
changes, local scattering transitions, and other spatial variations appear in
the same valid-pixel locations.
This follows the same motivation as gradient-correlation and
gradient-magnitude-similarity metrics in image registration, restoration, and
image-quality assessment.

Both metrics are full-reference, pixel-aligned similarity metrics. Higher is
better. A value near `1` means very strong agreement, `0` means little linear
agreement, and negative values would mean inverted structure.

## Why The Original Diagnostic Was Called valid_z_corr

The first diagnostic name, `valid_z_corr`, meant:

- `valid`: compute only on valid receiver pixels.
- `z`: z-score both maps independently.
- `corr`: compute correlation between those standardized valid-pixel maps.

The final report name is `MapCorr`, but mathematically it is the same idea.

## What A Z-Score Means

A z-score is a way to express a value relative to the distribution it came from:

```text
z = (value - mean) / standard_deviation
```

Source: NIST gives the standard z-score form as `(X(i) - xbar) / s`, where
`xbar` is the sample mean and `s` is the standard deviation:
https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/zscore.htm

Interpretation:

- `z = 0`: the value is exactly at the mean.
- `z = +1`: the value is one standard deviation above the mean.
- `z = -1`: the value is one standard deviation below the mean.
- `z = +2`: the value is unusually high relative to that map.

For `MapCorr`, z-scoring is done separately for each map using only valid
receiver pixels. This converts both maps into "relative spatial pattern" maps:
positive z-scores are above that map's own average, negative z-scores are below
that map's own average.

That means `MapCorr` answers:

```text
When the GT is high relative to its own average, is the prediction also high
relative to its own average at the same pixel?
```

This is useful because RMSE already measures whether the absolute values are
right. `MapCorr` instead isolates spatial organization.

## Heavy-Tailed Spread Maps Make Z-Correlation Harsher

The spread outputs (`DS` and `AS`) should not be interpreted as Gaussian-like
maps. Empirically, spread maps are heavy-tailed: most valid pixels live in a
moderate range, while a smaller number of scattering/ridge/transition pixels
take much larger values. A single Gaussian does not describe that behavior well,
and even a simple Gaussian-mixture view can miss the fact that rare tail events
dominate the second-moment statistics used by z-scoring and Pearson
correlation.

This matters because `MapCorr` is a z-normalized Pearson correlation. It uses
the mean, standard deviation, covariance, and variance of the valid-pixel
values. In heavy-tailed data, those quantities are strongly influenced by rare
extreme pixels. Pearson correlation is known to be sensitive to outliers and to
behave less robustly under heavy-tailed / high-kurtosis distributions:

- Pernet et al., 2013, "Robust Correlation Analyses": Pearson correlation is
  restricted to linear associations and is overly sensitive to outliers:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3541537/
- de Winter et al., 2024, "Comparing the Pearson and Spearman Correlation
  Coefficients Across Distributions and Sample Sizes": Pearson correlation is
  more suitable for light-tailed distributions, while rank-based alternatives
  are preferable with heavy-tailed variables or outliers:
  https://arxiv.org/abs/2408.15979

So a lower `MapCorr` for `DS` or `AS` does not necessarily mean the prediction
has no visible structure. It can mean that the metric is heavily penalizing
rare high-spread structures that are slightly shifted, smoothed, or not matched
pixel-for-pixel. The same logic is even stronger for `GradCorr`, because
gradient magnitude emphasizes local transitions and makes rare sharp structures
even more influential.

This is why `DS MapCorr = 0.374` can still be useful: it is a strict
valid-pixel z-correlation on a heavy-tailed output, and it improves over the
prior by `+0.121` under exactly the same samples and masks. The correct claim is
not "the absolute z-correlation is huge"; the correct claim is "Try80 improves
the heavy-tailed spread-map structure over the calibrated prior under a harsh
full-map correlation metric."

## Similar Metrics In Other Fields

### Image similarity and medical image synthesis

In medical image-to-image translation, Pearson correlation coefficient (PCC) is
used as a reference similarity metric. A recent Scientific Reports review says
PCC measures linear dependency between intensities in a prediction and reference
image at each pixel location, and that pure scale/shift normalization does not
change this score:

- Dohmen et al., 2025, "Similarity and quality metrics for MR image-to-image
  translation": https://www.nature.com/articles/s41598-025-87358-0

This is the closest general analogue of `MapCorr`.

### Template matching and image registration

Normalized cross-correlation (NCC) is a classic image matching metric. It
subtracts local means and normalizes by variance, making the comparison more
robust to brightness and contrast changes:

- Briechle and Hanebeck, 2001, "Template Matching using Fast Normalized Cross
  Correlation": https://isas.iar.kit.edu/pdf/SPIE01_BriechleHanebeck_CrossCorr.pdf

`MapCorr` is not a sliding template matcher, but it is the same normalized
correlation principle applied to the whole valid map.

### Gradient correlation in image registration

Gradient correlation is used directly in image registration. MATLAB's
`imregcorr` documentation describes the method as estimating the geometric
transformation that aligns a moving image with a reference image using
gradient correlation, and its `Method` option includes `"gradcorr"` for
gradient correlation:

- MathWorks, `imregcorr` documentation:
  https://www.mathworks.com/help/images/ref/imregcorr.html

A review of image registration methods notes that gradient-based metrics use
image gradients because they capture structural and contrast changes, and lists
gradient correlation as a common gradient-based metric:

- Haleem and Garg, 2024, "Review on image registration methods for the quality
  control in additive manufacturing":
  https://link.springer.com/article/10.1007/s40964-024-00932-2

Tzimiropoulos et al. introduce normalized gradient correlation (NGC) for robust
FFT-based image registration. They state that standard correlation can be
replaced with gradient-based correlation schemes, and that gradient correlation
combines image-gradient magnitude and orientation. Their version uses gradient
orientation and magnitude to recover motion, while ours correlates gradient
magnitudes after the maps are already aligned. The shared idea is that gradients
emphasize salient spatial structure and suppress slow background variation:

- Tzimiropoulos et al., 2010, "Robust FFT-Based Scale-Invariant Image
  Registration with Image Gradients":
  https://ibug.doc.ic.ac.uk/media/uploads/documents/ieee_tpami_2010.pdf

### Edge preservation and denoising

In image restoration and SAR despeckling, edge-preservation metrics often compare
gradient or edge information between an original/reference image and a processed
image. One SAR edge-preservation review describes an edge correlation index that
evaluates gradient correlations between the reference and filtered image, with
`1` representing ideal edge preservation:

- Ma et al., 2022, "A No-Reference Edge-Preservation Assessment Index for SAR
  Image Filters under a Bayesian Framework Based on the Ratio Gradient":
  https://www.mdpi.com/2072-4292/14/4/856

This is conceptually close to `GradCorr`: both ask whether the processed image
keeps the same edge/transition structure as the reference.

### Gradient-magnitude image quality metrics

Gradient Magnitude Similarity Deviation (GMSD) is a full-reference image quality
metric built from pixel-wise gradient-magnitude similarity. It is not the same
formula as `GradCorr`, but it supports the same motivation: image gradients are
sensitive to local distortions and local structure:

- Xue et al., 2014, "Gradient Magnitude Similarity Deviation: A Highly Efficient
  Perceptual Image Quality Index": https://arxiv.org/pdf/1308.3052

## What Counts As A Good Score

There is no universal threshold. Correlation interpretation depends on the data,
the domain, the mask, the output variable, and the exact metric. General
statistics guides explicitly warn that correlation thresholds are contextual:

- effectsize interpretation guide:
  https://easystats.github.io/effectsize/articles/interpret.html

As a rough anchor, common Pearson-correlation rules often put `r >= 0.3` in the
moderate/large region and `r >= 0.5` in the large/strong region, depending on the
rule. These thresholds come from general effect-size interpretation guides, not
from radio-map-specific `MapCorr` or `GradCorr` papers. For these radio spread
maps, the better thesis interpretation is not a universal label; it is the
matched comparison:

```text
GT-model score > GT-prior score on the same test samples, masks, and pixels.
```

### Practical MapCorr Ranges

Because `MapCorr` is a Pearson correlation on full valid maps, these are useful
general anchors. They are based on common Pearson `r` interpretation rules
summarized by effectsize, with the caveat that image/map data are not the same
as psychology or social-science effect sizes:

Where the cutoffs come from:

| Source summarized by effectsize | Pearson `r` cutoffs |
|---|---|
| Funder and Ozer (2019) | `0.05`, `0.10`, `0.20`, `0.30`, `0.40` |
| Gignac and Szodorai (2016) | `0.10`, `0.20`, `0.30` |
| Cohen (1988) | `0.10`, `0.30`, `0.50` |
| Evans (1996) | `0.20`, `0.40`, `0.60`, `0.80` |

Source: https://easystats.github.io/effectsize/articles/interpret.html

The `MapCorr` table below uses the most defensible shared anchors from those
Pearson-correlation grids: `0.10`, `0.30`, `0.50`, and `0.80`. The first three
come directly from Cohen-style correlation interpretation, and `0.80` is the
Evans-style boundary for very strong correlation. This is cleaner than adding
extra unsourced cutoffs.

| MapCorr | General reading |
|---:|---|
| `< 0.10` | Negligible spatial pattern agreement |
| `0.10 - 0.30` | Weak spatial pattern agreement |
| `0.30 - 0.50` | Moderate / useful spatial pattern agreement |
| `0.50 - 0.80` | Strong spatial pattern agreement |
| `>= 0.80` | Very strong spatial pattern agreement |

For clean, smooth maps like pathloss, a good model can reasonably reach very
high `MapCorr` values; scores above `0.90` can be described as near-ceiling in
that specific setting, but `0.90` is not a separate literature threshold. For
noisy/heavy-tailed spread maps, `0.30 - 0.60` can already be a meaningful
structure score, especially when it clearly beats the physics/calibrated prior
on the same pixels.

### Practical GradCorr Ranges

`GradCorr` is usually harsher than `MapCorr`. It correlates local derivative
magnitude, so a ridge shifted by a few pixels, a smoother prediction, or noisy
fine-scale scattering can lower the score sharply.

Sources for the metric concept:

- MathWorks documents `imregcorr` as an image-registration method that can use
  `"gradcorr"` / gradient correlation:
  https://www.mathworks.com/help/images/ref/imregcorr.html
- Tzimiropoulos et al. introduce normalized gradient correlation for robust
  image registration and describe gradient correlation as combining
  image-gradient magnitude and orientation:
  https://ibug.doc.ic.ac.uk/media/uploads/documents/ieee_tpami_2010.pdf

The table below is an interpretation guide, not a published universal scale.
It starts from the same Pearson-correlation anchors as `MapCorr`, then uses
stricter wording because gradient-based metrics are known to focus on local
edge/transition structure and can be sensitive to local shifts, smoothing, and
noise. That motivation is supported by gradient-correlation registration work,
SAR edge-preservation indices, and GMSD-style gradient-magnitude IQA.

So the `GradCorr` numbers are not taken from a paper as fixed thresholds. To
avoid inventing a second scale, use the same Pearson `r` anchors as `MapCorr`
for rough orientation, but interpret them more cautiously because the correlated
variables are gradient magnitudes rather than raw map values.

For `GradCorr`, the most important number is often not the absolute score but
the improvement over the prior, because both predictions are evaluated on the
same hard-to-align gradients.

Useful general anchors:

| GradCorr | General reading |
|---:|---|
| `< 0.10` | Negligible or very weak edge/transition agreement |
| `0.10 - 0.30` | Weak but detectable local-structure agreement; useful for hard/noisy maps if it beats the prior |
| `0.30 - 0.50` | Moderate edge/transition preservation |
| `>= 0.50` | Strong edge/transition preservation |

For spread maps, `GradCorr` should be interpreted mainly as a baseline-relative
metric. A score can look numerically low and still be useful if it consistently
improves over the prior, because the prior and model are judged on exactly the
same GT maps, masks, and pixels.

These bands are intentionally coarser than the `MapCorr` bands. The cited
gradient-correlation sources justify the metric concept, but they do not define
universal quality thresholds for our specific full-map gradient-magnitude
correlation. The safest thesis claim is therefore comparative: `GradCorr`
increases from prior to Try80 for every output.

## Reading Our Scores

Global test-split results:

| Output | GT-prior MapCorr | GT-model MapCorr | Gain | GT-prior GradCorr | GT-model GradCorr | Gain |
|---|---:|---:|---:|---:|---:|---:|
| PL | 0.948640 | 0.962807 | +0.014167 | 0.995352 | 0.996257 | +0.000905 |
| DS | 0.253268 | 0.374358 | +0.121090 | 0.092910 | 0.145567 | +0.052658 |
| AS | 0.390245 | 0.596313 | +0.206068 | 0.298502 | 0.444885 | +0.146383 |

Practical interpretation:

- `PL`: both metrics are near ceiling because the calibrated prior is already
  very structurally strong. Try80 still improves them.
- `DS MapCorr = 0.374`: this sounds modest, but it is a harsh full-map
  pixelwise structure metric on noisy/heavy-tailed delay-spread maps. The gain
  over prior is large: about `+48%` relative to the prior score.
- `DS GradCorr = 0.146`: absolute value is low because gradient agreement is
  stricter than map agreement. The `+57%` relative gain says Try80 preserves
  more local transition structure than the prior.
- `AS MapCorr = 0.596`: this is strong for this setting, and about `+53%`
  relative gain over the prior.
- `AS GradCorr = 0.445`: this is a meaningful edge/transition preservation
  score, with about `+49%` relative gain over the prior.

Do not compare these numbers directly to SSIM values like `0.90+`. They are
different metrics. `MapCorr` and especially `GradCorr` are deliberately more
punishing for local spatial shifts, smoothing, and heavy-tailed spread maps.

## Suggested Thesis Wording

`MapCorr` is a valid-pixel Pearson/normalized-correlation metric that measures
whether the predicted map reproduces the spatial high/low pattern of the ground
truth after removing global offset and scale.

`GradCorr` applies the same correlation idea to gradient-magnitude maps, making
it an edge/transition preservation metric. It is harsher than `MapCorr` because
small displacements or smoothing can decorrelate gradients even when RMSE
improves.

Together, these metrics show that Try80 improves not only absolute error but
also spatial structure: `MapCorr` and `GradCorr` improve for `PL`, `DS`, and
`AS`, including all environment classes in the full test split.
