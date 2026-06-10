# Rerun Required: MapCorr and GradCorr Fix

Created: 2026-06-10

The structure metric files in this folder were generated before the
MapCorr/GradCorr metric fix in:

`C:\TFG\TFGAllProgress_Tries_and_Attempts\TFGEightiethTry80\scripts\compare_prior_try80_structure_metrics.py`

Treat these existing generated files as stale until the calculation is rerun:

- `structure_metrics_summary.json`
- `structure_metrics_summary.csv`
- `structure_metrics_summary.md`
- `gt_prior_vs_gt_model_structure_comparison.csv`
- `gt_prior_vs_gt_model_structure_comparison.md`
- Any thesis/report text copied from those numbers

## What Was Wrong

1. The aggregate MapCorr was computed as one pooled Pearson correlation over all
   raw valid pixels in the group. That is a valid statistic, but it did not
   match the written explanation that each map/scope is z-scored independently
   before comparison.

2. GradCorr computed gradients on full arrays before applying the requested
   valid/LoS/NLoS mask. This allowed no-data, LoS/NLoS, or other mask
   boundaries to influence gradient magnitudes at valid pixels.

3. For GradCorr, invalid prediction pixels were filled from the reference map
   before gradient calculation. This could artificially make target and
   prediction gradients agree near mask boundaries.

4. Non-finite target or prediction values were converted to zero. That can hide
   corrupted values and introduce artificial zero pixels into RMSE and
   correlation statistics.

## What The Fix Does

1. MapCorr is now computed per sample and per scope over finite valid pixels,
   which is equivalent to z-scoring each map/scope independently. Group-level
   MapCorr is the valid-pixel-weighted mean of those per-map correlations.

2. GradCorr is now computed per sample and per scope on mask-aware gradient
   magnitudes. A gradient at a pixel only uses same-mask finite neighbor pixels.
   It no longer crosses no-data, LoS/NLoS, or scope boundaries.

3. Non-finite paired pixels are excluded instead of being converted to zero.

## Why The Calculation Must Be Rerun

The old numeric summaries cannot be corrected by editing the CSV or Markdown
after the fact. The model, prior, target, and mask arrays must be passed through
the fixed metric code because:

- the MapCorr aggregation changed from pooled raw-pixel Pearson correlation to
  per-map z-normalized correlation aggregation;
- the GradCorr valid pixel set can change near mask boundaries;
- gradient magnitudes themselves can change because invalid neighbors are no
  longer used;
- non-finite values are now dropped instead of zero-filled.

## Rerun Command

Use a new output directory first, compare the regenerated numbers, then replace
the stale files once accepted:

```powershell
cd C:\TFG\TFGAllProgress_Tries_and_Attempts\TFGEightiethTry80
python scripts\compare_prior_try80_structure_metrics.py `
  --config C:\TFG\TFGAllProgress_Tries_and_Attempts\TFGEightiethTry80\experiments\try80_joint_huge_pathloss_finetune.yaml `
  --checkpoint C:\TFG\CKMGenerator\models\best_model.pt `
  --out-dir C:\TFG\TFGAllProgress_Tries_and_Attempts\TFGEightiethTry80\outputs\structure_metrics_full_test_amp_b2_rerun `
  --split test `
  --hdf5-path C:\TFG\TFGAllProgress_Tries_and_Attempts\Datasets\CKM_Dataset_270326.h5 `
  --try78-los-calibration-json C:\TFG\CKMGenerator\calibrations\try78_los_two_ray_calibration.json `
  --try78-nlos-calibration-json C:\TFG\CKMGenerator\calibrations\try78_nlos_regime_calibration.json `
  --try79-calibration-json C:\TFG\CKMGenerator\calibrations\try79_calibration.json `
  --batch-size 2 `
  --mixed-precision
```
