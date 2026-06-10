# Rerun Required: SSIM Mask Fix And Combined Metrics

Created: 2026-06-10

The SSIM/RMSE files in this folder were generated before the SSIM masking fix.
Treat these files as stale until they are regenerated:

- `prior_vs_try80_ssim_rmse_summary.json`
- `prior_vs_try80_ssim_rmse_summary.csv`
- `prior_vs_try80_ssim_rmse_per_sample.csv`
- `compact_global_and_environment_summary.csv`
- `compact_summary.md`
- Any thesis/report text copied from those numbers

## What Was Wrong

1. SSIM filled invalid prediction pixels with target values before computing the
   SSIM map. That can inflate SSIM near no-data boundaries because the local
   SSIM window sees artificial perfect-match pixels.

2. Non-finite values were converted to zero for SSIM. That can hide corrupted
   values and introduce artificial zero-valued pixels.

## What The Fix Does

1. SSIM is computed with a mask-aware local window for each task.

2. Local SSIM means, variances, and covariances use only finite task-valid
   pixels. No-data pixels are excluded from the local window.

3. LoS/NLoS scopes select center pixels for aggregation; they are not treated as
   SSIM-window boundaries.

4. Finite center pixels with fewer than two finite task-valid window pixels are
   excluded from SSIM aggregation.

5. RMSE is still pixel-weighted RMSE over valid pixels, with the safety
   improvement that non-finite paired pixels are excluded.

## Correct Combined Script

Use the combined evaluator for the final numbers:

`C:\TFG\TFGAllProgress_Tries_and_Attempts\TFGEightiethTry80\scripts\compare_prior_try80_all_quality_metrics.py`

It calculates RMSE, SSIM, MapCorr, and GradCorr together for:

- GT-prior
- GT-model
- prior-model

The combined output includes both absolute values and deltas. In particular,
`all_quality_metrics_model_prior_comparison.csv` contains GT-prior values,
GT-model values, model-minus-prior deltas, and prior-model values for all four
metrics.

## Rerun Command

```powershell
cd C:\TFG\TFGAllProgress_Tries_and_Attempts\TFGEightiethTry80
python scripts\compare_prior_try80_all_quality_metrics.py `
  --config C:\TFG\TFGAllProgress_Tries_and_Attempts\TFGEightiethTry80\experiments\try80_joint_huge_pathloss_finetune.yaml `
  --checkpoint C:\TFG\CKMGenerator\models\best_model.pt `
  --out-dir C:\TFG\TFGAllProgress_Tries_and_Attempts\TFGEightiethTry80\outputs\all_quality_metrics_full_test_amp_b2 `
  --split test `
  --hdf5-path C:\TFG\TFGAllProgress_Tries_and_Attempts\Datasets\CKM_Dataset_270326.h5 `
  --try78-los-calibration-json C:\TFG\CKMGenerator\calibrations\try78_los_two_ray_calibration.json `
  --try78-nlos-calibration-json C:\TFG\CKMGenerator\calibrations\try78_nlos_regime_calibration.json `
  --try79-calibration-json C:\TFG\CKMGenerator\calibrations\try79_calibration.json `
  --batch-size 2 `
  --mixed-precision
```

For DirectML/AMD, add `--device directml`. Mixed precision is only used on
CUDA, so DirectML runs should normally omit `--mixed-precision`.
