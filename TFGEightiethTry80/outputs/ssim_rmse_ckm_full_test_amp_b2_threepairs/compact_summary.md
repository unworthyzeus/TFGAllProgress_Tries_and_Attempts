# Try80 Test SSIM/RMSE Comparison

- Samples: 2590
- Elapsed: 2877.1s
- SSIM backend: skimage
- Mixed precision: True
- Checkpoint: `C:\TFG\CKMGenerator\models\best_model.pt`
- Calibrations: CKMGenerator calibration JSONs

## Global Overall

| Output | GT-prior RMSE | GT-prior SSIM | GT-model RMSE | GT-model SSIM | dRMSE model-prior | dSSIM model-prior | prior-model RMSE | prior-model SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PL | 1.9383 | 0.9720 | 1.6519 | 0.9803 | -0.2864 | 0.008242 | 1.0253 | 0.9868 |
| DS | 28.1023 | 0.8187 | 26.5572 | 0.7697 | -1.5451 | -0.049006 | 5.5903 | 0.8818 |
| AS | 13.7416 | 0.5908 | 11.3855 | 0.5505 | -2.3562 | -0.040231 | 5.5322 | 0.6888 |

## By Environment Class 6, Model Minus Prior

Negative dRMSE is better; positive dSSIM is better.

| Environment | Samples | PL dRMSE | PL dSSIM | DS dRMSE | DS dSSIM | AS dRMSE | AS dSSIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_block_highrise | 55 | -0.2027 | 0.005490 | -1.5335 | -0.065339 | -2.2326 | -0.059223 |
| dense_block_midrise | 505 | -0.2345 | 0.002941 | -1.3619 | -0.050609 | -2.7887 | -0.033455 |
| mixed_compact_lowrise | 670 | -0.3124 | 0.006458 | -0.7278 | -0.028223 | -2.6155 | -0.013220 |
| mixed_compact_midrise | 620 | -0.2639 | 0.008493 | -2.2660 | -0.074705 | -2.4891 | -0.060119 |
| open_sparse_lowrise | 520 | -0.3478 | 0.010399 | -0.5812 | -0.035526 | -1.7625 | -0.040371 |
| open_sparse_vertical | 220 | -0.3173 | 0.011933 | -2.7116 | -0.069640 | -1.8331 | -0.071495 |

## By Macro Environment Class 3, Model Minus Prior

| Macro environment | Samples | PL dRMSE | PL dSSIM | DS dRMSE | DS dSSIM | AS dRMSE | AS dSSIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 560 | -0.2307 | 0.003151 | -1.3666 | -0.052058 | -2.7359 | -0.035991 |
| mixed | 1290 | -0.2876 | 0.007362 | -1.6326 | -0.050206 | -2.5519 | -0.035400 |
| open | 740 | -0.3379 | 0.010853 | -1.5058 | -0.045662 | -1.7742 | -0.049619 |
