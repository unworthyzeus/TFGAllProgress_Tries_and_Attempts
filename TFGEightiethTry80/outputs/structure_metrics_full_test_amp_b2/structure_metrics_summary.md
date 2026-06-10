# Structure Metric Comparison

Negative dRMSE is better; positive correlation deltas are better.

## Global Overall

| Output | dRMSE model-prior | dMapCorr model-prior | dGradCorr model-prior | prior-model RMSE | prior-model MapCorr | prior-model GradCorr |
|---|---:|---:|---:|---:|---:|---:|
| PL | -0.2864 | 0.014167 | 0.000905 | 1.0253 | 0.984185 | 0.933843 |
| DS | -1.5451 | 0.121090 | 0.052658 | 5.5903 | 0.761951 | 0.765841 |
| AS | -2.3562 | 0.206068 | 0.146383 | 5.5322 | 0.723492 | 0.728685 |

## By Environment Class 6

| Environment | Samples | PL dMapCorr | DS dMapCorr | AS dMapCorr | PL dGradCorr | DS dGradCorr | AS dGradCorr |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_block_highrise | 55 | 0.010428 | 0.080884 | 0.132473 | 0.000299 | 0.015480 | 0.085179 |
| dense_block_midrise | 505 | 0.013045 | 0.115091 | 0.207952 | 0.000399 | 0.053148 | 0.137421 |
| mixed_compact_lowrise | 670 | 0.017257 | 0.177000 | 0.242241 | 0.000810 | 0.079000 | 0.172352 |
| mixed_compact_midrise | 620 | 0.013049 | 0.100560 | 0.174837 | 0.000733 | 0.031945 | 0.112912 |
| open_sparse_lowrise | 520 | 0.013300 | 0.196540 | 0.244334 | 0.002009 | 0.075646 | 0.184838 |
| open_sparse_vertical | 220 | 0.010637 | 0.151391 | 0.180051 | 0.002782 | 0.035645 | 0.104298 |
