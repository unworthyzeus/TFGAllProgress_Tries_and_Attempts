# All Quality Metric Comparison

Negative dRMSE is better; positive dSSIM/dMapCorr/dGradCorr are better.

## Global Overall: Values And Deltas

| Output | GT-prior RMSE | GT-model RMSE | dRMSE | GT-prior SSIM | GT-model SSIM | dSSIM | GT-prior MapCorr | GT-model MapCorr | dMapCorr | GT-prior GradCorr | GT-model GradCorr | dGradCorr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PL | 1.9383 | 1.6519 | -0.2865 | 0.916580 | 0.937838 | 0.021259 | 0.947710 | 0.962556 | 0.014846 | 0.791801 | 0.830075 | 0.038274 |
| DS | 28.1023 | 26.5570 | -1.5453 | 0.872754 | 0.847016 | -0.025738 | 0.306040 | 0.415482 | 0.109442 | 0.166359 | 0.247311 | 0.080952 |
| AS | 13.7416 | 11.3854 | -2.3562 | 0.664690 | 0.625074 | -0.039616 | 0.377700 | 0.592630 | 0.214930 | 0.231119 | 0.376741 | 0.145622 |

## Global Overall: Absolute Pair Metrics

| Pair | Output | RMSE | SSIM | MapCorr | GradCorr |
|---|---|---:|---:|---:|---:|
| GT-prior | PL | 1.9383 | 0.916580 | 0.947710 | 0.791801 |
| GT-prior | DS | 28.1023 | 0.872754 | 0.306040 | 0.166359 |
| GT-prior | AS | 13.7416 | 0.664690 | 0.377700 | 0.231119 |
| GT-model | PL | 1.6519 | 0.937838 | 0.962556 | 0.830075 |
| GT-model | DS | 26.5570 | 0.847016 | 0.415482 | 0.247311 |
| GT-model | AS | 11.3854 | 0.625074 | 0.592630 | 0.376741 |
| prior-model | PL | 1.0253 | 0.973958 | 0.983720 | 0.943484 |
| prior-model | DS | 5.5917 | 0.924029 | 0.700643 | 0.715267 |
| prior-model | AS | 5.5331 | 0.747592 | 0.683472 | 0.647705 |
