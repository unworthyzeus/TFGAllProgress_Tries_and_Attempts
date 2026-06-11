# All Quality Metric Comparison

Negative dRMSE is better; positive dSSIM/dMapCorr/dGradCorr are better.

## Global Overall: Values And Deltas

| Output | GT-prior RMSE | GT-model RMSE | dRMSE | GT-prior SSIM | GT-model SSIM | dSSIM | GT-prior MapCorr | GT-model MapCorr | dMapCorr | GT-prior GradCorr | GT-model GradCorr | dGradCorr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PL | 1.9383 | 1.6737 | -0.2646 | 0.916580 | 0.935623 | 0.019044 | 0.947710 | 0.961729 | 0.014019 | 0.791801 | 0.828308 | 0.036507 |
| DS | 28.1211 | 26.6888 | -1.4323 | 0.872754 | 0.853670 | -0.019085 | 0.306026 | 0.415563 | 0.109537 | 0.166344 | 0.249299 | 0.082955 |
| AS | 13.7416 | 11.4002 | -2.3414 | 0.664690 | 0.637383 | -0.027308 | 0.377700 | 0.593148 | 0.215448 | 0.231119 | 0.376801 | 0.145682 |

## Global Overall: Absolute Pair Metrics

| Pair | Output | RMSE | SSIM | MapCorr | GradCorr |
|---|---|---:|---:|---:|---:|
| GT-prior | PL | 1.9383 | 0.916580 | 0.947710 | 0.791801 |
| GT-prior | DS | 28.1211 | 0.872754 | 0.306026 | 0.166344 |
| GT-prior | AS | 13.7416 | 0.664690 | 0.377700 | 0.231119 |
| GT-model | PL | 1.6737 | 0.935623 | 0.961729 | 0.828308 |
| GT-model | DS | 26.6888 | 0.853670 | 0.415563 | 0.249299 |
| GT-model | AS | 11.4002 | 0.637383 | 0.593148 | 0.376801 |
| prior-model | PL | 1.0073 | 0.974498 | 0.984052 | 0.943794 |
| prior-model | DS | 5.0356 | 0.934904 | 0.727196 | 0.726820 |
| prior-model | AS | 5.4040 | 0.762589 | 0.687572 | 0.644948 |
