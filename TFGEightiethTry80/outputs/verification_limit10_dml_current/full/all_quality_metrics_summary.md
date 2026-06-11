# All Quality Metric Comparison

Negative dRMSE is better; positive dSSIM/dMapCorr/dGradCorr are better.

## Global Overall: Values And Deltas

| Output | GT-prior RMSE | GT-model RMSE | dRMSE | GT-prior SSIM | GT-model SSIM | dSSIM | GT-prior MapCorr | GT-model MapCorr | dMapCorr | GT-prior GradCorr | GT-model GradCorr | dGradCorr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PL | 1.7164 | 1.3346 | -0.3818 | 0.909284 | 0.940994 | 0.031710 | 0.956760 | 0.973882 | 0.017122 | 0.841768 | 0.886052 | 0.044284 |
| DS | 5.9166 | 5.4866 | -0.4300 | 0.865761 | 0.862231 | -0.003530 | 0.415962 | 0.559414 | 0.143453 | 0.204700 | 0.355923 | 0.151223 |
| AS | 12.1701 | 9.7891 | -2.3810 | 0.716798 | 0.742861 | 0.026064 | 0.350068 | 0.645114 | 0.295046 | 0.242088 | 0.412846 | 0.170758 |

## Global Overall: Absolute Pair Metrics

| Pair | Output | RMSE | SSIM | MapCorr | GradCorr |
|---|---|---:|---:|---:|---:|
| GT-prior | PL | 1.7164 | 0.909284 | 0.956760 | 0.841768 |
| GT-prior | DS | 5.9166 | 0.865761 | 0.415962 | 0.204700 |
| GT-prior | AS | 12.1701 | 0.716798 | 0.350068 | 0.242088 |
| GT-model | PL | 1.3346 | 0.940994 | 0.973882 | 0.886052 |
| GT-model | DS | 5.4866 | 0.862231 | 0.559414 | 0.355923 |
| GT-model | AS | 9.7891 | 0.742861 | 0.645114 | 0.412846 |
| prior-model | PL | 1.0522 | 0.962250 | 0.982120 | 0.942227 |
| prior-model | DS | 2.0410 | 0.924758 | 0.735981 | 0.638722 |
| prior-model | AS | 4.7194 | 0.801247 | 0.618295 | 0.536715 |
