# Try80 Best Model Comparison

Source: `deep_breakdown_try80_best_model_all_and_unseen_test.json`. Negative delta means the model is better than the frozen prior.

## Split Comparison, Overall

| split | n | PL dRMSE | PL dMAE | DS dRMSE | DS dMAE | AS dRMSE | AS dMAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 10840 | -0.2895 | -0.3030 | -2.0157 | 0.6135 | -2.3578 | 0.0108 |
| val | 2750 | -0.2758 | -0.2865 | -1.3433 | 0.5869 | -2.4396 | -0.0413 |
| test | 2590 | -0.2865 | -0.3020 | -1.5453 | 0.6864 | -2.3562 | 0.0198 |
| all | 16180 | -0.2867 | -0.3000 | -1.8406 | 0.6214 | -2.3710 | 0.0036 |

## Delay Spread by Scope

| split | scope | model_rmse | prior_rmse | d_rmse | model_mae | prior_mae | d_mae |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | overall | 28.7929 | 30.8086 | -2.0157 | 9.0696 | 8.4561 | 0.6135 |
| train | los | 28.7144 | 31.5063 | -2.7919 | 11.7089 | 11.2575 | 0.4515 |
| train | nlos | 28.9429 | 29.4242 | -0.4813 | 4.0070 | 3.0828 | 0.9242 |
| val | overall | 22.1061 | 23.4494 | -1.3433 | 6.2994 | 5.7124 | 0.5869 |
| val | los | 21.2027 | 22.9962 | -1.7935 | 7.6035 | 7.0733 | 0.5302 |
| val | nlos | 24.1815 | 24.5268 | -0.3452 | 3.0975 | 2.3713 | 0.7262 |
| test | overall | 26.5570 | 28.1023 | -1.5453 | 7.9897 | 7.3033 | 0.6864 |
| test | los | 25.4419 | 27.4851 | -2.0432 | 9.5210 | 8.9415 | 0.5795 |
| test | nlos | 29.2045 | 29.6157 | -0.4112 | 4.0907 | 3.1320 | 0.9587 |
| all | overall | 27.4121 | 29.2527 | -1.8406 | 8.4257 | 7.8043 | 0.6214 |
| all | los | 26.9653 | 29.4780 | -2.5127 | 10.5983 | 10.1101 | 0.4882 |
| all | nlos | 28.3236 | 28.7760 | -0.4525 | 3.8838 | 2.9840 | 0.8998 |

## Delay Spread by Global Subexpert

| subexpert | train dRMSE | train dMAE | val dRMSE | val dMAE | test dRMSE | test dMAE | all dRMSE | all dMAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense|high_ant | -0.7010 | 0.0893 | -0.4782 | -0.0421 | -0.4883 | 0.0715 | -0.6473 | 0.0638 |
| dense|low_ant | -1.6373 | 0.4895 | -1.2134 | 0.6786 | -1.7282 | 0.7432 | -1.5779 | 0.5463 |
| dense|mid_ant | -1.4582 | 0.6042 | -1.1263 | 0.4027 | -1.4765 | 1.0397 | -1.4074 | 0.6147 |
| mixed|high_ant | -0.7605 | 0.2840 | -0.3205 | 0.0882 | -0.6731 | 0.2326 | -0.6794 | 0.2382 |
| mixed|low_ant | -3.1585 | 0.7375 | -2.3129 | 0.9625 | -2.1797 | 1.0009 | -2.8379 | 0.8299 |
| mixed|mid_ant | -2.2837 | 0.9944 | -1.4449 | 0.7281 | -1.7314 | 0.9660 | -2.0522 | 0.9436 |
| open|high_ant | -0.6704 | 0.1672 | -0.2788 | 0.0568 | -0.1679 | 0.0276 | -0.5428 | 0.1224 |
| open|low_ant | -3.4866 | 1.1297 | -2.0973 | 1.8909 | -2.3422 | 1.4037 | -3.1170 | 1.2934 |
| open|mid_ant | -1.7614 | 1.0091 | -0.9900 | 0.7620 | -1.2380 | 0.5584 | -1.5789 | 0.8869 |

## Worst Test Cities for Delay Spread MAE Delta

| city | model_rmse | prior_rmse | d_rmse | model_mae | prior_mae | d_mae |
| --- | --- | --- | --- | --- | --- | --- |
| Beijing | 23.2843 | 24.1400 | -0.8557 | 8.7301 | 6.9803 | 1.7498 |
| Barcelona | 32.9650 | 34.3271 | -1.3620 | 12.1206 | 10.9012 | 1.2195 |
| Key West | 19.9495 | 20.8658 | -0.9163 | 7.4173 | 6.2044 | 1.2129 |
| Osaka | 37.2518 | 38.9156 | -1.6638 | 12.2397 | 11.2167 | 1.0231 |
| Surat | 23.4494 | 24.2288 | -0.7794 | 6.8760 | 5.9412 | 0.9348 |
| Kuala Lumpur | 44.0028 | 46.9254 | -2.9226 | 17.0066 | 16.2349 | 0.7717 |
| Jaipur | 21.5521 | 22.6629 | -1.1107 | 5.9860 | 5.2213 | 0.7646 |
| Pune | 32.9067 | 34.5571 | -1.6504 | 10.6864 | 9.9994 | 0.6870 |
| Johannesburg | 20.9019 | 22.3865 | -1.4846 | 6.5540 | 5.8930 | 0.6610 |
| Phuket | 36.9039 | 39.6675 | -2.7636 | 14.4692 | 13.9292 | 0.5401 |

## Worst Validation Cities for Delay Spread MAE Delta

| city | model_rmse | prior_rmse | d_rmse | model_mae | prior_mae | d_mae |
| --- | --- | --- | --- | --- | --- | --- |
| Dubrovnik | 23.4290 | 24.4490 | -1.0200 | 8.0449 | 6.5924 | 1.4525 |
| Nice | 20.2841 | 21.1564 | -0.8723 | 6.6084 | 5.5266 | 1.0818 |
| Segovia | 14.3195 | 14.5254 | -0.2059 | 4.6344 | 3.6064 | 1.0279 |
| Nazareth | 26.2719 | 27.2400 | -0.9682 | 8.3878 | 7.4707 | 0.9171 |
| Cleveland | 19.7223 | 20.5934 | -0.8711 | 6.0449 | 5.2386 | 0.8063 |
| Bilbao | 41.4483 | 43.7409 | -2.2926 | 14.6948 | 13.9120 | 0.7828 |
| Dhaka | 36.0585 | 39.3156 | -3.2571 | 14.1436 | 13.4141 | 0.7295 |
| Mexico City | 21.8190 | 23.0395 | -1.2205 | 5.9915 | 5.2895 | 0.7020 |
| Chiang Mai | 38.6230 | 41.1233 | -2.5004 | 14.5426 | 13.9022 | 0.6404 |
| Marrakesh | 16.9633 | 17.6455 | -0.6822 | 4.3498 | 3.7127 | 0.6371 |

## Reading

- Path loss is stable: train/val/test all improve RMSE and MAE versus the prior.

- Delay spread consistently improves RMSE but worsens MAE, including train. This points to a training objective/tradeoff rather than a pure unseen-city failure.

- Angular spread strongly improves RMSE. MAE is nearly neutral overall, improves on LoS, and worsens on NLoS.
