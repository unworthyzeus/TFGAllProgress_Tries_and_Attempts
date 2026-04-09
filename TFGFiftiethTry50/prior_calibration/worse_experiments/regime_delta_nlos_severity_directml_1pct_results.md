# Formula Prior Generalization Check

This report fits every calibration only on the training split and evaluates only on the validation split.
That avoids leaking validation information into the prior calibration step.

## Dataset

- Try folder: `C:\TFG\TFGpractice\TFGFiftiethTry50`
- Config: `C:\TFG\TFGpractice\TFGFiftiethTry50\experiments\fiftiethtry50_pmnet_prior_gan_fastbatch\fiftiethtry50_pmnet_prior_stage1_widen112_initial.yaml`
- Dataset: `C:\TFG\TFGpractice\Datasets\CKM_Dataset_270326.h5`

## Ground-truth Support

- Train valid pixels: `22284056`
- Train zero-valued valid pixels: `5930517` (26.61%)
- Val valid pixels: `4784130`
- Val zero-valued valid pixels: `1140463` (23.84%)

## Train-defined Regimes

- Density tertiles: `0.1557`, `0.3038`
- Height tertiles: `8.3770`, `14.8770`
- Antenna-height tertiles: `59.0929`, `114.8712`

## Validation Results

- `raw_prior` overall: RMSE `49.2202 dB`, MAE `27.0249 dB`, count `4784130`
  `LoS`: RMSE `4.7071 dB`, MAE `3.7940 dB`, count `3364365`
  `NLoS`: RMSE `90.0607 dB`, MAE `82.0745 dB`, count `1419765`
- `global_affine` overall: RMSE `40.6012 dB`, MAE `34.4693 dB`, count `4784130`
  `LoS`: RMSE `27.1741 dB`, MAE `23.8902 dB`, count `3364365`
  `NLoS`: RMSE `61.6840 dB`, MAE `59.5381 dB`, count `1419765`
- `city_type_affine` overall: RMSE `38.3195 dB`, MAE `30.8380 dB`, count `4784130`
  `LoS`: RMSE `23.8206 dB`, MAE `19.7465 dB`, count `3364365`
  `NLoS`: RMSE `60.0281 dB`, MAE `57.1212 dB`, count `1419765`
- `city_type_los_affine` overall: RMSE `23.4530 dB`, MAE `12.2791 dB`, count `4784130`
  `LoS`: RMSE `4.5888 dB`, MAE `3.6862 dB`, count `3364365`
  `NLoS`: RMSE `42.4684 dB`, MAE `32.6413 dB`, count `1419765`
- `city_type_los_ant_affine` overall: RMSE `23.4386 dB`, MAE `12.1321 dB`, count `4784130`
  `LoS`: RMSE `4.4220 dB`, MAE `3.5796 dB`, count `3364365`
  `NLoS`: RMSE `42.4835 dB`, MAE `32.3985 dB`, count `1419765`
- `city_type_los_ant_quadratic` overall: RMSE `23.4290 dB`, MAE `11.9827 dB`, count `4784130`
  `LoS`: RMSE `4.2028 dB`, MAE `3.3789 dB`, count `3364365`
  `NLoS`: RMSE `42.5184 dB`, MAE `32.3708 dB`, count `1419765`
- `delta_nlos_city_type_ant_affine` overall: RMSE `23.3279 dB`, MAE `12.4198 dB`, count `4784130`
  `LoS`: RMSE `4.7071 dB`, MAE `3.7940 dB`, count `3364365`
  `NLoS`: RMSE `42.2048 dB`, MAE `32.8599 dB`, count `1419765`
- `delta_nlos_city_type_ant_quadratic` overall: RMSE `23.3317 dB`, MAE `12.4131 dB`, count `4784130`
  `LoS`: RMSE `4.7071 dB`, MAE `3.7940 dB`, count `3364365`
  `NLoS`: RMSE `42.2119 dB`, MAE `32.8375 dB`, count `1419765`
- `delta_nlos_city_type_ant_severity_affine` overall: RMSE `23.3343 dB`, MAE `12.4396 dB`, count `4784130`
  `LoS`: RMSE `4.7071 dB`, MAE `3.7940 dB`, count `3364365`
  `NLoS`: RMSE `42.2166 dB`, MAE `32.9266 dB`, count `1419765`
- `delta_nlos_city_type_ant_severity_quadratic` overall: RMSE `23.3087 dB`, MAE `12.3688 dB`, count `4784130`
  `LoS`: RMSE `4.7071 dB`, MAE `3.7940 dB`, count `3364365`
  `NLoS`: RMSE `42.1690 dB`, MAE `32.6880 dB`, count `1419765`
- `global_affine_support_scaled` overall: RMSE `45.6490 dB`, MAE `43.7547 dB`, count `4784130`
  `LoS`: RMSE `43.5025 dB`, MAE `41.4295 dB`, count `3364365`
  `NLoS`: RMSE `50.3718 dB`, MAE `49.2649 dB`, count `1419765`
- `city_type_los_affine_support_scaled` overall: RMSE `25.2728 dB`, MAE `9.4682 dB`, count `4784130`
  `LoS`: RMSE `4.5888 dB`, MAE `3.6862 dB`, count `3364365`
  `NLoS`: RMSE `45.8515 dB`, MAE `23.1696 dB`, count `1419765`
- `city_type_los_ant_affine_support_scaled` overall: RMSE `25.1925 dB`, MAE `9.4081 dB`, count `4784130`
  `LoS`: RMSE `4.4220 dB`, MAE `3.5796 dB`, count `3364365`
  `NLoS`: RMSE `45.7412 dB`, MAE `23.2197 dB`, count `1419765`
- `city_type_los_ant_quadratic_support_scaled` overall: RMSE `25.1702 dB`, MAE `9.2657 dB`, count `4784130`
  `LoS`: RMSE `4.2028 dB`, MAE `3.3789 dB`, count `3364365`
  `NLoS`: RMSE `45.7489 dB`, MAE `23.2154 dB`, count `1419765`

## Recommended Prior-Only System

- Best validation system: `delta_nlos_city_type_ant_severity_quadratic`
- Validation RMSE: `23.3087 dB`
- Validation `LoS` RMSE: `4.7071 dB`
- Validation `NLoS` RMSE: `42.1690 dB`

The systems compared are:

- `raw_prior`: direct formula map as-is
- `global_affine`: one train-only affine calibration for all valid pixels
- `city_type_affine`: one affine calibration per train-defined urban morphology class
- `city_type_los_affine`: one affine calibration per urban morphology class and pixel LoS/NLoS
- `city_type_los_ant_affine`: same as above, also split by antenna-height tertile
- `*_support_scaled`: multiply the calibrated prediction by the train-only positive-support rate of that regime
- `city_type_los_ant_quadratic`: train-only quadratic calibration per urban type, LoS/NLoS, and antenna-height tertile

This separation matters because many valid ground pixels have a target value of exactly `0 dB`, which can dominate the global RMSE.
