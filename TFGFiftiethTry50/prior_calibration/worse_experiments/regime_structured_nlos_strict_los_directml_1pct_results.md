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

- `raw_prior` overall: RMSE `52.7767 dB`, MAE `28.6022 dB`, count `4784130`
  `LoS`: RMSE `4.7071 dB`, MAE `3.7940 dB`, count `3364365`
  `NLoS`: RMSE `96.6089 dB`, MAE `87.3893 dB`, count `1419765`
- `global_affine` overall: RMSE `35.4365 dB`, MAE `28.0878 dB`, count `4784130`
  `LoS`: RMSE `22.8044 dB`, MAE `19.2907 dB`, count `3364365`
  `NLoS`: RMSE `54.7643 dB`, MAE `48.9339 dB`, count `1419765`
- `city_type_affine` overall: RMSE `34.4967 dB`, MAE `26.3822 dB`, count `4784130`
  `LoS`: RMSE `20.2314 dB`, MAE `16.9125 dB`, count `3364365`
  `NLoS`: RMSE `55.1366 dB`, MAE `48.8220 dB`, count `1419765`
- `city_type_los_affine` overall: RMSE `23.4393 dB`, MAE `12.3065 dB`, count `4784130`
  `LoS`: RMSE `4.5888 dB`, MAE `3.6862 dB`, count `3364365`
  `NLoS`: RMSE `42.4429 dB`, MAE `32.7338 dB`, count `1419765`
- `city_type_los_ant_affine` overall: RMSE `23.4553 dB`, MAE `12.2067 dB`, count `4784130`
  `LoS`: RMSE `4.4220 dB`, MAE `3.5796 dB`, count `3364365`
  `NLoS`: RMSE `42.5146 dB`, MAE `32.6499 dB`, count `1419765`
- `city_type_los_ant_quadratic` overall: RMSE `23.4217 dB`, MAE `12.0566 dB`, count `4784130`
  `LoS`: RMSE `4.2028 dB`, MAE `3.3789 dB`, count `3364365`
  `NLoS`: RMSE `42.5049 dB`, MAE `32.6198 dB`, count `1419765`
- `global_affine_support_scaled` overall: RMSE `40.9892 dB`, MAE `37.1865 dB`, count `4784130`
  `LoS`: RMSE `38.0593 dB`, MAE `35.3797 dB`, count `3364365`
  `NLoS`: RMSE `47.2117 dB`, MAE `41.4679 dB`, count `1419765`
- `city_type_los_affine_support_scaled` overall: RMSE `25.2538 dB`, MAE `9.4737 dB`, count `4784130`
  `LoS`: RMSE `4.5888 dB`, MAE `3.6862 dB`, count `3364365`
  `NLoS`: RMSE `45.8161 dB`, MAE `23.1882 dB`, count `1419765`
- `city_type_los_ant_affine_support_scaled` overall: RMSE `25.1261 dB`, MAE `9.4235 dB`, count `4784130`
  `LoS`: RMSE `4.4220 dB`, MAE `3.5796 dB`, count `3364365`
  `NLoS`: RMSE `45.6179 dB`, MAE `23.2714 dB`, count `1419765`
- `city_type_los_ant_quadratic_support_scaled` overall: RMSE `25.0996 dB`, MAE `9.2806 dB`, count `4784130`
  `LoS`: RMSE `4.2028 dB`, MAE `3.3789 dB`, count `3364365`
  `NLoS`: RMSE `45.6180 dB`, MAE `23.2654 dB`, count `1419765`

## Recommended Prior-Only System

- Best validation system: `city_type_los_ant_quadratic`
- Validation RMSE: `23.4217 dB`
- Validation `LoS` RMSE: `4.2028 dB`
- Validation `NLoS` RMSE: `42.5049 dB`

The systems compared are:

- `raw_prior`: direct formula map as-is
- `global_affine`: one train-only affine calibration for all valid pixels
- `city_type_affine`: one affine calibration per train-defined urban morphology class
- `city_type_los_affine`: one affine calibration per urban morphology class and pixel LoS/NLoS
- `city_type_los_ant_affine`: same as above, also split by antenna-height tertile
- `*_support_scaled`: multiply the calibrated prediction by the train-only positive-support rate of that regime
- `city_type_los_ant_quadratic`: train-only quadratic calibration per urban type, LoS/NLoS, and antenna-height tertile

This separation matters because many valid ground pixels have a target value of exactly `0 dB`, which can dominate the global RMSE.
