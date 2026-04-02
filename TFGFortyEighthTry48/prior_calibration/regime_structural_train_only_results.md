# Formula Prior Generalization Check

This report fits every calibration only on the training split and evaluates only on the validation split.
That avoids leaking validation information into the prior calibration step.

## Dataset

- Try folder: `C:\TFG\TFGpractice\TFGFortyEighthTry48`
- Config: `C:\TFG\TFGpractice\TFGFortyEighthTry48\experiments\fortyeighthtry48_pmnet_prior_gan\fortyeighthtry48_pmnet_prior_gan.yaml`
- Dataset: `C:\TFG\TFGpractice\Datasets\CKM_Dataset_270326.h5`

## Ground-truth Support

- Train valid pixels: `2265154084`
- Train zero-valued valid pixels: `590665125` (26.08%)
- Val valid pixels: `485380287`
- Val zero-valued valid pixels: `124935435` (25.74%)

## Train-defined Regimes

- Density tertiles: `0.1957`, `0.2549`
- Height tertiles: `10.9137`, `15.9506`
- Antenna-height tertiles: `58.1218`, `103.8548`

## Validation Results

- `raw_prior`: RMSE `67.9603 dB`, MAE `43.4729 dB`, count `485380287`
- `global_affine`: RMSE `25.7045 dB`, MAE `15.9611 dB`, count `485380287`
- `city_type_affine`: RMSE `25.7014 dB`, MAE `15.9454 dB`, count `485380287`
- `city_type_los_affine`: RMSE `24.3649 dB`, MAE `13.0883 dB`, count `485380287`
- `city_type_los_ant_affine`: RMSE `24.1978 dB`, MAE `12.8049 dB`, count `485380287`
- `city_type_los_ant_quadratic`: RMSE `24.1777 dB`, MAE `12.7637 dB`, count `485380287`
- `global_affine_support_scaled`: RMSE `33.1522 dB`, MAE `27.6211 dB`, count `485380287`
- `city_type_los_affine_support_scaled`: RMSE `26.2347 dB`, MAE `9.8863 dB`, count `485380287`
- `city_type_los_ant_affine_support_scaled`: RMSE `26.1230 dB`, MAE `9.6922 dB`, count `485380287`
- `city_type_los_ant_quadratic_support_scaled`: RMSE `26.1129 dB`, MAE `9.6600 dB`, count `485380287`

## Recommended Prior-Only System

- Best validation system: `city_type_los_ant_quadratic`
- Validation RMSE: `24.1777 dB`

The systems compared are:

- `raw_prior`: direct formula map as-is
- `global_affine`: one train-only affine calibration for all valid pixels
- `city_type_affine`: one affine calibration per train-defined urban morphology class
- `city_type_los_affine`: one affine calibration per urban morphology class and pixel LoS/NLoS
- `city_type_los_ant_affine`: same as above, also split by antenna-height tertile
- `*_support_scaled`: multiply the calibrated prediction by the train-only positive-support rate of that regime
- `city_type_los_ant_quadratic`: train-only quadratic calibration per urban type, LoS/NLoS, and antenna-height tertile

This separation matters because many valid ground pixels have a target value of exactly `0 dB`, which can dominate the global RMSE.
