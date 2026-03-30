# Formula Prior Generalization Check

This report fits every calibration only on the training split and evaluates only on the validation split.
That avoids leaking validation information into the prior calibration step.

## Dataset

- Try folder: `C:\TFG\TFGpractice\TFGFortyFirstTry41`
- Config: `C:\TFG\TFGpractice\TFGFortyFirstTry41\experiments\fortyfirsttry41_prior_residual_formula\fortyfirsttry41_prior_residual_formula.yaml`
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

- `raw_prior`: RMSE `24.1602 dB`, MAE `12.6748 dB`, count `485380287`
- `global_affine`: RMSE `24.1602 dB`, MAE `12.6748 dB`, count `485380287`
- `city_type_affine`: RMSE `24.1602 dB`, MAE `12.6748 dB`, count `485380287`
- `city_type_los_affine`: RMSE `24.1602 dB`, MAE `12.6747 dB`, count `485380287`
- `city_type_los_ant_affine`: RMSE `24.1602 dB`, MAE `12.6748 dB`, count `485380287`
- `city_type_los_ant_quadratic`: RMSE `24.1600 dB`, MAE `12.6747 dB`, count `485380287`
- `global_affine_support_scaled`: RMSE `32.0272 dB`, MAE `26.6930 dB`, count `485380287`
- `city_type_los_affine_support_scaled`: RMSE `26.1246 dB`, MAE `9.5640 dB`, count `485380287`
- `city_type_los_ant_affine_support_scaled`: RMSE `26.0968 dB`, MAE `9.5715 dB`, count `485380287`
- `city_type_los_ant_quadratic_support_scaled`: RMSE `26.0968 dB`, MAE `9.5714 dB`, count `485380287`

## Recommended Prior-Only System

- Best validation system: `city_type_los_ant_quadratic`
- Validation RMSE: `24.1600 dB`

The systems compared are:

- `raw_prior`: direct formula map as-is
- `global_affine`: one train-only affine calibration for all valid pixels
- `city_type_affine`: one affine calibration per train-defined urban morphology class
- `city_type_los_affine`: one affine calibration per urban morphology class and pixel LoS/NLoS
- `city_type_los_ant_affine`: same as above, also split by antenna-height tertile
- `*_support_scaled`: multiply the calibrated prediction by the train-only positive-support rate of that regime
- `city_type_los_ant_quadratic`: train-only quadratic calibration per urban type, LoS/NLoS, and antenna-height tertile

This separation matters because many valid ground pixels have a target value of exactly `0 dB`, which can dominate the global RMSE.
