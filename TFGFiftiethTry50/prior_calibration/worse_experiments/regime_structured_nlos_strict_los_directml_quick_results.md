# Formula Prior Generalization Check

This report fits every calibration only on the training split and evaluates only on the validation split.
That avoids leaking validation information into the prior calibration step.

## Dataset

- Try folder: `C:\TFG\TFGpractice\TFGFiftiethTry50`
- Config: `C:\TFG\TFGpractice\TFGFiftiethTry50\experiments\fiftiethtry50_pmnet_prior_gan_fastbatch\fiftiethtry50_pmnet_prior_stage1_widen112_initial.yaml`
- Dataset: `C:\TFG\TFGpractice\Datasets\CKM_Dataset_270326.h5`

## Ground-truth Support

- Train valid pixels: `225478079`
- Train zero-valued valid pixels: `60431740` (26.80%)
- Val valid pixels: `48655972`
- Val zero-valued valid pixels: `12260388` (25.20%)

## Train-defined Regimes

- Density tertiles: `0.1856`, `0.2928`
- Height tertiles: `11.2424`, `17.1636`
- Antenna-height tertiles: `56.9514`, `105.8016`

## Validation Results

- `raw_prior`: RMSE `54.2848 dB`, MAE `29.9629 dB`, count `48655972`
- `global_affine`: RMSE `35.0464 dB`, MAE `27.6012 dB`, count `48655972`
- `city_type_affine`: RMSE `34.9336 dB`, MAE `27.3789 dB`, count `48655972`
- `city_type_los_affine`: RMSE `24.3348 dB`, MAE `13.0583 dB`, count `48655972`
- `city_type_los_ant_affine`: RMSE `24.2264 dB`, MAE `12.9831 dB`, count `48655972`
- `city_type_los_ant_quadratic`: RMSE `24.1927 dB`, MAE `12.8422 dB`, count `48655972`
- `global_affine_support_scaled`: RMSE `40.6441 dB`, MAE `37.0240 dB`, count `48655972`
- `city_type_los_affine_support_scaled`: RMSE `26.3919 dB`, MAE `10.0301 dB`, count `48655972`
- `city_type_los_ant_affine_support_scaled`: RMSE `26.2977 dB`, MAE `9.9918 dB`, count `48655972`
- `city_type_los_ant_quadratic_support_scaled`: RMSE `26.2705 dB`, MAE `9.8515 dB`, count `48655972`

## Recommended Prior-Only System

- Best validation system: `city_type_los_ant_quadratic`
- Validation RMSE: `24.1927 dB`

The systems compared are:

- `raw_prior`: direct formula map as-is
- `global_affine`: one train-only affine calibration for all valid pixels
- `city_type_affine`: one affine calibration per train-defined urban morphology class
- `city_type_los_affine`: one affine calibration per urban morphology class and pixel LoS/NLoS
- `city_type_los_ant_affine`: same as above, also split by antenna-height tertile
- `*_support_scaled`: multiply the calibrated prediction by the train-only positive-support rate of that regime
- `city_type_los_ant_quadratic`: train-only quadratic calibration per urban type, LoS/NLoS, and antenna-height tertile

This separation matters because many valid ground pixels have a target value of exactly `0 dB`, which can dominate the global RMSE.
