# TFGThirdTry3 Implementation

`TFGThirdTry3` is the new path-loss-focused experiment branch.

It is intentionally based on `TFG_FirstTry1`, because the currently best saved base checkpoint for path loss comes from that branch:

- `TFG_FirstTry1 / cgan_unet_hdf5_amd_midvram`: best validation `path_loss.rmse_physical ~= 19.08 dB`
- `TFGSecondTry2 / cgan_unet_hdf5_amd_midvram`: best validation `path_loss.rmse_physical ~= 19.89 dB`

So `TFGThirdTry3` should be understood as:

- `TFG_FirstTry1` as the real base
- plus only the selected path-loss-specific features imported from `TFGSecondTry2`
- plus the new try-3 hybrid model work

## Goals

- Prioritize `path_loss.rmse_physical` over the other targets.
- Keep a clean path-loss-only baseline for ablations.
- Add a confidence head that can trigger heuristic fallback where the model is unreliable.

## Retained Imported Features

Only the path-loss-related additions from `TFGSecondTry2` are intentionally retained:

- linear path-loss target support with `predict_linear: true`
- path-loss saturation masking
- optional distance-from-center geometry channel
- path-loss-specific dB and linear metric reporting
- optional LoS-based path-loss correction heuristic

The branch is not meant to become a general merge of `TFGSecondTry2`. The working rule is:

- stay as close as possible to `TFG_FirstTry1`
- import only path-loss features that directly help the new try

## Main Design

### 1. Path-loss-only baseline

The baseline family predicts only `path_loss`:

- `target_columns: [path_loss]`
- `model.out_channels: 1`
- `target_metadata.path_loss.predict_linear: true`
- `data.path_loss_saturation_db: 175`

This gives a direct answer to the question: does a single-target model reduce path-loss error better than the shared model?

### 2. Hybrid path-loss model

The hybrid family predicts:

- channel 0: coarse `path_loss`
- channel 1: confidence logits

The hybrid output is still returned as a flat tensor so the surrounding pipeline stays simple.

## Hybrid Training Logic

The confidence target is generated on the fly from path-loss error:

- convert predicted and target `path_loss` to dB
- compute `abs(error_db)`
- mark pixels as high-confidence when `abs(error_db) <= confidence_error_threshold_db`
- train confidence with masked MSE on the confidence probability

That means:

- `confidence_prob = sigmoid(confidence_logits)`
- per-pixel loss is `(confidence_prob - confidence_target)^2`
- only valid path-loss pixels contribute through the path-loss mask

Masked MSE is used because it is simpler and more DirectML-friendly than BCE for the AMD setup.

Current default:

- `confidence_error_threshold_db: 8.0`
- `confidence_loss_weight: 0.3`

## Hybrid Inference Logic

At inference/evaluation time:

1. Predict coarse `path_loss`
2. Predict confidence map
3. Build a heuristic prior from:
   - regression heuristics
   - optional LoS path-loss correction
4. Use confidence to choose between coarse DL output and heuristic prior

Current fallback mode:

- `fallback_mode: replace`
- `fallback_threshold: 0.5`

That means:

- confidence >= 0.5 -> keep DL path loss
- confidence < 0.5 -> use heuristic path loss

## New Config Families

### Path-loss-only

- `configs/cgan_unet_hdf5_pathloss_only_amd_midvram.yaml`
- `configs/cgan_unet_hdf5_pathloss_only_amd_midvram_nogan.yaml`
- `configs/cgan_unet_hdf5_pathloss_only_amd_midvram_bigger.yaml`
- `configs/cgan_unet_hdf5_pathloss_only_amd_lowvram.yaml`
- `configs/cgan_unet_hdf5_pathloss_only_amd_max.yaml`
- `configs/cgan_unet_hdf5_pathloss_only_cuda_max.yaml`

### Path-loss hybrid

- `configs/cgan_unet_hdf5_pathloss_hybrid_amd_midvram.yaml`
- `configs/cgan_unet_hdf5_pathloss_hybrid_amd_midvram_bigger.yaml`
- `configs/cgan_unet_hdf5_pathloss_hybrid_amd_lowvram.yaml`
- `configs/cgan_unet_hdf5_pathloss_hybrid_amd_max.yaml`
- `configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max.yaml`

## Recommended First Runs

1. `cgan_unet_hdf5_pathloss_only_amd_midvram.yaml`
2. `cgan_unet_hdf5_pathloss_only_amd_midvram_nogan.yaml`
3. `cgan_unet_hdf5_pathloss_hybrid_amd_midvram.yaml`

## Key Files

- `data_utils.py`
- `model_unet.py`
- `model_cgan.py`
- `train_cgan.py`
- `evaluate_cgan.py`
- `predict_cgan.py`
- `heuristics_cgan.py`
