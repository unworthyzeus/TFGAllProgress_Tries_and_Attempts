# Supervisor Update

## Context and current scope

I adapted the current prototype so it can train directly on the available HDF5 dataset `CKM_Dataset.h5`.

The dataset contains the following map-level targets:
- `delay_spread`
- `angular_spread`
- `path_loss`
- `los_mask`

It does **not** include:
- explicit antenna metadata inside the file
- varying bandwidth metadata inside the file
- direct `channel_power`
- direct `augmented_los`

The antenna is fixed at the center of the map and all samples are already aligned to the same antenna-centered spatial reference frame.

## Main technical work completed

### 1. Direct HDF5 training support

The codebase now supports two training modes:
- original manifest-based mode
- direct HDF5 mode

For HDF5 mode, the pipeline now uses:
- inputs: `topology_map`, `los_mask`
- targets: `delay_spread`, `angular_spread`, `path_loss`

This support is integrated into:
- `train.py`
- `train_cgan.py`
- `evaluate.py`
- `evaluate_cgan.py`
- `predict_cgan.py`

### 2. Native 513x513 support

The HDF5 dataset works at native `513 x 513` resolution and the model code was adapted accordingly.

The U-Net implementation was fixed so it can handle odd spatial dimensions correctly.

### 3. Local AMD and cluster support

I added:
- DirectML-based configs for local AMD GPU runs
- cluster SLURM scripts for remote execution
- automatic batch-size adaptation by GPU VRAM on cluster runs
- a multi-GPU launcher for cluster allocations that runs one independent training process per visible GPU

### 4. Checkpointing and resume

Training now supports automatic resume from previous checkpoints.

For new checkpoints, the resume path restores:
- model weights
- optimizer state
- scaler state
- epoch counter
- best validation value

### 5. Evaluation tooling

I added a dedicated cGAN evaluation script:
- `evaluate_cgan.py`

This script evaluates the generator on the validation split and reports both normalized errors and physical-unit errors.

## Current interpretation of the physical targets

### Path loss vs channel power

The dataset stores `path_loss`, not absolute `channel_power`.

Under fixed link-budget assumptions, `channel_power` can be derived from `path_loss` through:

$$
P_r = P_t + G_t + G_r - L_{path} - L_{other}
$$

Therefore, the key dB-domain quantity to learn correctly is `path_loss`.

If `path_loss` is predicted poorly, any derived `channel_power` map will also be poor by essentially the same dB error.

### LoS semantics

The HDF5 route now uses ground-truth `los_mask` as a trusted binary LoS input channel.

This means the model no longer predicts LoS in HDF5 mode. Instead, it uses LoS as prior information while predicting the three regression maps.

## Heuristics currently implemented

At inference time, the current pipeline applies two levels of heuristics.

### 1. Conservative sanity heuristics

These include:
- physical clipping for regression targets
- median filtering for regression maps

### 2. Physics-aware derived heuristics

I also added derived-map post-processing based on explicit formulas:
- derived `channel_power` from predicted `path_loss`
- optional derived SNR maps if bandwidth is configured
- optional derived link-availability map if a reception threshold is configured

These are implemented as post-processing, so they do not require retraining existing checkpoints.

## Current validation result

I evaluated the existing checkpoint:
- `outputs/cgan_unet_hdf5_amd_midvram/best_cgan.pt`

Important caveat:
- this checkpoint is still very early, at epoch `2`

Current validation metrics in physical units are:

- `delay_spread`: MSE `2376.08 ns^2`, which implies RMSE about `48.7 ns`
- `angular_spread`: MSE `183.65 deg^2`, which implies RMSE about `13.6 deg`
- `path_loss`: MSE `1128.73 dB^2`, which implies RMSE about `33.6 dB`

## Interpretation of the current result

### Positive points

- `delay_spread` is already close to the target range stated in the proposal
- `angular_spread` is already comfortably within the intended target range
- the end-to-end pipeline is operational for training, evaluation, inference, checkpointing, and cluster execution

### Main weakness right now

- `path_loss` is still far from the desired dB accuracy

This is the main bottleneck because the dB-domain interpretation of the system depends on `path_loss` being good.

## Recent changes specifically to improve path loss

To prioritize the dB-domain target, I changed the training and inference pipeline so that:

- `path_loss` now receives a higher target loss weight during training
- `path_loss` now has a dedicated median-filter setting at inference time

This means the training objective is now more aligned with the thesis requirement that the dB-domain quantity should be modeled accurately.

## Honest project status

At this moment, the work supports the following claim:

> The prototype already works as a complete CKM prediction pipeline on the currently available HDF5 dataset, and it learns `delay_spread` and `angular_spread` reasonably well. The remaining major challenge is improving `path_loss`, which is the key quantity required to support a physically meaningful dB-domain interpretation and a derived `channel_power` result.

What I would **not** claim yet:

- that the final dB target has been achieved
- that the current HDF5 route fully matches the original `channel_power + augmented_los + varying UAV height` formulation without qualification

## Recommended next steps

1. Continue training longer with the updated path-loss-prioritized configs.
2. Re-evaluate after a significantly later checkpoint, not only epoch 2.
3. Compare cGAN and plain U-Net specifically on `path_loss` physical RMSE.
4. If needed, tune loss weights and post-processing further for the dB-domain target.
5. Present the current result as a strong intermediate milestone: the pipeline is operational and two of the three main physical targets already look promising, while `path_loss` remains the primary optimization focus.