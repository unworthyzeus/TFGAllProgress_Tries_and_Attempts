# IMPROVEMENTS V2

Status of each item from the original [IMPROVEMENTS.md](IMPROVEMENTS.md), plus additional work done.

**Target**: Path loss RMSE <= 5 dB (current ~19.7 dB)

**Important clarification**:
- The antenna is a single point and is always at the center.
- The intended `2->3` change refers to the **model output**, not the input.
- The extra output is a **confidence map** for path loss, not a distance-map input channel.

---

## DONE

### 1. Path loss in linear scale

> "predict the path loss linearly, converting it to linear before training... the loss will be higher, it will be more varied data"

- **Status**: Done
- **Config**: `target_metadata.path_loss.predict_linear: true`
- **Implementation**: Loader converts dB → linear → log-normalized [0,1]; loss is MSE on that; eval converts back to dB
- **Details**: [CHANGES_PATHLOSS_IMPROVEMENTS.md](CHANGES_PATHLOSS_IMPROVEMENTS.md)

### 2. Normalization fix (no dataset min/max)

> "I'm normalizing the values with the highest and minimum of the dataset. Maybe not doing it that way, if the value goes higher than the highest of the dataset can give problems"

- **Status**: Done (for path loss)
- **Implementation**: Path loss uses fixed physical bounds (0–180 dB) via log-scale normalization, not dataset min/max
- **Details**: [CHANGES_PATHLOSS_IMPROVEMENTS.md](CHANGES_PATHLOSS_IMPROVEMENTS.md)

### 3. Path loss data missing / saturation

> "Path loss is data missing because sometimes it was too high (so no signal). This affected delay spread and angular spread"

- **Status**: Done
- **Config**: `data.path_loss_saturation_db: 175`
- **Implementation**: Pixels with path_loss >= 175 dB are masked out during training (mask = 0)

### 4. Bigger model

> "Train a BIGGER model, with more parameters"

- **Status**: Done (config ready)
- **Config**: `configs/cgan_unet_hdf5_pathloss_focus.yaml` (base_channels 96, disc 96)
- **Usage**: On cluster with `sbatch --export=ALL,CONFIG_PATH=configs/cgan_unet_hdf5_pathloss_focus.yaml cluster/run_train_cgan_hdf5.slurm`

### 5. Optional distance-from-center geometry input

> (From papers: distance-to-Tx improves path loss prediction)

- **Status**: Implemented, disabled for initial runs
- **Config**: `data.distance_map_channel: false` (set to `true` when ready to retrain with it)
- **Implementation**: 2D horizontal distance from map center (antenna is always at center), normalized [0,1]
- **Note**: This is geometry (distance in meters), NOT a confidence map
- **Note**: This is an optional experiment only and is **not** the requested `2->3` confidence design

### 6. LoS-aware path loss correction (formula heuristic)

> (From IMPROVEMENTS: path loss exponent decreases with height; we use LoS as proxy)

- **Status**: Implemented, optional
- **Config**: `postprocess.path_loss_los_correction: false`
- **Formula**: Free-space `PL_fs = 20*log10(d) + 20*log10(f) + 92.45`; blends with prediction in LoS regions
- **Use**: `predict_cgan.py` only; enable when LoS input is available

### 7. Dual error reporting (linear + dB)

- **Status**: Done
- **Implementation**: Evaluation reports path loss error in three spaces:
  - `mse`, `rmse`, `mae`: normalized model space
  - `mse_physical`, `rmse_physical`, `mae_physical`: dB
  - `mse_linear`, `rmse_linear`, `mae_linear`: linear ratio space (`P_rx / P_tx`, unitless)

### 8. Multi-GPU cluster scripts

- **Status**: Done
- **Files**: `cluster/run_train_cgan_hdf5_4gpu.slurm`, `cluster/run_train_cgan_hdf5_8gpu.slurm`

### 9. Metadata inspection

- **Status**: Done
- **File**: `inspect_hdf5_metadata.py`
- **Result**: No antenna height in HDF5; Vinogradov formula cannot be applied without it

---

## TODO / NOT DONE

### 1. Vinogradov height-dependent path loss exponent

> "Postprocessing: The path loss exponent decreases with higher height. Maybe doing some checking algorithm"

- **Status**: Blocked
- **Reason**: Antenna height is not in HDF5 metadata (verified with `inspect_hdf5_metadata.py`)
- **Formula** (2511.10763): `n(h) = n_inf + (n_0 - n_inf)*exp(-h/h_0)`
- **Next step**: Add antenna height to HDF5 when generating the dataset, or obtain it from another source

### 2. City detection / per-city models

> "Preprocessing: City detection, because the path loss formula depends on type of cities. Maybe with the number or height of buildings. And distribution, percentage of it. Train different models depending on type of city and classify the image before processing it"

- **Status**: Not started
- **Complexity**: High (city classifier + per-city models)
- **Next step**: Define city types (e.g. suburban, urban, dense, high-rise); train classifier on topology; train separate models per type

### 3. Confidence map + heuristic fallback (DL + algorithmic hybrid)

> "Giving me the confidence of the path loss based on the error by groundtruth... Where the model fails, or gets a very wrong value, put low confidence (after getting the results from the model). And then we train again with the ground truth of confidence (based on errors), predicting the confidence, and if the confidence is low, path loss is calculated algorithmically"

- **Status**: Not started
- **Concept**: Add a third **output** channel for a per-pixel **confidence map** (not a distance map):
  - High error region → low confidence → use heuristic (algorithmic) path loss
  - Low error region → high confidence → use DL prediction
- **Architecture**:
  - Current idea: output channels `2 -> 3`
  - The extra output channel is `confidence_pred`
  - This does **not** require adding a distance-map input channel
- **Ground truth needed**: There is no direct confidence label in the dataset, so confidence target must be built from path loss error against ground truth
- **Suggested flow**:
  - Stage 1: train the normal model for path loss
  - Stage 2: compare prediction vs ground truth and build a confidence/error target map
  - Stage 3: retrain or extend the model to predict both path loss and confidence
  - Inference: where confidence is low, replace DL path loss with heuristic path loss
- **Loss**: Separate confidence loss on the confidence target matrix, in addition to the path loss loss
- **Complexity**: High (two-stage: confidence target generation + confidence prediction + algorithmic fallback)
- **Next step**: Define confidence target from path loss error; add confidence output head/channel; choose confidence loss and low-confidence threshold; implement heuristic path loss fallback

### 4. Optional distance-from-center geometry experiment

- **Status**: Pending (code ready, config has `distance_map_channel: false`)
- **Note**: This is geometry (distance from antenna at center), separate from the confidence map (item 3)
- **Note**: This is **not required** for the confidence-based hybrid approach
- **Next step**: Only test this if you want a separate geometry experiment; enabling it would change the **input** channels, but this is unrelated to the confidence-map output design

### 5. Enable LoS path loss correction and evaluate

- **Status**: Pending (code ready, config has `path_loss_los_correction: false`)
- **Next step**: Set `path_loss_los_correction: true` in predict config; test if it improves path loss in LoS regions

### 6. Train on cluster and validate RMSE

- **Status**: Pending
- **Next step**: Run `cgan_unet_hdf5_pathloss_focus.yaml` on cluster; check if path_loss RMSE <= 5 dB

---

## Summary

| Category        | Done | Pending / Blocked |
|-----------------|------|-------------------|
| Data / loader   | 4    | 0 |
| Model / training| 2    | 1 (cluster run) |
| Postprocessing  | 2    | 2 (LoS correction, Vinogradov) |
| Preprocessing   | 0    | 1 (city detection) |
| Hybrid / confidence | 0 | 1 (confidence-output + fallback) |
| Optional experiments | 1 | 1 (distance-map geometry input) |
