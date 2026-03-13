# Path Loss RMSE Improvement Changes

Summary of all changes implemented to reduce path loss RMSE from ~19.7 dB toward the target of <= 5 dB.

---

## 1. Path Loss Training in Linear Scale

**Yes.** Path loss is trained in linear scale when `target_metadata.path_loss.predict_linear: true` (enabled in both `cgan_unet_hdf5_amd_midvram.yaml` and `cgan_unet_hdf5_pathloss_focus.yaml`).

### How it works

- **Loader** (`data_utils.py`): When loading path_loss target with `predict_linear: true`, raw dB values are converted to linear (power ratio) and then normalized:
  - `linear = 10^(-path_loss_db/10)` (clipped to [1e-18, 1])
  - `log_linear = log10(linear)` (range [-18, 0] for 0–180 dB)
  - `normalized = (log_linear + 18) / 18` (range [0, 1])

- **Loss**: MSE is computed on these normalized linear values.

- **Evaluation/Export**: Predicted normalized values are converted back to dB:
  - `log_linear = normalized * 18 - 18`
  - `linear = 10^log_linear`
  - `path_loss_db = -10 * log10(linear)`

### Rationale (from IMPROVEMENTS.md)

> "predict the path loss linearly, converting it to linear before training... the loss will be higher, it will be more varied data."

MSE in linear (power ratio) space penalizes dB errors more appropriately across the path loss range than MSE on raw dB.

---

## 2. Normalization Fix

### Previous approach (problematic)

- Path loss was normalized with `scale: 180`, `offset: 0` → `normalized = path_loss_db / 180`
- Values outside [0, 180] dB could fall outside [0, 1]
- IMPROVEMENTS.md: "I'm normalizing the values with the highest and minimum of the dataset. Maybe not doing it that way, if the value goes higher than the highest of the dataset can give problems."

### New approach for path loss (when `predict_linear: true`)

1. **Physical bounds**: Linear power ratio is bounded by physics:
   - 0 dB → linear = 1
   - 180 dB → linear = 10^(-18) = 1e-18

2. **Log-scale normalization**: Instead of `value / 180`, we use:
   - `linear = 10^(-dB/10)` ∈ [1e-18, 1]
   - `normalized = (log10(linear) + 18) / 18` ∈ [0, 1]
   - This maps the full physical range to [0, 1] without dataset min/max.

3. **No dataset min/max**: Normalization is based on fixed physical limits (0–180 dB), not on dataset statistics.

### Other targets (unchanged)

- `delay_spread`: `scale: 1000`, `offset: 0` (ns)
- `angular_spread`: `scale: 180`, `offset: 0` (deg)
- These still use simple `(value - offset) / scale`.

---

## 3. Dual Error Reporting (Linear + dB)

When `predict_linear: true`, evaluation reports metrics in multiple domains:

| Metric | Unit | Description |
|--------|------|-------------|
| `mse`, `rmse`, `mae` | normalized_0_1 | Error in model output space (normalized target used for training) |
| `mse_physical`, `rmse_physical`, `mae_physical` | dB | Error in dB (physical units) |
| `mse_linear`, `rmse_linear`, `mae_linear` | unitless (`P_rx / P_tx`) | Error in linear path-loss ratio space |

Notes:
- `unit_physical: "dB"` marks the physical-domain path loss metrics.
- `unit_linear: "unitless"` plus `linear_quantity: "received_to_transmitted_power_ratio"` mark the linear-domain metrics.
- This is not watts unless a transmit-power reference is introduced and the code converts ratio to absolute received power.

---

## 4. Distance-from-Center Map (Optional)

- **Config**: `data.distance_map_channel: true/false` (currently `false` for initial runs)
- **Meaning**: 2D horizontal distance from map center (antenna at center), normalized to [0, 1]
- **Formula**: `dist = sqrt((x - cx)^2 + (y - cy)^2)`, normalized by `256 * sqrt(2)` (max distance for 513×513 map)
- **Note**: This is horizontal distance, not antenna height (height is not in the HDF5 metadata).

---

## 5. Path Loss Saturation Masking

- **Config**: `data.path_loss_saturation_db: 175` (or `null` to disable)
- **Effect**: Pixels with path_loss >= 175 dB are masked out during training (mask = 0)
- **Rationale**: IMPROVEMENTS.md: "Path loss is data missing because sometimes it was too high (so no signal)."

---

## 6. LoS-Aware Path Loss Correction (Optional Postprocessing)

- **Config**: `postprocess.path_loss_los_correction: true/false`
- **Effect**: In LoS regions, blends predicted path loss with free-space prior:
  - `PL_fs = 20*log10(d) + 20*log10(f) + 92.45` (d in m, f in GHz)
  - `blend_weight`: fraction of prior used (default 0.3)
- **Use**: Mainly for `predict_cgan.py` when LoS input is available.

---

## 7. Path-Loss-Focused Cluster Config

- **File**: `configs/cgan_unet_hdf5_pathloss_focus.yaml`
- **Changes**: Larger model (base_channels 96), `path_loss` loss weight 5.0, gradient checkpointing, 60 epochs
- **Usage**: `sbatch --export=ALL,CONFIG_PATH=configs/cgan_unet_hdf5_pathloss_focus.yaml cluster/run_train_cgan_hdf5.slurm`

---

## 8. Multi-GPU SLURM Scripts

- `cluster/run_train_cgan_hdf5_4gpu.slurm` – 4 GPUs, medium_gpu
- `cluster/run_train_cgan_hdf5_8gpu.slurm` – 8 GPUs, big_gpu

---

## 9. Metadata Inspection Script

- **File**: `inspect_hdf5_metadata.py`
- **Purpose**: Check for antenna height or other per-sample metadata in the HDF5
- **Result**: No attributes found; antenna height is not stored (Vinogradov formula cannot be applied without it).

---

## Files Modified

| File | Changes |
|------|---------|
| `data_utils.py` | `_path_loss_db_to_linear_normalized`, `path_loss_linear_normalized_to_db`, `_compute_distance_map_2d`, `distance_map_channel`, `path_loss_saturation_db`, `predict_linear` in loader |
| `heuristics_cgan.py` | `denormalize_array` with `predict_linear`, `apply_path_loss_los_correction` |
| `evaluate_cgan.py` | `denormalize_channel` with `predict_linear`, linear metrics (`mse_linear`, etc.) |
| `predict_cgan.py` | `compute_input_channels`, distance map, LoS correction |
| `configs/cgan_unet_hdf5_amd_midvram.yaml` | `distance_map_channel: false`, `path_loss_saturation_db: 175`, `predict_linear: true` |
| `configs/cgan_unet_hdf5_pathloss_focus.yaml` | New config with all improvements |
