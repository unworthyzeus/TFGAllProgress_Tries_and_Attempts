# Try 67 — SOA claims vs. what is implemented

> Companion to **`SOA_COMPARISON.md`**. This file tracks which *thesis / SOA narrative* items are **already in the Try 67 codebase** versus **not implemented, partial, or config-only**.
> Last updated: April 2026.

## At a glance

| Implemented in Try67 | Still missing / future |
|---|---|
| 3 city-morphology experts (`open_lowrise`, `mixed_midrise`, `dense_highrise`) | Test-time adaptation / calibration |
| 513x513 city-holdout training with continuous height FiLM | RadioDiff-style diffusion refinement |
| Geometric augmentation in the dataset worker | FM-RME-style foundation pre-training |
| Knife-edge diffraction prior channel | RadioPiT-style adaptation loop beyond D4 TTA |
| PDE residual loss | GAN/discriminator in the default path-loss experts |
| NLoS focus auxiliary loss (default on) | Dual LoS/NLoS **decode** head (`out_channels: 2` + flag) defaults **off** |
| D4 test-time augmentation for validation/final test | Optional: disable per-epoch validation JSON with `save_validation_json_each_epoch: false` |

---

## 1. Problem setup & data (what the SOA table claims)

| Claim in `SOA_COMPARISON.md` (sections 1 and 4) | Status | Where / notes |
|---|---|---|
| Outdoor path loss, real-city geometry + simulator labels | **Implemented** | HDF5 pipeline, `data_utils` / dataset loaders |
| **Fixed ground receiver (Rx) height** (1.5 m) in analytic channels | **Implemented** | Same value in `path_loss_formula_input.receiver_height_m`, elevation map (`data_utils.py`), and `knife_edge_channel.rx_height_m` |
| **UAV transmitter (Tx) height from HDF5** per sample | **Implemented** | Dataset field `uav_height` (m) exposed as scalar `antenna_height_m` in `data_utils` / `hdf5_scalar_specs`; drives formula prior, tx-depth, elevation, knife-edge, and **sinusoidal FiLM** in `model_pmhhnet.py` (not a fixed Tx height) |
| **City holdout** (unseen city geometries) | **Implemented** | `data.split_mode: city_holdout` in expert YAMLs |
| **513×513** maps | **Implemented** | `data.image_size: 513`, single-stage training |
| **66-city** training diversity | **Implemented** | Dataset + holdout split; not all cities in every expert filter |
| **Topology-partitioned experts** (3 city-morphology classes + registry) | **Implemented** | `scripts/generate_try67_configs.py`, `partition_filter.city_type`, separate checkpoints |
| **No test-time calibration** (zero-shot vs. measured env) | **Implemented** | No field-measurement calibration path in train/eval; contrast called out in SOA doc vs. AIRMap |

---

## 2. Model & training recipe (internal “SOTA at our formulation”)

| Component | Status | Notes |
|---|---|---|
| **PMHHNet** encoder / residual head | **Implemented** | `model_pmhhnet.py`, `arch: pmhhnet` |
| **Formula prior** channel (hybrid COST231 / two-ray, etc.) | **Implemented** | `path_loss_formula_input` in YAML + `train_partitioned_pathloss_expert.py` |
| **Regime calibration JSON** (obstruction / height thresholds) | **Implemented** | `regime_calibration_json` loaded for formula / gates where used |
| **Tx depth map** (Gao-style motivation) | **Implemented** | `data.tx_depth_map_channel` |
| **Knife-edge diffraction channel** | **Implemented** | `data.knife_edge_channel.enabled` in `data_utils.py`; `knife_edge.py` computes the extra channel per sample |
| **Corridor weighting** (Gao-style emphasis near Tx–Rx path) | **Not in trainer** | Was YAML-only and never read; **removed from Try 67 expert YAMLs** (April 2026). Implement in `train_partitioned_pathloss_expert.py` before re-adding config. |
| **LoS mask** input | **Implemented** | `los_input_column: los_mask` |
| **NLoS pixel reweighting** in loss mask | **Implemented** | `training.nlos_reweight_factor` |
| **CutMix** | **Implemented** | `training.cutmix_*` |
| **Geometric augmentation** (hflip / vflip / rot90) | **Implemented** | `data_utils.py::_apply_sync_aug` in the dataset; controlled by `augmentation.enable` and the per-flip probabilities |
| **EMA** of generator weights | **Implemented** | `training.ema_decay`, used for validation when enabled |
| **Multi-scale path-loss auxiliary** | **Implemented** | `multiscale_path_loss` (downscaled dB MSE, weighted) |
| **Huber main loss** with **δ in physical dB** (vs. mistaken normalized δ) | **Implemented** | `effective_huber_delta()` in `train_partitioned_pathloss_expert.py`; opt-out: `loss.huber_delta_normalized: true` |
| **Output clamp** in normalized space | **Implemented** | `clip_to_target_range()`; optional **`clip_min_db` / `clip_max_db`** override vs. `clip_min` / `clip_max` |
| **Physical dB clamp in predict** | **Implemented** | Same logic in `predict.py` (and tail refiner duplicate) |
| **NLoS focus auxiliary** (`hard_huber` / `rmse` modes) | **Implemented** | `compute_nlos_focus_loss()`; enabled in the Try67 expert configs as a conservative RMSE tail loss |
| **PDE residual loss** (ReVeal-style) | **Implemented** | `pde_residual_loss.enabled` and `compute_pde_residual_loss()` in `train_partitioned_pathloss_expert.py` |
| **GAN / discriminator** | **Not used** | `lambda_gan: 0`, `disc_base_channels: 0` in typical expert YAML |
| **Separated refiner / gate** | **Optional / off** in default experts | `separated_refiner` path exists for tail experiments, not default 6 experts |
| **Topology classifier** (separate script) | **Separate script** | `train_topology_classifier.py` + the classifier YAML under `experiments/sixtyseventh_try67_classifier/` — not the same run as path-loss experts |

---

## 3. Evaluation, metrics & logging

| Item | Status | Notes |
|---|---|---|
| Physical RMSE, LoS / NLoS breakdown, regime summaries | **Implemented** | `evaluate_validation` + JSON payloads |
| **`validate_metrics_latest.json`** + per-epoch files | **Implemented** | `write_validation_json()` after each full train+val epoch on **rank 0** |
| **`save_validation_json_each_epoch`** in YAML | **Implemented** | When **false**, rank 0 skips `write_validation_json` after an epoch; default **true** preserves previous behaviour |
| Progress JSON lines to stdout | **Implemented** | Includes LR, loss components; can show **NaN** if training diverges before val |

---

## 4. Inference & “paper SOA” extras

| Item | Status | Notes |
|---|---|---|
| **`test_time_augmentation` in YAML** (`enabled: true`) | **Implemented** | D4 TTA is wired in `train_partitioned_pathloss_expert.py::evaluate_validation()` and used for final test when enabled |
| **Test-time adaptation / calibration** (cf. RadioPiT, AIRMap-style) | **Not implemented** | Explicitly “No” in the SOA fair-comparison table; still absent in Try67 |
| **Per-topology tight inference clamp** (e.g. 70–140 dB) | **Implemented** | `target_metadata.path_loss.clip_{min,max}_db` is written by the generator; `clip_to_target_range()` / `predict.py` apply it |
| **Dual-head LoS/NLoS** or dedicated NLoS architecture | **Code path exists; off in default experts** | `_apply_dual_los_nlos_head()` when `dual_los_nlos_head.enabled` and `out_channels: 2`; default Try 67 YAML uses **`out_channels: 1`** and dual flag **false** |

---

## 5. “Emerging directions” from SOA section 7 (what still remains)

These are cited in **`SOA_COMPARISON.md` (section 7)** as *future* ideas. Try67 already implements the first two rows below; the last three remain future work.

| Direction | Status |
|---|---|
| ReVeal-style **PDE / physics loss** | **Implemented in Try67** |
| Geometry-assisted **knife-edge / scattering features** as extra channels | **Implemented in Try67** |
| **RadioDiff**-style diffusion refinement stage | **Missing** (separate Try / repo would be needed) |
| **RadioPiT**-style pixel transformer + test-time adaptation loop | **Missing** (D4 TTA exists, but the transformer/adaptation loop does not) |
| **FM-RME**-style foundation pre-training across experts | **Missing** |

---

## 6. Short summary for the thesis

- **Implemented and central to the SOA story:** outdoor 513², city holdout, continuous height FiLM, topology experts, geometric augmentation, knife-edge prior, PDE residual loss, D4 TTA, rich geometric + prior inputs, strong training recipe (Huber + multiscale + NLoS reweight; default experts use **single-channel** residual, dual-head path **off**), no measurement calibration at test.
- **Disk I/O:** set `save_validation_json_each_epoch: false` to skip per-epoch `validate_metrics_*.json` when profiling or saving inode traffic.
- **Deliberately absent vs. other papers:** test-time adaptation/calibration, diffusion, foundation pre-training, and the RadioPiT-style transformer loop.
- **Main remaining quality gap** (as in SOA + internal notes): **NLoS tail** (~30+ dB NLoS RMSE on `open_sparse_lowrise`) — not closed by architecture changes in the default expert setup.

For literature numbers and fair-comparison narrative, keep using **`SOA_COMPARISON.tex`** / PDF; use **this file** when you need a quick **engineering checklist** of what the repo actually does.

**Further papers (roadmap):** [`TFGpractice/SOA_PAPERS_ROADMAP.md`](../SOA_PAPERS_ROADMAP.md) — PathFinder (2025–26), reciprocity-aware aug (2025), RadioMamba (2025), priorities S/M/L.
