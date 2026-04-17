# Try 73 — SOA claims vs. what is implemented

> Companion to **`SOA_COMPARISON.md`**. This file tracks which *thesis / SOA narrative* items are **already in the Try 73 codebase** versus **not implemented, partial, or config-only**.  
> Last updated: April 2026.

---

## 1. Problem setup & data (what the SOA table claims)

| Claim in `SOA_COMPARISON.md` (§1, §4) | Status | Where / notes |
|---|---|---|
| Outdoor path loss, real-city geometry + simulator labels | **Implemented** | HDF5 pipeline, `data_utils` / dataset loaders |
| **City holdout** (unseen city geometries) | **Implemented** | `data.split_mode: city_holdout` in expert YAMLs |
| **513×513** maps | **Implemented** | `data.image_size: 513`, single-stage training |
| **Continuous UAV Tx height** (12–478 m) as conditioning | **Implemented** | Scalar `antenna_height_m` + **sinusoidal FiLM** in `model_pmhhnet.py` (`use_scalar_film`, etc.) |
| **66-city** training diversity | **Implemented** | Dataset + holdout split; not all cities in every expert filter |
| **Topology-partitioned experts** (6 classes + registry) | **Implemented** | `try73_expert_registry.yaml`, `partition_filter.topology_class`, separate checkpoints |
| **No test-time calibration** (zero-shot vs. measured env) | **Implemented** | No field-measurement calibration path in train/eval; contrast called out in SOA doc vs. AIRMap |

---

## 2. Model & training recipe (internal “SOTA at our formulation”)

| Component | Status | Notes |
|---|---|---|
| **PMHHNet** encoder / residual head | **Implemented** | `model_pmhhnet.py`, `arch: pmhhnet` |
| **Formula prior** channel (hybrid COST231 / two-ray, etc.) | **Implemented** | `path_loss_formula_input` in YAML + `train_partitioned_pathloss_expert.py` |
| **Regime calibration JSON** (obstruction / height thresholds) | **Implemented** | `regime_calibration_json` loaded for formula / gates where used |
| **Tx depth map** (Gao-style motivation) | **Implemented** | `data.tx_depth_map_channel` |
| **Corridor weighting** (Gao-style emphasis near Tx–Rx path) | **Implemented** | `corridor_weighting` block in YAML + loss path in trainer |
| **LoS mask** input | **Implemented** | `los_input_column: los_mask` |
| **NLoS pixel reweighting** in loss mask | **Implemented** | `training.nlos_reweight_factor` |
| **CutMix** | **Implemented** | `training.cutmix_*` |
| **EMA** of generator weights | **Implemented** | `training.ema_decay`, used for validation when enabled |
| **Multi-scale path-loss auxiliary** | **Implemented** | `multiscale_path_loss` (downscaled dB MSE, weighted) |
| **Huber main loss** with **δ in physical dB** (vs. mistaken normalized δ) | **Implemented** | `effective_huber_delta()` in `train_partitioned_pathloss_expert.py`; opt-out: `loss.huber_delta_normalized: true` |
| **Output clamp** in normalized space | **Implemented** | `clip_to_target_range()`; optional **`clip_min_db` / `clip_max_db`** override vs. `clip_min` / `clip_max` |
| **Physical dB clamp in predict** | **Implemented** | Same logic in `predict.py` (and tail refiner duplicate) |
| **NLoS focus auxiliary** (`hard_huber` / `rmse` modes) | **Implemented (optional)** | `compute_nlos_focus_loss()`; often **disabled** in YAML after stability issues |
| **GAN / discriminator** | **Not used** | `lambda_gan: 0`, `disc_base_channels: 0` in typical expert YAML |
| **Separated refiner / gate** | **Optional / off** in default experts | `separated_refiner` path exists for tail experiments, not default 6 experts |
| **Topology classifier** (Try 73) | **Separate script** | `train_topology_classifier.py` + `try73_topology_classifier.yaml` — not the same run as path-loss experts |

---

## 3. Evaluation, metrics & logging

| Item | Status | Notes |
|---|---|---|
| Physical RMSE, LoS / NLoS breakdown, regime summaries | **Implemented** | `evaluate_validation` + JSON payloads |
| **`validate_metrics_latest.json`** + per-epoch files | **Implemented** | `write_validation_json()` after each full train+val epoch on **rank 0** |
| **`save_validation_json_each_epoch`** in YAML | **Not wired** | Flag exists in configs but **not read** by `train_partitioned_pathloss_expert.py`; JSON is written whenever the epoch completes successfully |
| Progress JSON lines to stdout | **Implemented** | Includes LR, loss components; can show **NaN** if training diverges before val |

---

## 4. Inference & “paper SOA” extras

| Item | Status | Notes |
|---|---|---|
| **`test_time_augmentation` in YAML** (`enabled: true`) | **Not implemented** | No reference in `train_partitioned_pathloss_expert.py` or `predict.py`; **config is misleading** — safe to set `enabled: false` or implement later |
| **Test-time adaptation / calibration** (cf. RadioPiT, AIRMap-style) | **Not implemented** | Explicitly “No” in SOA fair-comparison table for Try 73 |
| **Per-topology tight inference clamp** (e.g. 70–140 dB) | **Documented only** | Proposed in `CLAMPING_AND_IMPROVEMENTS.md` / `BENCHMARKS_AND_CONTEXT.md`; not a separate default in all expert YAMLs |
| **Dual-head LoS/NLoS** or dedicated NLoS architecture | **Not implemented** | Single residual head; NLoS is handled via weighting / optional auxiliary loss |

---

## 5. “Emerging directions” from SOA §7 (explicitly *not* in pipeline)

These are cited in **`SOA_COMPARISON.md` §7** as *future* ideas — **none** are integrated as first-class Try 73 features today:

| Direction | Status |
|---|---|
| ReVeal-style **PDE / physics loss** | **Missing** |
| Geometry-assisted **knife-edge / scattering features** as extra channels | **Missing** (beyond depth / corridor / formula) |
| **RadioDiff**-style diffusion refinement stage | **Missing** (separate Try / repo would be needed) |
| **RadioPiT**-style pixel transformer + **TTA** | **Missing** |
| **FM-RME**-style foundation pre-training across experts | **Missing** |

---

## 6. Short summary for the thesis

- **Implemented and central to the SOA story:** outdoor 513², city holdout, continuous height FiLM, topology experts, rich geometric + prior inputs, strong training recipe (Huber+multiscale+corridor+NLoS reweight), no measurement calibration at test.
- **Partial / misleading in config:** TTA block; `save_validation_json_each_epoch` unused.
- **Deliberately absent vs. other papers:** test-time calibration, TTA, diffusion/PINN/foundation add-ons from §7.
- **Main remaining quality gap** (as in SOA + internal notes): **NLoS tail** (~30+ dB NLoS RMSE on `open_sparse_lowrise`) — not closed by architecture changes in the default expert setup.

For literature numbers and fair-comparison narrative, keep using **`SOA_COMPARISON.md`**; use **this file** when you need a quick **engineering checklist** of what the repo actually does.
