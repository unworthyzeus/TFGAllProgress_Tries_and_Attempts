# Try 69 — SOA claims vs. what is implemented

> Companion to **`SOA_COMPARISON.tex`**. This file tracks which *thesis / SOA narrative* items are **already in the Try 69 codebase** versus **not implemented, partial, or config-only**.
> Last updated: April 2026.

## At a glance

| Implemented in Try69 | Still missing / future |
|---|---|
| 3 city-morphology experts (`open_lowrise`, `mixed_midrise`, `dense_highrise`) | Test-time adaptation / calibration |
| 513x513 city-holdout training with continuous height FiLM | RadioDiff-style diffusion refinement |
| Geometric augmentation in the dataset worker | FM-RME-style foundation pre-training |
| Knife-edge diffraction prior channel | RadioPiT-style adaptation loop beyond D4 TTA |
| PDE residual loss | GAN / discriminator in the default path-loss experts (still off) |
| NLoS focus auxiliary loss (default on) | RadioDiff-style diffusion refinement |
| **Dual LoS/NLoS decode head** (`out_channels: 2`, enabled in YAML) | FM-RME-style foundation pre-training |
| **Radial corridor loss weights** (Tx-centred; `corridor_weighting` + trainer) | Test-time adaptation / calibration |
| D4 TTA on **validation** and final test | Optional: disable per-epoch validation JSON with `save_validation_json_each_epoch: false` |

---

## 1. Problem setup & data (what the SOA table claims)

| Claim in `SOA_COMPARISON.tex` (sections 1 and 4) | Status | Where / notes |
|---|---|---|
| Outdoor path loss, real-city geometry + simulator labels | **Implemented** | HDF5 pipeline, `data_utils` / dataset loaders |
| **Fixed ground receiver (Rx) height** (1.5 m) in analytic channels | **Implemented** | `path_loss_formula_input.receiver_height_m`, elevation map reads the same key in `data_utils.py`; `knife_edge_channel.rx_height_m` |
| **UAV transmitter (Tx) height from HDF5** per sample | **Implemented** | Field `uav_height` (m) → scalar `antenna_height_m`; formula, tx-depth, elevation, knife-edge, and **sinusoidal FiLM** (`model_pmhhnet.py`) all use this **per-sample Tx** height (12–478 m) |
| **City holdout** (unseen city geometries) | **Implemented** | `data.split_mode: city_holdout` in expert YAMLs |
| **513×513** maps | **Implemented** | `data.image_size: 513`, single-stage training |
| **66-city** training diversity | **Implemented** | Dataset + holdout split; not all cities in every expert filter |
| **Topology-partitioned experts** (6 `topology_class` + registry) | **Implemented** | `experiments/sixtyninth_try69_experts/try69_expert_*.yaml`, `partition_filter.topology_class`, separate checkpoints |
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
| **Corridor weighting** (loss mask) | **Implemented** | `corridor_weighting.mode`: `radial` / `anisotropic` (cached) / `gao_ray` (per-step ray obstruction along centre→pixel, see trainer). Multiplies **training** loss mask only. |
| **LoS mask** input | **Implemented** | `los_input_column: los_mask` |
| **NLoS pixel reweighting** in loss mask | **Implemented** | `training.nlos_reweight_factor` |
| **CutMix** | **Implemented** | `training.cutmix_*`; with FiLM + `scalar_cond`, use **`training.cutmix_film_safe.enabled: true`** to mix **geometry-only** then **`recompute_height_dependent_input_channels`** (formula, confidence, tx-depth, elevation, building, knife). Otherwise CutMix is skipped when FiLM scalars are present. |
| **Short component ablations** | **Script** | `scripts/run_try69_component_ablation.py` + `train_partitioned_pathloss_expert.py --epochs-override` |
| **Geometric augmentation** (hflip / vflip / rot90) | **Implemented** | `data_utils.py::_apply_sync_aug` in the dataset; controlled by `augmentation.enable` and the per-flip probabilities |
| **EMA** of generator weights | **Implemented** | `training.ema_decay`, used for validation when enabled |
| **Multi-scale path-loss auxiliary** | **Implemented** | `multiscale_path_loss` (downscaled dB MSE, weighted) |
| **Huber main loss** with **δ in physical dB** (vs. mistaken normalized δ) | **Implemented** | `effective_huber_delta()` in `train_partitioned_pathloss_expert.py`; opt-out: `loss.huber_delta_normalized: true` |
| **Output clamp** in normalized space | **Implemented** | `clip_to_target_range()`; optional **`clip_min_db` / `clip_max_db`** override vs. `clip_min` / `clip_max` |
| **Physical dB clamp in predict** | **Implemented** | Same logic in `predict.py` (and tail refiner duplicate) |
| **NLoS focus auxiliary** (`hard_huber` / `rmse` modes) | **Implemented** | `compute_nlos_focus_loss()`; enabled in the Try69 expert configs as a conservative RMSE tail loss |
| **PDE residual loss** (ReVeal-style) | **Implemented** | `pde_residual_loss.enabled` and `compute_pde_residual_loss()` in `train_partitioned_pathloss_expert.py` |
| **GAN / discriminator** | **Not used** | `lambda_gan: 0`, `disc_base_channels: 0` in typical expert YAML |
| **Separated refiner / gate** | **Optional / off** in default experts | `separated_refiner` path exists for tail experiments, not default 6 experts |
| **Topology classifier** (separate script) | **Separate script** | `train_topology_classifier.py` + the classifier YAML under `experiments/sixtyninth_try69_classifier/` — not the same run as path-loss experts |

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
| **`test_time_augmentation` in YAML** (`enabled: true`) | **Implemented** | D4 TTA in `evaluate_validation()`; Try 69 defaults **`use_in_validation: true`** and **`use_in_final_test: true`** |
| **Test-time adaptation / calibration** (cf. RadioPiT, AIRMap-style) | **Not implemented** | Explicitly “No” in the SOA fair-comparison table; still absent in Try69 |
| **Per-topology tight inference clamp** (e.g. 70–140 dB) | **Implemented** | `target_metadata.path_loss.clip_{min,max}_db` is written by the generator; `clip_to_target_range()` / `predict.py` apply it |
| **Dual-head LoS/NLoS** or dedicated NLoS architecture | **On in default Try 69 experts** | `_apply_dual_los_nlos_head()` with **`out_channels: 2`** and **`dual_los_nlos_head.enabled: true`** in generated YAMLs |

---

## 5. “Emerging directions” from SOA section 7 (what still remains)

These are cited in **`SOA_COMPARISON.tex` (section 7)** as *future* ideas. Try69 already implements the first two rows below; the last three remain future work.

| Direction | Status |
|---|---|
| ReVeal-style **PDE / physics loss** | **Implemented in Try69** |
| Geometry-assisted **knife-edge / scattering features** as extra channels | **Implemented in Try69** |
| **RadioDiff**-style diffusion refinement stage | **Missing** (separate Try / repo would be needed) |
| **RadioPiT**-style pixel transformer + test-time adaptation loop | **Missing** (D4 TTA exists, but the transformer/adaptation loop does not) |
| **FM-RME**-style foundation pre-training across experts | **Missing** |

---

## 6. Short summary for the thesis

- **Implemented and central to the SOA story:** outdoor 513², city holdout, continuous height FiLM, topology experts, geometric augmentation, knife-edge prior, PDE residual loss, **dual LoS/NLoS residual heads**, **corridor loss weights** (`radial` / `anisotropic` / `gao_ray`), **FiLM-safe CutMix** (`cutmix_film_safe` + on-GPU physics channel recompute), D4 TTA (including validation), multiscale + NLoS reweight, MSE main loss in `full_map_rmse_only`, no measurement calibration at test. **Ablations:** `scripts/run_try69_component_ablation.py` + `--epochs-override`.
- **Disk I/O:** set `save_validation_json_each_epoch: false` to skip per-epoch `validate_metrics_*.json` when profiling or saving inode traffic.
- **Deliberately absent vs. other papers:** test-time adaptation/calibration, diffusion, foundation pre-training, and the RadioPiT-style transformer loop.
- **Main remaining quality gap** (as in SOA + internal notes): **NLoS tail** (~30+ dB NLoS RMSE on `open_sparse_lowrise`) — not closed by architecture changes in the default expert setup.

For literature numbers and fair-comparison narrative, keep using **`SOA_COMPARISON.tex`** / PDF; use **this file** when you need a quick **engineering checklist** of what the repo actually does.

---

## 7. Extended literature roadmap (Try 69 folder copy of the global list)

Canonical detail, horizons **S/M/L**, and caveats live in **[`TFGpractice/SOA_PAPERS_ROADMAP.md`](../SOA_PAPERS_ROADMAP.md)**. The table below is the **Try 69** checklist: what is **only cited / future** vs. **already touched** by this codebase.

| Work | Identifier | Relation to Try 69 |
|------|------------|------------------|
| **PathFinder** (disentangled features, Transmitter-Oriented Mixup) | [arXiv:2512.14150](https://arxiv.org/abs/2512.14150) | **Not implemented** — candidate aug / attention (roadmap priority **M**) |
| **Reciprocity-aware CNNs** (DL-only symmetries) | [arXiv:2504.03625](https://arxiv.org/abs/2504.03625) | **Not implemented** — physics-style aug on topology + height (**S**–**M**) |
| **RadioMamba** (Mamba–UNet) | [arXiv:2508.09140](https://arxiv.org/abs/2508.09140) | **Not implemented** — backbone-scale change (**L**) |
| **RadioTransformer** | [arXiv:2501.05190](https://arxiv.org/abs/2501.05190) | **Not implemented** — encoder replacement / cross-attn to height (**L**) |
| **Gao et al.** (effective outdoor path loss, corridor) | [arXiv:2601.08436](https://arxiv.org/abs/2601.08436) | **Partial** — `corridor_weighting.mode: radial` / `anisotropic` / **`gao_ray`** (ray-sampled building occlusion on the loss mask); not the full multi-layer segmentation pipeline from the paper |
| **ReVeal** (PINN-style wave residual) | [arXiv:2502.19646](https://arxiv.org/abs/2502.19646) | **Partial** — masked Laplacian / PDE auxiliary in train; not full Helmholtz operator |
| **AIRMap** (fast U-Net surrogate + few-sample calibration) | [arXiv:2511.05522](https://arxiv.org/abs/2511.05522) | **Not implemented** — narrative contrast (“no test calibration”); optional future adapter (**M**–**L**) |
| **TransPathNet** (two-stage transformer + decoder) | [arXiv:2501.16023](https://arxiv.org/abs/2501.16023) | **Not implemented** — indoor challenge; ideas for refiner / multiscale decoder (**M**–**L**) |
| **IPP-Net** / ICASSP 2025 indoor SIA | Leaderboard / paper trail | **Not implemented** — positioning only (~9.5 dB indoor CW-RMSE class) |
| **Triple-layer UAV ML** (tabular + GPR) | [arXiv:2505.19478](https://arxiv.org/abs/2505.19478) | **Not implemented** — non-map baseline for thesis discussion (**S** analysis) |
| **NeWRF** | [arXiv:2403.03241](https://arxiv.org/abs/2403.03241), [ICML 2024](https://proceedings.mlr.press/v235/lu24j.html) | **Not implemented** — sparse-measurement / neural field (**L**) |
| **GRaF** | [arXiv:2502.05708](https://arxiv.org/abs/2502.05708) | **Not implemented** — cross-scene RF field (**L**) |
| **NeRF²** (MobiCom 2023) | [PDF](https://web.comp.polyu.edu.hk/csyanglei/data/files/nerf2-mobicom23.pdf) | **Not implemented** — cite as bridge to neural RF fields |
| **UAV A2G measurements** (1 & 4 GHz) | [arXiv:2501.17303](https://arxiv.org/abs/2501.17303) | **Not in code** — supports FiLM / elevation rationale in text |
| **RadioDiff** / **FM-RME** / **RadioPiT TTA** | See `SOA_COMPARISON.tex` section 7 | **Not implemented** (already rows in section 5) |

**Repos / leaderboards:** [Awesome Radio Map (categorized)](https://github.com/UNIC-Lab/Awesome-Radio-Map-Categorized), [RadioMap Challenge results](https://radiomapchallenge.github.io/results.html).

**Suggested experiment order** (mirrors roadmap section D): PathFinder-style mixup → reciprocity-style augs → full Gao corridor → AIRMap-style calibration (ablation only) → sparse-measurement / neural-field branch → Mamba or slim transformer encoder.
