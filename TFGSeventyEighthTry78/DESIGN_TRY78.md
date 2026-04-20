# Try 78 — Unified Best-of-Breed Inference Pipeline

## What is Try 78?

Try 78 is **not a new training setup from scratch**. It is a validation experiment that
combines the three best-performing model families trained in prior tries into a single
unified inference pipeline. No new neural network architecture is introduced; the only
new artifact is `predict_try78.py`, the unified inference script.

The guiding hypothesis is:

> Routing predictions through family-specific experts — one for LoS path-loss, one for
> NLoS path-loss, and one for spreads — will produce lower overall RMSE than any single
> generalised model, because each expert trains on exactly the data distribution it was
> designed for.

---

## Architecture Families

### 1. LoS Path-Loss — Try 74 (PMHNet / PMHHNet)

**Source directory:** `los_pathloss/`  
**Model file:** `los_pathloss/model_pmhhnet.py`  
**Architecture class:** `PMHNetResidualRegressor`

Try 74 introduced the **PMHNet** family: a PMNet-style residual encoder–FPN decoder
augmented with a lightweight high-frequency branch driven by a fixed Laplacian filter.
The LoS expert (`arch: pmhnet`) trains on LoS-only pixels
(`los_region_mask_mode: los_only`) and achieves approximately **3.68 dB allcity RMSE**
on the held-out test set, the best LoS RMSE recorded across the project.

Key design choices preserved in try78:
- `base_channels: 60`, `hf_channels: 20`
- `arch: pmhnet` (PMHNet, not PMHHNet — the FiLM height-conditioning variant)
- `in_channels: 9` (topology, LoS mask, distance, tx-depth, elevation angle, building mask, 3 obstruction features)
- Prior-residual formula: `path_loss_formula_input.enabled: false` (absolute prediction)
- Obstruction features from Try 47 precomputed HDF5

**Prior calibration note:** The LoS yamls reference
`../../prior_calibration/regime_obstruction_train_only_from_try47.json`.
This file lives in `TFGSeventyFourthTry74/prior_calibration/` in the original tree.
It is NOT copied into `los_pathloss/` — if the path-loss formula input is disabled
(as it is in try78 by default), this reference is never read. If you re-enable formula
input, you must ensure this file is accessible at the relative path or update the yaml.

### 2. NLoS Path-Loss — Try 76 (Distribution-First GMM)

**Source directory:** `nlos_pathloss/`  
**Model file:** `nlos_pathloss/model.py`  
**Architecture class:** `Try76Model`

Try 76 introduced a **two-stage distribution-first** architecture:

- **Stage A:** A shallow conv encoder + GAP head that estimates a global K-component
  Gaussian Mixture Model (GMM) over the scene's path-loss distribution.
- **Stage B:** A U-Net decoder conditioned on the Stage-A GMM summary and the antenna
  height via FiLM, predicting per-pixel soft component assignments (`p`), residual `z`,
  and heteroscedastic uncertainty `log_sigma_tilde`.

Training loss: NLL under the mixture model (map_nll) + KL between empirical histogram
and Stage-A mixture (dist_kl) + moment matching + RMSE + outlier-budget regulariser.

The NLoS experts train on `los_region_mask_mode: nlos_only`, one expert per topology
class. Try 76 achieves the best NLoS RMSE from the current cluster run.

### 3. Spreads — Try 77 (Spike + GMM Distribution-First)

**Source directory:** `spreads/`  
**Model file:** `spreads/model.py`  
**Architecture class:** `Try77Model`

Try 77 extends the Try-76 distribution-first approach to predict **delay spread** (ns)
and **angular spread** (deg). The key difference is the **spike + GMM** head in Stage A:
component 0 is constrained to a narrow near-zero spike (capturing the dominant "no delay"
mode common in sparse scenes), while components 1..K are free-tail Gaussians.

Experts: 12 = 6 topology classes × {delay_spread, angular_spread}.
- Delay spread: `clamp_hi: 400.0`, `sigma_max: 120.0`
- Angular spread: `clamp_hi: 90.0`, `sigma_max: 45.0`, narrower spike parameters

---

## Topology Classification

All three families share the same 6-class topology partitioning rule, derived from
Try 54 / Try 67:

| Class                   | Density        | Mean height  |
|-------------------------|----------------|--------------|
| open_sparse_lowrise     | ≤ 0.12         | ≤ 12 m       |
| open_sparse_vertical    | ≤ 0.12         | > 12 m       |
| mixed_compact_lowrise   | 0.12 – 0.28    | ≤ 12 m       |
| mixed_compact_midrise   | 0.12 – 0.28    | > 12 m       |
| dense_block_midrise     | ≥ 0.28         | ≤ 28 m       |
| dense_block_highrise    | ≥ 0.28         | > 28 m       |

Thresholds: `density_q1=0.12`, `density_q2=0.28`, `height_q1=12.0`, `height_q2=28.0`.

---

## Expert Count

| Family            | Experts | Rule                                    |
|-------------------|---------|-----------------------------------------|
| LoS path-loss     | 6       | one per topology class, LoS pixels only |
| NLoS path-loss    | 6       | one per topology class, NLoS pixels only|
| Spreads           | 12      | one per (topology class × metric)       |
| **Total**         | **24**  |                                         |

---

## Inference Pipeline (`predict_try78.py`)

```
topology_map + los_mask + height
        |
        v
classify_topology()  -->  topology_class
        |
        +--------> [LoS PMHNet expert] --> los_pred  (H x W, dB)
        |
        +--------> [NLoS Try76 expert] --> nlos_pred (H x W, dB)
        |
        +--------> [Delay Try77 expert] --> delay_pred (H x W, ns)
        |
        +--------> [Angular Try77 expert] --> angular_pred (H x W, deg)
        |
        v
  ROUTING (LoS mask):
    path_loss_combined[LoS pixels]  = los_pred
    path_loss_combined[NLoS pixels] = nlos_pred
    delay_spread  = delay_pred  (ground pixels only)
    angular_spread = angular_pred (ground pixels only)
```

All four models are loaded into VRAM simultaneously before inference begins. The script
accepts `--topology-class` to skip auto-detection, or auto-detects it from the topology
map if not provided.

### Input preparation per family

**LoS (PMHNet):** 9-channel input matching try74's `data_utils.py`:
`[topo/255, los_mask, dist_map, tx_depth(0), elevation(0), building_mask, obstr×3(0)]`.
The three obstruction channels and the depth/elevation channels are zeroed out at
inference because they require precomputed features not available in a standalone
prediction context. If you need full accuracy, provide the obstruction HDF5 as in
training.

**NLoS / Spreads (Try76/77):** 4-channel input matching try76/77's `data_utils.py`:
`[topo*ground/90, los*ground, (1-los)*ground, ground]`, plus the sinusoidal height
embedding computed on-the-fly.

---

## Directory Layout

```
TFGSeventyEighthTry78/
  DESIGN_TRY78.md              ← this file
  predict_try78.py             ← unified inference entry point
  los_pathloss/                ← try74 codebase (LoS training)
    model_pmhhnet.py           ← verbatim copy of try74/model_pmhhnet.py
    config_utils.py            ← verbatim copy
    data_utils.py              ← verbatim copy
    train.py                   ← copy of train_partitioned_pathloss_expert.py
    evaluate.py                ← copy with import: train → train
    experiments/
      try78_expert_<topo>_los.yaml  ×6
      try78_los_expert_registry.yaml
  nlos_pathloss/               ← try76 codebase (NLoS training)
    model.py                   ← verbatim copy of src/model_try76.py
    config.py                  ← verbatim copy of src/config_try76.py
    data_utils.py              ← verbatim copy of src/data_utils.py
    losses.py                  ← verbatim copy of src/losses_try76.py
    metrics.py                 ← verbatim copy of src/metrics_try76.py
    train.py                   ← copy with imports fixed (from src.X → from X)
    evaluate.py                ← copy with imports fixed
    experiments/
      try78_expert_<topo>_nlos.yaml  ×6
      try78_nlos_expert_registry.yaml
  spreads/                     ← try77 codebase (spread training)
    model.py                   ← verbatim copy of src/model_try77.py
    config.py                  ← verbatim copy of src/config_try77.py
    data_utils.py              ← verbatim copy of src/data_utils.py
    losses.py                  ← verbatim copy of src/losses_try77.py
    metrics.py                 ← verbatim copy of src/metrics_try77.py
    train.py                   ← copy with imports fixed
    evaluate.py                ← copy with imports fixed
    experiments/
      try78_expert_<topo>_{delay,angular}_spread.yaml  ×12
      try78_spread_expert_registry.yaml
```

---

## Training Each Family

### LoS experts

```bash
cd los_pathloss
python train.py --config experiments/try78_expert_open_sparse_lowrise_los.yaml
# ... repeat for all 6 topology classes
```

### NLoS experts

```bash
cd nlos_pathloss
python train.py --config experiments/try78_expert_open_sparse_lowrise_nlos.yaml
# ... repeat for all 6 topology classes
```

### Spread experts

```bash
cd spreads
python train.py --config experiments/try78_expert_open_sparse_lowrise_delay_spread.yaml
python train.py --config experiments/try78_expert_open_sparse_lowrise_angular_spread.yaml
# ... repeat for all 6 topology classes × 2 metrics
```

---

## Inference

```bash
python predict_try78.py \
    --topology-map /path/to/topology_map.npy \
    --los-mask /path/to/los_mask.npy \
    --height 50.0 \
    --topology-class open_sparse_lowrise \
    --los-checkpoint los_pathloss/outputs/try78_expert_open_sparse_lowrise_los/best_model.pt \
    --nlos-checkpoint nlos_pathloss/outputs/try78_expert_open_sparse_lowrise_nlos/best_model.pt \
    --delay-checkpoint spreads/outputs/try78_expert_open_sparse_lowrise_delay_spread/best_model.pt \
    --angular-checkpoint spreads/outputs/try78_expert_open_sparse_lowrise_angular_spread/best_model.pt \
    --output-dir /tmp/try78_outputs
```

Output files:
- `path_loss_combined.npy` — merged LoS/NLoS path-loss map (dB)
- `path_loss_los.npy` — raw LoS model output (dB)
- `path_loss_nlos.npy` — raw NLoS model output (dB)
- `delay_spread.npy` — delay spread map (ns)
- `angular_spread.npy` — angular spread map (deg)
- `los_mask_used.npy` — LoS mask used for routing

---

## Dataset reference

All yamls point to `../../Datasets/CKM_Dataset_270326.h5` (two levels up from the
`los_pathloss/`, `nlos_pathloss/`, or `spreads/` subdirectory, i.e. relative to the
`TFGPractice/` root). Adjust this path if your dataset is located elsewhere.
