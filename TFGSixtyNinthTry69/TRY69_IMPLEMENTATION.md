# Try 69 — Implementation Details

Companion to `TRY69_DESIGN.md`. That doc explains *why*; this one documents
*where the code lives* and *how to enable each feature*. Use this as the
review checklist when porting a feature forward to a later try.

---

## 1. Knife-edge diffraction channel (SOA #1)

### Files
- `knife_edge.py` — new module. Pure-torch, no external deps.
- `data_utils.py` — appends the channel to `model_input_channels` inside
  `CKMHDF5Dataset.__getitem__`.
- `scripts/generate_try69_configs.py` — writes the `knife_edge_channel`
  block into every expert YAML.

### Public API (`knife_edge.py`)
```python
compute_knife_edge_loss_map(
    topo_m: Tensor,            # (1, H, W) building heights in meters
    *,
    antenna_height_m: float,
    frequency_ghz: float,
    meters_per_pixel: float,
    rx_height_m: float = 1.5,
    num_ray_samples: int = 48,
) -> Tensor                    # (1, H, W) diffraction loss in dB, >= 0

normalize_knife_edge_db(
    loss_db: Tensor,
    *, scale_db: float = 40.0,
) -> Tensor                    # (1, H, W) in [0, 1]
```

### Physics
- TX fixed at map center at altitude `antenna_height_m` AGL.
- RX at `rx_height_m` AGL at every pixel (3GPP TR 38.901 UE height).
- Along each TX→RX ray, `num_ray_samples` points are bilinearly sampled from
  `topo_m` via `F.grid_sample(..., align_corners=True)`. The dominant edge
  is the argmax of `(topo_sample - ray_altitude).clamp_min(0)` — Bullington
  1947 single-edge approximation.
- Fresnel parameter `v = h · sqrt(2(d1 + d2) / (λ · d1 · d2))`.
- Lee 1985 / ITU-R P.526-15 §4.5.1 closed form, valid for `v > −0.78`:
  `J(v) = 6.9 + 20·log10( sqrt((v−0.1)² + 1) + (v−0.1) )`.
- Pure-LoS gate: `has_edge = excess > 1e-2` prevents the `~6 dB at v=0`
  artefact of the Lee formula for grazing rays.

### YAML toggle
```yaml
data:
  knife_edge_channel:
    enabled: true
    frequency_ghz: 3.5
    meters_per_pixel: 1.0
    rx_height_m: 1.5
    num_ray_samples: 48
    scale_db: 40.0
```
Computed per-sample in the dataset worker (~10 ms / 513×513 map on one
CPU). No offline cache.

### Sanity checks (done)
- Pure LoS → 0.00 dB
- Behind a 25 m wall at 50 m → ~46 dB
- UAV at 40 m clearing the wall → ~43 dB (slightly less shadowed)

---

## 2. Dual LoS/NLoS head (SOA #3)

### Files
- `train_partitioned_pathloss_expert.py`:
  - `_dual_head_cfg(cfg)` — reads `cfg.dual_los_nlos_head.enabled`.
  - `_apply_dual_los_nlos_head(raw_out, x, cfg)` — blends the two heads
    with the binary LoS mask (channel 1 of `x`).
  - `_compose_residual_prediction_with_aux(..., cfg=cfg)` — when dual
    head is enabled, reads channels 0/1 as LoS/NLoS heads; aux channels
    shift to index 2+.
- `scripts/generate_try69_configs.py` — sets `model.out_channels = 2`
  and writes `dual_los_nlos_head.enabled = true`.

### Blending rule (single source of truth)
```python
residual_los  = raw_out[:, 0:1]
residual_nlos = raw_out[:, 1:2]
los_mask      = (x[:, 1:2] > 0.5).to(residual_los.dtype)  # straight-through
blended       = los_mask * residual_los + (1 - los_mask) * residual_nlos
```

### Why no per-head loss term
With a *binary* mask, `∂L/∂H_los = los_mask · ∂L/∂pred`, so the LoS head
only sees gradient in LoS pixels (and likewise for NLoS). Disjoint
gradient paths are obtained for free — adding an explicit per-head MSE
would only dilute the main loss.

### YAML toggle
```yaml
model:
  out_channels: 2
dual_los_nlos_head:
  enabled: true
```

---

## 3. PDE residual loss (SOA #2, simplified ReVeal)

### Files
- `train_partitioned_pathloss_expert.py`:
  - `_PDE_LAPLACIAN_KERNEL` — 5-point finite-difference stencil.
  - `compute_pde_residual_loss(pred, mask, x, cfg)` — returns
    `mean( |∇²pred| · gate )` where `gate = (LoS == 1) ∧ (valid_mask == 1)`.
  - Training loop accumulates `running_pde` (raw) and
    `running_term_pde` (weighted) separately.

### Why LoS-only
The Laplacian of a 1/r²-ish LoS field is smooth; the NLoS tail has
diffraction discontinuities that the penalty should *not* smear.

### YAML toggle
```yaml
pde_residual_loss:
  enabled: true
  loss_weight: 0.01
```
Low weight on purpose: regulariser, not a fitting term.

---

## 4. D4 test-time augmentation (existing SOA #4)

### Files
- `train_partitioned_pathloss_expert.py`:
  - `_d4_forward_inverse()` — 8 (forward, inverse) pairs (identity, hflip,
    vflip, rot180, rot90, rot270, transpose, anti-transpose).
  - `_tta_predict_residual_d4(..., cfg=cfg)` — averages the 8 inverse-
    transformed predictions.
  - `evaluate_validation(..., is_final_test=False)` — picks TTA vs plain
    composer based on `test_time_augmentation.use_in_{validation,final_test}`.

### YAML toggle
```yaml
test_time_augmentation:
  enabled: true
  transforms: d4
  use_in_validation: false     # off during training — too slow
  use_in_final_test: true      # on at the end for the reported number
```

---

## 5. Topology-class partition (Try 54 six experts)

### Files
- `data_utils.py::_filter_hdf5_refs_by_partition` — accepts
  `topology_class` **or** `city_type` in `partition_filter` (mutually exclusive
  per YAML). Topology labels use the same density/height thresholds as Try 54/68
  via `topology_partitioning` (`density_q1/q2`, `height_q1/q2`).
- For **formula prior** routing, `prefer_threshold_city_type` + calibration JSON
  may still infer a coarse `city_type` for the hybrid model; that is independent
  of which **training expert** YAML (`topology_class`) filters the sample list.
- `experiments/sixtyninth_try69_experts/try69_expert_registry.yaml` — **six**
  `topology_class` experts (Try 54 partition), each YAML
  `try69_expert_<topology_class>.yaml`. **Knife-edge on**;
  **`path_loss_obstruction_features.enabled: false`** (all `include_*` false).

### Why per-sample thresholds, not a city-name map
A lookup table would fail on any city not in the training set. Thresholds
on density / mean height let the classifier route arbitrary new cities
into the right expert at inference time.

### YAML toggle (partition)
```yaml
data:
  partition_filter:
    topology_class: open_sparse_lowrise   # or open_sparse_vertical, mixed_compact_* , dense_block_*
  topology_partitioning:
    density_q1: 0.12
    density_q2: 0.28
    height_q1: 12.0
    height_q2: 28.0
```

---

## 6. Per-expert tight output clamp

### Current six-expert YAMLs
- Optional `target_metadata.path_loss.clip_{min,max}_db` may be omitted
  (full 0–180 dB normalized range via `clip_min` / `clip_max` only), as in
  Try 68-style configs.
- To re-enable tight dB clamps per `topology_class`, add the keys back under
  `target_metadata.path_loss` and keep them consistent with `clip_to_target_range`.

### Legacy 3-class ITU reference (superseded in repo)
| Expert (old)     | clip_min_db | clip_max_db |
|------------------|-------------|-------------|
| open_lowrise     | 60          | 125         |
| mixed_midrise    | 58          | 135         |
| dense_highrise   | 55          | 145         |

---

## 7. JSON output additions

### `runtime.loss_components` (per epoch)
New fields written every epoch by the training loop:
```json
"loss_components": {
  "final_loss": ...,
  "multiscale_loss": ...,
  "nlos_focus_loss": ...,
  "pde_loss": ...,       // raw |∇²pred| mean on LoS∧valid
  "term_pde": ...        // pde_loss * loss_weight (actual gradient term)
}
```

### `experiment.soa_features` (once per run)
Written by `build_experiment_summary` for the plotter and for future
paper tables:
```json
"soa_features": {
  "knife_edge_channel": true,
  "pde_residual_loss": true,
  "pde_residual_loss_weight": 0.01,
  "dual_los_nlos_head": true,
  "tta_d4_enabled": true,
  "tta_in_validation": false,
  "tta_in_final_test": true,
  "nlos_reweight_factor": 4.0,
  "cutmix_prob": 0.45,
  "clip_min_db": 60.0,
  "clip_max_db": 125.0,
  "partition_city_type": "open_lowrise",
  "loss_type": "mse"
}
```
`experiment.city_type` is also written alongside for convenience.

---

## 8. Plotter (`scripts/plot_try69_metrics.py`)

Renamed from the Try 66 plotter. Differences:

- `DEFAULT_ROOT = .../TFGSixtyNinthTry69`.
- `EXPERT_PREFIX = sixtyninth_try69_expert_` (old prefixes accepted
  for reading legacy output dirs).
- Output filename `metrics_plot_try67.png`.
- Loss-components panel now also plots `pde_loss` (dashed) and
  `term_pde` (solid weighted).
- A figure-level `suptitle` renders the **SOA features caption**
  summarising the active features for the run, read from
  `experiment.soa_features`. If missing (older checkpoint), the caption
  degrades gracefully to an informative note.

### Usage
```bash
python scripts/plot_try69_metrics.py
python scripts/plot_try69_metrics.py --expert-id open_lowrise
python scripts/plot_try69_metrics.py --root-dir D:/cluster_outputs/TFGSixtyNinthTry69
```

---

## 9. Quick enable/disable map

| Feature                    | YAML key                                 | Default |
|---------------------------|-------------------------------------------|---------|
| Knife-edge channel        | `data.knife_edge_channel.enabled`         | true    |
| Dual LoS/NLoS head        | `dual_los_nlos_head.enabled` + `model.out_channels: 2` | true |
| PDE residual              | `pde_residual_loss.enabled`               | true    |
| D4 TTA (final test)       | `test_time_augmentation.use_in_final_test`| true    |
| D4 TTA (per-epoch val)    | `test_time_augmentation.use_in_validation`| false   |
| City-type routing         | `data.partition_filter.city_type`         | *required* |
| Tight output clamp        | `target_metadata.path_loss.clip_{min,max}_db` | per expert |

Disable any feature by flipping its flag to `false` (or removing the
block). Every feature is independent; removing the knife-edge channel
does not change the number of model output channels, for instance.

---

## 10. Future work hooks

Not implemented in Try 69, but the codebase has stubs / easy seams:

- **Physics-consistent multi-edge Deygout** — `_bullington_edge` in
  `knife_edge.py` could be extended to second-order edges by re-running
  the argmax on the residual excess after clipping the dominant edge.
- **Cross-regime distillation** — the city-type registry at
  `experiments/sixtyninth_try69_experts/try69_expert_registry.yaml`
  could feed a teacher-student loss where each expert predicts all cities
  at val time (currently each is trained/validated only on its slice).
- **Soft (non-binary) LoS gate** — currently the dual-head blend uses a
  hard `> 0.5` mask; replacing it with the continuous LoS probability
  channel would give both heads partial gradient on ambiguous pixels
  (e.g. rooftops).
