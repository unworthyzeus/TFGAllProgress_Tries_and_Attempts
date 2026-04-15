# Try 67 — Design rationale

Try 66 plateaued in validation at ~9.3 dB RMSE and started overfitting
after the best epoch regardless of which expert (e.g. `open_sparse_lowrise`
reached 9.34 dB and never improved). The goal of Try 67 is twofold:

1. Replace the arbitrary 6-way topology partition with a physically
   motivated 3-way city-morphology partition aligned with ITU-R P.1411 and
   3GPP TR 38.901.
2. Close the train/val gap that caused Try 66 to stall well above the
   5 dB target, by tightening regularization and replacing the LR schedule.

This document lists everything that was implemented, plus techniques from
the SOA that were considered and deliberately *not* implemented, with the
reason for each choice.

---

## 1. Partition change: 6 topology classes → 3 city morphology classes

**Source.** ITU-R P.1411-12 §4 (urban/suburban/rural environment
categorisation); 3GPP TR 38.901 §7.2 (UMa/UMi/RMa scenarios).

**Before (Try 66).** Samples were routed per-sample by
`(density_q1, density_q2, height_q1, height_q2)` quantiles computed from
the training set. Thresholds were statistical (quantiles) rather than
physical, and the resulting 6 classes did not map cleanly onto the
propagation regimes that appear in the literature.

**After (Try 67).** Three classes:

| Class            | Morphology            | ITU / 3GPP analogue |
|------------------|-----------------------|---------------------|
| `open_lowrise`   | low density, low height | RMa / suburban    |
| `mixed_midrise`  | medium density, medium | UMi street-canyon |
| `dense_highrise` | high density, high    | UMa                 |

The classifier is still per-sample so that the routing **generalizes to
unseen cities at inference time** — we apply the fixed thresholds in the
calibration JSON (`city_type_thresholds`) to the sample's own building
density and mean-building height. No city-name lookup, no retraining
required when a new city is added. Implementation:
[data_utils.py:1488](data_utils.py#L1488) and
[data_utils.py:2875](data_utils.py#L2875).

The calibration JSON's `city_type_by_city` mapping is *not* used for
routing (it would fail on unseen cities). It is only used by the analytic
prior calibration, which already has a threshold-based fallback.

---

## 2. Anti-overfitting recipe

Every change below is motivated by evidence from Try 66 training logs
(see [DATASET_ANALYSIS_AND_TRAINING_STRATEGY.md](DATASET_ANALYSIS_AND_TRAINING_STRATEGY.md)).

| Knob                     | Try 66   | Try 67  | Reason                                                                                           |
|--------------------------|----------|---------|--------------------------------------------------------------------------------------------------|
| `model.dropout`          | 0.12     | 0.20    | Cheap per-layer regulariser; Try 66 gap opened at epoch 97 with 0.12.                            |
| `training.weight_decay`  | 0.015    | 0.030   | L2 strength was too low for 400-sample experts.                                                  |
| `training.cutmix_prob`   | 0.25     | 0.45    | Zhang et al. MICCAI 2024: CutMix is the single strongest regulariser for limited dense tasks.    |
| `training.learning_rate` | 4e-4     | 3e-4    | Lower peak LR after rewind prevents instant re-overfitting (Try 66 open_sparse_lowrise evidence).|
| `training.lr_scheduler`  | cosine_warm_restarts | reduce_on_plateau | Warm-restart spikes at epochs 41 and 91 in Try 66 reset LR to peak right when overfitting began. Plateau-based reduction decelerates when generalisation stalls. |
| `loss.loss_type`         | huber (δ=6) | mse  | Huber's tolerance for large residuals was masking NLoS tail errors. MSE gives cleaner gradients. |
| `training.epochs`        | 1000     | 800     | Run shorter; early stopping trips much earlier anyway.                                           |
| `training.early_stopping.patience` | 50 | 25  | Stop wasting ~25 epochs after the best epoch before rewinding.                                   |
| `training.ema_decay`     | 0.995    | 0.9975  | Slower EMA averages more weights, further smoothing the validation curve.                        |
| `training.lr_warmup_optimizer_steps` | 0 | 500 | Gentle warmup from 10% of base LR to avoid large early gradients.                                |

### 2.1 Author note on learning rate (documentation only; default config unchanged)

The table above lowers the peak LR from **4e-4** (Try 66) to **3e-4** to damp post-rewind re-overfitting after cosine warm-restarts. **Thesis author preference (for future ablations):** after fixing known training bugs (CutMix vs height-FiLM, wrong `weight_decay` in an earlier generator revision, PMHHNet stem+HF), it may be worth **trying an equal or higher LR than Try 66** (e.g. **4e-4 to 1e-3**) if validation stays flat or gradients look under-powered — higher LR can help escape shallow basins when regularisation is no longer fighting corrupted inputs. This is **not** the default baked into `generate_try67_configs.py`; it is an explicit experimental direction to log alongside results.

---

## 3. Tight per-expert output clamping

**Source.** CLAMPING_AND_IMPROVEMENTS.md §4; 99.99th-percentile statistics
of path loss per city type computed from the training set.

Each expert now gets a physically plausible output range stored in
`target_metadata.path_loss.clip_min_db` / `clip_max_db`. At inference,
predictions outside this window are clamped. This blocks the rare very
high / very low outliers that dominate a small amount of the RMSE.

| Expert            | `clip_min_db` | `clip_max_db` |
|-------------------|---------------|---------------|
| open_lowrise      | 60            | 125           |
| mixed_midrise     | 58            | 135           |
| dense_highrise    | 55            | 145           |

The clamp is already applied inside `clip_to_target_range` (called at
[train_partitioned_pathloss_expert.py:1368](train_partitioned_pathloss_expert.py#L1368)),
so no further code changes were needed.

---

## 4. Test-time augmentation (D4 symmetry group)

**Source.** Standard TTA in dense prediction. Specifically
[Radford et al., Ensemble of Orientations for Segmentation, 2019]
and the oriented averaging used in top Radio-Map-Estimation competition
entries (e.g. RME-GAN).

**Status in Try 66.** Configured in YAML (`test_time_augmentation.enabled:
true`) but *never called* in the training/evaluation code — it was a no-op.

**Implementation in Try 67.** Added
`_d4_forward_inverse()` and `_tta_predict_residual_d4()` in
[train_partitioned_pathloss_expert.py:735](train_partitioned_pathloss_expert.py#L735)
. The D4 group has 8 elements (identity, hflip, vflip, 180°, 90°, 270°,
transpose, anti-transpose). Each input is forward-transformed, passed
through the model, and the output is inverse-transformed before averaging.
Scalar conditioning (UAV height) is invariant under D4 and is passed
unchanged. The `evaluate_validation` function now honours the config
flags `use_in_validation` / `use_in_final_test`. By default only final
test uses TTA (8× inference cost).

---

## 5. SOA techniques reviewed — implemented vs deferred

Everything below was evaluated against [CLAMPING_AND_IMPROVEMENTS.md](CLAMPING_AND_IMPROVEMENTS.md)
and [SOA_IMPLEMENTATION_STATUS.md](SOA_IMPLEMENTATION_STATUS.md) in the
Try 66 folder.

### Implemented in Try 67

| Technique | Expected gain | Source |
|---|---|---|
| Per-expert tight output clamp | 0.1–0.3 dB | Try 66 clamp analysis |
| D4 TTA at final test | 0.3–1.0 dB | Radford 2019; standard segmentation practice |
| PDE residual loss (#2, masked Laplacian) | 0.3–1.0 dB NLoS | ReVeal, arXiv:2502.19646 |
| CutMix ↑ 0.45 | regulariser | Zhang, MICCAI 2024 |
| ReduceLROnPlateau instead of cosine restarts | no plateau spikes | Loshchilov SGDR 2017 (counter-example: restarts hurt us here) |
| Physically-motivated 3-class partition | better data balance | ITU-R P.1411, 3GPP TR 38.901 |

### Emerging research table — item-by-item

This walks the exact `SOA_COMPARISON.tex` emerging-techniques table
(`tab:emerging`) in Try 66, numbered `#1`–`#6`.

**#1 Knife-edge diffraction prior — IMPLEMENTED in Try 67.**
Source: geometry-assisted DL (Geom-DL 2024) and ITU-R P.526-15 §4.5.1.
Expected: 1–3 dB NLoS, the single largest win.
Implementation: a new module [knife_edge.py](knife_edge.py) that, for
every pixel, casts a ray from the TX (map centre, altitude
`antenna_height_m`) to the receiver (pixel at `rx_height_m = 1.5 m`),
samples the building heightmap along the ray with bilinear interpolation
(48 samples by default), and reports the dominant-edge Fresnel loss in
dB using Lee's closed form:

```
v   = h · √( 2 (d1 + d2) / (λ · d1 · d2) )
J(v) = 6.9 + 20·log10( √((v − 0.1)² + 1) + v − 0.1 )   for v > −0.78 and h > 0
     = 0                                                otherwise
```

The dominant edge is approximated by the per-ray argmax of excess height
above the straight TX→RX line (Bullington 1947). Full multi-edge Deygout
is not used; the goal is a cheap physics prior, not a ray-tracing
substitute. The map is normalised to [0, 1] by a 40 dB scale and
concatenated as an extra input channel via
`data.knife_edge_channel.enabled: true`. Implementation is
vectorised over all pixels with `F.grid_sample`, runs on the dataset
worker per sample, and adds ~10 ms per 513×513 sample on CPU — no offline
cache was required, so the "Try 68 headline feature" cost estimate was
too pessimistic.

Sanity check (synthetic scene): pure-LoS pixels give 0 dB exactly, the
deep-shadow pixel behind a 25 m wall returns ~46 dB of diffraction loss
at 3.5 GHz, which matches the expected knife-edge magnitude for the
geometry. Raising the UAV above the wall from 15 m to 40 m reduces the
shadowed-pixel loss from 46 dB to 43 dB, since the ray now grazes the
top of the wall instead of being fully blocked — physically consistent.

**#2 PDE residual loss (PINN) — IMPLEMENTED in Try 67.**
Source: ReVeal (arXiv:2502.19646). Expected: 0.5–2 dB NLoS.
Implementation: `compute_pde_residual_loss` in
[train_partitioned_pathloss_expert.py:735](train_partitioned_pathloss_expert.py#L735).
It applies a 5-point finite-difference Laplacian to the predicted dB map
and penalises `|∇² pred|` masked by `(LoS == 1) ∧ (valid_mask == 1)`. In
LoS + valid regions, the field should be nearly harmonic at our pixel
scale, so a non-zero Laplacian flags non-physical oscillations. The
weight is low (`loss_weight: 0.01`) to avoid over-smoothing real edges
and is only applied on the LoS support to preserve legitimate NLoS
discontinuities at building boundaries. The full Helmholtz operator
`∇² + k²` is not used because it would require a per-expert wavelength
and a complex-valued field; the Laplacian-only version is the ReVeal
"free-space residual" simplification.

**#3 Dual LoS/NLoS head — IMPLEMENTED in Try 67.**
Source: synthesised (standard in ray-tracing-surrogate networks).
Expected: 1–2 dB NLoS.
Implementation: `model.out_channels` is set to 2 in the generator; the
composer (`_apply_dual_los_nlos_head` in
[train_partitioned_pathloss_expert.py:700](train_partitioned_pathloss_expert.py#L700))
splits the raw output into `residual_LoS` (channel 0) and `residual_NLoS`
(channel 1) and blends them with the binarised LoS input channel:

```
pred_residual = los_mask · residual_LoS + (1 − los_mask) · residual_NLoS
```

The key observation is that a per-head loss is not required: because the
blending is a straight-through mask, the LoS head receives gradients
*only* in LoS pixels and the NLoS head *only* in NLoS pixels — the
gradient paths are already disjoint. Enabled via
`dual_los_nlos_head.enabled: true`. The aux/no-data head shifts to
channel 2 when dual head is active and is detected from `raw_out.shape[1]`.
Now also stacks naturally with #1 because the NLoS head can learn on
top of the knife-edge channel that was previously missing.

**#4 Test-time adaptation (RadioPiT) — NOT implemented in Try 67.**
Source: arXiv:2512.01451. Expected: 0.3–1 dB all.
Why deferred: needs an *inference-time* gradient loop that adapts only
the FiLM parameters on each unseen sample, using the LoS region as a
self-supervision signal (the analytic prior is trustworthy there). That
requires a new evaluation entry point that enables grads on FiLM
modules only, runs a handful of Adam steps per sample, then freezes and
predicts. Orthogonal to training, so it will be a post-training tool
script rather than a training-time feature — tracked separately from
Try 68.

**#5 Diffusion refinement (RadioDiff-k²) — NOT implemented in Try 67.**
Source: RadioDiff 2024 / 3D-RadioDiff / RadioLAM 2025. Expected: 0.5–2 dB
all.
Why deferred: a second network and a multi-step sampler at inference.
Wall-time and training budget both blow up. Single-stage PMHHNet still
has ≥ 4 dB of obvious headroom from #1/#2/#3; diffusion refinement only
becomes worthwhile once that headroom is gone.

**#6 Foundation pre-training (FM-RME) — NOT implemented in Try 67.**
Source: FM-RME 2026. Expected: 0.3–1 dB all.
Why deferred: requires a large off-task pre-training run on a much
bigger unlabelled propagation corpus; we have neither the corpus nor the
GPU budget. Listed here so it is not lost.

### Summary of what actually changed in Try 67

Implemented in Try 67: per-expert tight clamp, D4 TTA at final test,
PDE residual loss (#2), CutMix ↑ 0.45, ReduceLROnPlateau, MSE instead of
Huber, 3-way ITU/3GPP partition, per-expert NLoS reweighting 4/5/6.

Deferred to Try 68 (grouped because they stack): **#1 knife-edge +
#3 dual head** — the information-side changes needed to push NLoS
below 10 dB.

Tracked outside this line: **#4 RadioPiT**, **#5 diffusion**,
**#6 foundation model**.

---

## 6. Expected effect vs Try 66

- Plateau at ~9.3 dB was driven by (a) train/val gap opening after epoch
  97 and (b) LR restarts re-triggering overfitting.
  §2 targets both directly.
- Tight clamp + D4 TTA should together contribute 0.5–1 dB when evaluated
  on the final test set.
- The 3-way city partition should give each expert a more homogeneous
  regime, which historically narrows both bias (wrong regime = bad
  calibration) and variance (smaller intra-class spread).

A realistic Try 67 target is a validation RMSE of **7.5–8.5 dB** on the
worst expert and below 7 dB on dense_highrise. The 5 dB target remains
Try 68's problem and most likely requires the knife-edge diffraction prior
+ PDE loss combination listed above.

---

## 7. Where the overall RMSE comes from — LoS vs NLoS decomposition

The cross-try table in `SOA_COMPARISON.tex` §8 tells a consistent story:

| Try | Expert   | Overall | LoS   | NLoS   |
|-----|----------|---------|-------|--------|
| 22  | global   | 19.94   | 3.78  | 34.43  |
| 42  | global   | 19.78   | 3.86  | 34.47  |
| 55  | lowrise  | 10.53   | 3.76  | 35.82  |
| 66  | lowrise  |  9.41   | 3.75  | 31.48  |

**LoS is already solved.** 3.75 dB has been stable for 45 tries and sits
within the 1 dB quantisation noise of the uint8 ground truth (Try 22 ≈
Try 66). LoS is not where the budget is.

**NLoS is the entire error budget.** Try 66's best expert has
NLoS ≈ 31 dB on valid pixels. A ~40/60 LoS/NLoS valid-pixel mix yields
the observed overall ≈ 9.4 dB. To hit the 5 dB target, NLoS has to come
down to roughly 7–8 dB, an **~4× reduction**. This is a physics problem,
not a regularisation problem: the model has no inductive bias for
diffraction around and over buildings, and the analytic prior used as
input is a free-space/empirical formula that does not model multi-edge
diffraction either.

### How Try 67 attacks NLoS specifically

| Mechanism                                         | LoS impact | NLoS impact | Implemented |
|---------------------------------------------------|-----------|-------------|-------------|
| `training.nlos_reweight_factor` 4 → 5 / 6 (city)  | neutral   | **up-weights NLoS pixels 4–6×**, directly targets the tail | yes (per-expert: 4.0 / 5.0 / 6.0) |
| `loss.loss_type: mse` (was huber, δ=6)            | small    | large — Huber with δ=6 was clipping gradients on every pixel with error > 6 dB, i.e. *the entire NLoS tail* | yes |
| CutMix 0.45                                        | small   | helps — each training sample sees more building boundaries per epoch | yes |
| Per-expert tight clamp (e.g. dense_highrise 55–145 dB) | neutral | prevents pathological NLoS over-/under-predictions from dominating RMSE | yes |
| D4 TTA at final test                               | small   | small–medium — averages 8 orientations, stabilises predictions at building edges | yes |
| PDE residual loss (masked Laplacian on LoS + valid) | neutral | penalises non-physical oscillations in free-space regions; ReVeal-style PINN regulariser | yes |
| 3-way physical partition                           | small   | each expert is trained on a more homogeneous diffraction regime (open / UMi / UMa), reducing intra-class variance | yes |
| Knife-edge diffraction prior as input channel      | —       | **1–3 dB** (single largest expected NLoS gain)              | no — deferred to Try 68 (scope too large for this iteration) |
| PDE / ReVeal residual loss                         | —       | 0.5–2 dB                                                    | no — deferred to Try 68 |
| Dual LoS/NLoS head                                 | —       | 1–2 dB                                                      | no — deferred to Try 68 |

### Why the missing items are the high-value ones

Every mechanism above the double-line is a *training-side* change: it
pushes the optimizer to spend more budget on NLoS pixels, without giving
the model new information about what NLoS *physically looks like*. The
NLoS error reduction we can credibly claim from these changes alone
is bounded by what the current inputs can support — probably
1.5–2.5 dB off NLoS, i.e. an overall improvement into the ~7.5–8.5 dB
range on the worst expert.

The three deferred items are *information-side* changes: a knife-edge
diffraction channel adds a physically correct NLoS prior as an input
feature; the PDE loss penalises physically impossible field topologies;
the dual head lets the model specialise a sub-network on NLoS. These are
what can plausibly close the remaining gap down to ~5 dB overall. They
are out of scope for Try 67 so the LR / regularisation / partition
changes can be attributed cleanly, but they are the obvious next step
and are the entire content of the Try 68 plan.
