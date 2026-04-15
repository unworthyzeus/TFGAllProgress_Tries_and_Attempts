# Try 67 — The 3-Expert City-Morphology Try

`Try 67` is the current synthesis of everything learned across the prior tries
and a focused literature review of the strongest recent papers in path-loss and
radio-map prediction. It keeps the height-aware PMHHNet core from the previous
synthesis, but replaces the old 6-way topology partition with 3
city-morphology experts and the current Try67 training recipe.

The goal is not to add one more isolated hypothesis. It is to combine every
proven ingredient from the project history with the most promising novel ideas
from the 2024-2026 literature into a single coherent pipeline.

## Core innovation: continuous multi-altitude prediction

The central contribution of this project is a **single model that predicts
path loss at arbitrary continuous antenna heights** — not by routing to
height-specific models, but by deeply conditioning the entire network on the
height scalar.

This matters because:

- **Real UAV deployments operate at continuously varying altitudes.** Discrete
  height bins (e.g. 10m / 30m / 100m) miss the smooth physical transition
  between regimes. At height `h`, LoS probability is a smooth sigmoid-like
  function of elevation angle θ (Al-Hourani et al., IEEE WCL 2014; 3GPP TR
  38.901 Table 7.4.2-1). The NLoS path-loss exponent also varies continuously
  with height (2.5-3 at high altitude → 4.5-5.4 at low altitude; Borhani et
  al., arXiv:2511.10763, 2025).

- **No existing radio-map neural network handles height this deeply.** PMNet
  (Lee et al., 2022) and RadioUNet (Levie et al., 2019) predict at fixed Tx
  heights. CKMImageNet (2025) includes height as metadata but uses it only for
  dataset partitioning. SenseRay-3D (2025) does 3D voxel prediction but for
  indoor scenes. RadioLAM (2025) uses MoE for different heights but treats
  them as separate generation targets. Our approach conditions a single model
  at every layer.

- **The conditioning mechanism follows the state of the art from generative
  modeling.** Diffusion models (DDPM, Ho et al. 2020; ADM, Dhariwal & Nichol,
  arXiv:2105.05233, 2021) showed that the right way to condition on a
  continuous scalar is: sinusoidal positional encoding → learned MLP → per-layer
  FiLM modulation. We apply this exact pattern to antenna height.

### How height enters the model (7 levels)

| Level | Mechanism | Where |
|---|---|---|
| 1 | **Physical prior** | Formula-prior channel (two-ray, COST231) uses h_tx directly |
| 2 | **Tx-depth map** | Per-pixel `building_height - h_tx`: signed distance to antenna horizon |
| 3 | **Elevation angle map** | Per-pixel θ = atan2(h_tx - h_rx, d): governs LoS probability |
| 4 | **Sinusoidal embedding** | h_tx → sin/cos at 32 frequencies (64-dim), resolving ~0.3m differences |
| 5 | **Per-layer FiLM** | Height modulates features at stem + 4 encoder stages + context + HF + fusion |
| 6 | **SE channel attention** | Per-channel recalibration is height-aware (operates on FiLM'd features) |
| 7 | ~~Stage 2 height FiLM~~ | **Disabled by default** — single-stage 513x513 direct training; the tail refiner exists in code but usually adds <0.5 dB |

This is qualitatively different from simply adding height as an input channel
(which projects it through one convolution and loses it in the encoder). The
sinusoidal → FiLM approach means height information influences every
computational stage of the network, at every spatial resolution.

### Why sinusoidal encoding matters

Raw scalar → MLP (the old approach) maps nearby heights to similar embeddings
but cannot distinguish fine differences. Sinusoidal encoding at multiple
frequencies creates a structured, high-dimensional representation where:

- Low frequencies capture the broad regime (ground level vs. rooftop vs. high
  altitude)
- High frequencies resolve fine differences within a regime (28m vs 32m)
- The encoding is smooth and periodic, so the model can interpolate between
  training heights and partially extrapolate beyond them

This is the same insight that makes NeRF work for novel view synthesis (Mildenhall
et al., ECCV 2020) and DDPM work for continuous-time diffusion (Ho et al., 2020).

## Design philosophy

The design follows three rules:

1. **Keep everything that already worked.** If a decision was validated across
   multiple tries, it stays.
2. **Add only ideas backed by published evidence.** Every new ingredient is
   traced to at least one paper.
3. **Remove anything that was tried and failed.** GANs, pure grokking, explicit
   NLoS loss reweighting, topology edge weighting, and bottleneck attention are
   not included.

## What is kept from the project history

These ingredients are retained because they were validated across multiple tries:

| Ingredient | First proven | Evidence |
|---|---|---|
| City-morphology experts (3 classes) | Try 67 | Per-expert RMSE is consistently lower than monolithic models |
| PMHHNet backbone (PMNet + HF branch + FiLM) | Try 54 | Custom architecture for height-aware propagation |
| Calibrated physical prior + residual learning | Try 41-42 | Anchors the model to physics; LoS RMSE drops to ~3-4 dB |
| Building mask exclusion | Try 33 | Building pixels are not valid receivers |
| `city_holdout` data splitting | Try 51 | Geography-aware generalization, not city memorization |
| EMA (decay 0.99) | Try 55 | Stabilizes validation and checkpoint selection |
| Group norm | Try 22 | Works with `batch_size = 1-2` unlike batch norm |
| Geometric augmentation (hflip, vflip, rot90) | Try 22+ | Standard practice in all radio-map papers |
| ~~Two-stage coarse-to-fine~~ (disabled in default Try67 configs) | Try 62-64 | Refiner added <0.5 dB gain; single-stage 513 is simpler and at full-res |
| Multiscale path-loss loss | Try 22 | Enforces correct low-frequency structure |
| ~~No-data auxiliary BCE head~~ (removed) | Try 54 | Removed: trivial task (99% acc in epoch 2), wastes gradient capacity; building mask given as input instead |
| Supervised regression only (no GAN) | Try 51 | Literature consensus; GANs are the exception, not the rule |

## What is new in Try 67 — Paper-backed additions

### 1. Sinusoidal height embedding with per-layer FiLM (the core height upgrade)

**Source:** Ho et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020.
Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis,"
arXiv:2105.05233, NeurIPS 2021.
Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer,"
arXiv:1709.07871, AAAI 2018.

**What it does:** Replaces the old raw-scalar → MLP height embedding with:

1. A sinusoidal positional encoding that maps the 1-D height scalar to a
   64-dimensional multi-frequency representation (32 sin/cos pairs with
   exponentially spaced frequencies from period=1 to period=1000).
2. A 2-layer MLP that projects this to the conditioning vector.
3. FiLM modulation at **8 points** in the network (stem + 4 encoder stages +
   context bottleneck + HF branch + final fusion), up from 4 in the previous
   version.

**Why it matters:** The old approach (raw scalar → MLP) could not resolve fine
height differences (e.g. 28m vs 35m mapped to nearly identical embeddings). The
sinusoidal encoding creates a structured high-dimensional representation where
different frequency bands capture different scales of variation. This is the
same technique that makes NeRF (Mildenhall et al., 2020) and DDPM (Ho et al.,
2020) work for continuous variables.

Per-layer FiLM means height information influences computation at every
spatial resolution, not just at the stem and bottleneck. This follows the
ADM (Dhariwal & Nichol 2021) pattern where the timestep embedding modulates
every residual block.

### 2. Elevation angle map as explicit input channel

**Source:** Al-Hourani et al., "Optimal LAP Altitude for Maximum Coverage,"
IEEE WCL 2014. 3GPP TR 38.901, Table 7.4.2-1 (height-dependent LoS
probability). Borhani et al., "Millimeter-Wave UAV Channel Model with
Height-Dependent Path Loss," arXiv:2511.10763, Nov 2025.

**What it does:** Computes per-pixel elevation angle θ(x,y) = atan2(h_tx -
h_rx, d(x,y)) normalized to [0, 1]. This is the primary physical variable
governing LoS probability — not distance alone, not height alone, but the
angle of incidence.

**Why it matters for us:** The LoS probability in the Al-Hourani/3GPP model
is: P_LoS(θ) = 1 / (1 + C·exp(-B·(θ - C))). This is a direct function of
elevation angle. By providing θ as an input, we give the model direct access
to the physically governing variable. Combined with the Tx depth map, this
provides a complete height-aware geometric context.

### 3. ~~Height-aware Stage 2 refiner~~ → Disabled in default Try67 configs (single-stage 513x513)

**Decision:** The two-stage pipeline (256→513 refiner) is kept out of the
default Try67 configs in favor of direct 513x513 training. Across Try 49-64,
the Stage 2 tail refiner
consistently added only 0.2-0.5 dB improvement — below the 1 dB quantization
floor of the uint8 ground truth. The added complexity, error propagation risk,
and doubled training time were not justified.

**What replaces it:** The PMHHNet backbone trains directly at 513x513 with
batch_size=1 and gradient accumulation (effective batch=4). This preserves
full-resolution detail from the start and simplifies the entire pipeline.

### 4. Propagation corridor weighting map

**Source:** Gao et al., "Effective outdoor pathloss prediction: A multi-layer
segmentation approach with weighting map," arXiv:2601.08436, Jan 2026.

**What it does:** A Gaussian-inverse-square spatial weighting mask that
emphasizes pixels near the direct Tx-Rx path. The ablation in the paper shows
improvements of **0.55 dB** at 800 MHz, **0.97 dB** at 7 GHz, and **1.16 dB**
at 28 GHz.

**Try 67 status:** The corridor map is **not** computed or applied in the current `data_utils.py` / trainer (it was YAML-only and removed so configs match code). The following bullets describe the **intended** Gao-style recipe if you add it later.

**How it would be implemented:** At data loading time, compute a soft 2D weight map for
each sample where:

- pixels along the direct line between Tx and any Rx get the highest weight
- weight decays with Gaussian profile perpendicular to the direct path
- weight decays with inverse-square along the path axis

This map is used as a **per-pixel loss weight** during training, so the model
focuses more on the propagation corridor where most signal energy actually
travels. This is a novel addition not present in any of the 65 prior tries.

**Why it matters for us:** The biggest remaining error is in NLoS regions. But
even in NLoS, most signal energy arrives through reflections near the direct
path corridor. Weighting the loss toward this corridor helps the model
prioritize the physically most relevant regions.

### 5. Tx-relative depth map as explicit input channel

**Source:** Gao et al., arXiv:2601.08436, Sec. III-A.

**What it does:** Instead of just a binary LoS mask, compute per-pixel
`building_height - antenna_height`. Positive values = antenna is below
buildings (likely NLoS with strong shadow). Negative values = antenna is above
buildings (likely LoS). Zero = ground level.

**How we implement it:** In the data pipeline, compute:

```
tx_depth_map = topology_map_meters - antenna_height_m
```

This is added as an extra input channel alongside the existing topology map,
LoS mask, and distance map. The existing binary LoS mask says "blocked or not",
but the depth map says "by how much", which is a much richer signal for
predicting shadow depth and diffraction loss.

**Why it matters for us:** The project has topology heights and antenna heights
but never explicitly computed their difference as a spatially resolved input.
This is a direct, cheap-to-compute physically meaningful feature.

### 6. Huber loss for NLoS robustness

**Source:** Ribeiro et al., "Residual-based Adaptive Huber Loss (RAHL) —
Design of an improved Huber loss for CQI prediction in 5G networks,"
arXiv:2408.14718, Aug 2024.

**Background source:** Huber (1964), "Robust Estimation of a Location
Parameter," Annals of Mathematical Statistics.

**What it does:** Instead of pure MSE, use Huber loss with a learned or
fixed `delta` threshold. Below `delta`, the loss is quadratic (like MSE),
giving smooth gradients for small errors. Above `delta`, the loss is linear
(like MAE), preventing extreme NLoS outlier pixels from dominating the
gradient.

**How we implement it:** Replace the main `MSE` loss term with
`torch.nn.SmoothL1Loss(beta=delta)` where `delta = 8.0 dB`. This threshold
was chosen because:

- LoS pixels have ~3-6 dB error (always below delta → quadratic)
- Easy NLoS pixels have ~10-15 dB error (near delta → transition)
- Hard NLoS pixels have 30-50 dB error (far above delta → linear, dampened)

This prevents the optimizer from being overwhelmed by the extreme NLoS tail
while still learning from it.

**Why it matters for us:** Across 65 tries, the NLoS RMSE has stubbornly
remained at 34-41 dB while LoS is at 3-6 dB. Pure MSE squares those 40+ dB
errors, creating enormous gradients that destabilize training. Huber loss
tames the tail while preserving sensitivity to LoS precision.

### 7. Cosine annealing with warm restarts (SGDR)

**Source:** Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with
Warm Restarts," arXiv:1608.03983, ICLR 2017.

**What it does:** Instead of ReduceOnPlateau (reactive, often too late) or
constant LR (no annealing), use a cosine schedule that smoothly decays the LR
from `lr_max` to `lr_min` over each cycle, then restarts. Each restart lets
the optimizer escape shallow local minima.

**How we implement it:** `CosineAnnealingWarmRestarts` with:

- `T_0 = 30` epochs (first cycle length)
- `T_mult = 2` (each cycle doubles: 30, 60, 120, ...)
- `eta_min = 1e-6`
- Initial LR = `5e-4`

This gives a natural curriculum: early cycles are short and exploratory, later
cycles are long and refining.

**Why it matters for us:** The project has tried constant LR (Try 65, slow),
ReduceOnPlateau (Try 64, reactive), and step decay (old tries). Cosine
annealing is now the standard in modern deep learning and has not been tested
in this project.

### 8. Lightweight Squeeze-and-Excitation (SE) attention in the encoder

**Source:** Hu et al., "Squeeze-and-Excitation Networks," CVPR 2018,
arXiv:1709.01507.

**Updated context:** SCSA (Spatial and Channel Synergistic Attention),
arXiv:2407.05128, Jul 2024, confirms that channel attention remains a
top-performing lightweight module for dense prediction.

**What it does:** After each encoder stage in PMHHNet, add a tiny SE block
that:

1. Global-average-pools the feature map to a channel descriptor
2. Passes through a 2-layer MLP bottleneck (reduction ratio 4)
3. Produces per-channel scaling factors via sigmoid
4. Recalibrates the feature map by multiplying each channel

This costs < 0.1% extra FLOPs but lets the model learn which feature channels
are most important at each spatial scale.

**How we implement it:** Add `SEBlock` after each `EncoderStage` in
`model_pmhhnet.py`. The block is ~10 lines of code.

**Why it matters for us:** Try 25 tested bottleneck attention (a heavier
global-context module) and it didn't help. SE is much lighter and acts on
channel importance, not spatial context. It has never been tested.

### 9. Test-time augmentation (TTA) at inference

**Source:** General practice in computer vision dense prediction. Confirmed
effective in recent radio-map work by PMNet (4-way rotation augmentation).

**Formalized in:** Shanmugam et al., "Test-time Augmentation Improves
Efficiency in Conformal Prediction," CVPR 2025.

**What it does:** At inference, run the model on the original input plus
geometric transforms (hflip, vflip, rot90 × 3), reverse the transforms on
the outputs, and average. This gives a free 0.1-0.5 dB improvement with no
retraining.

**How we implement it:** In `predict.py`, add a TTA wrapper that averages
predictions from 8 geometric transforms (D4 symmetry group: identity + 3
rotations + 4 flips).

**Why it matters for us:** Training augmentations have been used since Try 22
but TTA at inference has never been implemented in this project.

## Architecture summary

```
Input (9 channels):
  topology_map + los_mask + distance_map + formula_prior + confidence +
  tx_depth_map + elevation_angle_map + building_mask + knife_edge_channel

       ↓ [per-sample routing by city morphology thresholds]

City-morphology router → 1 of 3 experts:
  - open_lowrise     (3,475 samples total; base_ch=40, hf_ch=16)
  - mixed_midrise    (2,590 samples total; base_ch=44, hf_ch=18)
  - dense_highrise   (10,115 samples total; base_ch=48, hf_ch=20)

Height conditioning pipeline:
  antenna_height_m
    → sinusoidal positional encoding (64-dim, 32 freq bands)
    → 2-layer MLP (SiLU) → conditioning vector
    → per-layer FiLM γ,β at 8 network stages

Each expert — Single-stage 513×513:
  PMHHNet + sinusoidal FiLM + SE attention
  - ResNet-style encoder with per-stage FiLM + SE blocks
  - Dilated context module (ASPP-like) + FiLM
  - FPN-style top-down fusion
  - HF Laplacian branch + FiLM
  - 8 FiLM injection points
  - Predicts: residual over calibrated prior (1 output channel)

Loss:
  - MSE loss with NLoS pixel reweighting (4.0× / 5.0× / 6.0× by expert)
  - Dual LoS/NLoS head: two residual branches are blended by the LoS mask
  - NLoS focus auxiliary: enabled in Try67 as a conservative RMSE tail loss
    (`loss_weight` 0.15 / 0.20 / 0.25 by expert) so the NLoS branch gets
    direct supervision on top of the masked full-map objective
  - Multiscale loss at scales [2, 4]
  - LR warmup (500 optimizer steps, linear from 10% to 100%)
  - ReduceLROnPlateau after warmup
  - CutMix augmentation (p=0.45)
  - Gradient accumulation (8 steps)

Inference:
  - TTA (D4 group, 8 geometric transforms)
  - Final prediction = prior + residual
```

### City routing and sample counts

Try 67 routes each sample by its own topology statistics, not by city name. The
routing uses `city_type_thresholds` from
`prior_calibration/regime_obstruction_train_only_from_try47.json`:

- `density = mean(topology_map != 0)` over the sample
- `height = mean(non-ground topology values)` over the same pixels
- `open_lowrise` if `density <= q1` and `height <= h1`
- `dense_highrise` if `density >= q2` or `height >= h2`
- otherwise `mixed_midrise`

Thresholds:

| Threshold | Value |
|---|---:|
| `density_q1` | 0.1956760965207146 |
| `density_q2` | 0.2549084200418514 |
| `height_q1` | 10.913664557109195 m |
| `height_q2` | 15.950557886832314 m |

The `city_type_by_city` map in the calibration JSON is kept for analysis and
prior calibration only; routing itself is threshold-based so it generalizes to
unseen cities.

Dataset coverage on `CKM_Dataset_270326.h5` (16,180 samples; city_holdout,
seed 42, val/test 15%/15%):

| Expert | Total samples | Train | Val | Test |
|---|---:|---:|---:|---:|
| `open_lowrise` | 3,475 | 2,335 | 550 | 590 |
| `mixed_midrise` | 2,590 | 1,735 | 425 | 430 |
| `dense_highrise` | 10,115 | 6,900 | 1,655 | 1,560 |

## Training recipe

| Parameter | Value | Justification |
|---|---|---|
| Resolution | **513×513 direct** | Single-stage; Stage 2 is disabled in the default Try67 configs (added <0.5 dB in Try 49-64) |
| Optimizer | AdamW | Standard for modern training; decoupled weight decay |
| Learning rate | 3e-4 | Peak LR after 500-step warmup |
| Weight decay | 3e-2 | Higher regularization for the 3-city experts |
| LR schedule | ReduceLROnPlateau | `factor=0.5`, `patience=8`, `min_lr=1e-6`, 500-step linear warmup from 10% of base LR |
| Loss (main) | MSE | Current residual objective for the Try67 experts |
| Loss (multiscale) | Enabled, scales [2, 4], weight 0.3 | Proven in Try 22; enforces low-frequency correctness |
| ~~Loss (no_data)~~ | Removed | Trivial task; building mask given as input channel instead |
| Loss (corridor) | Propagation corridor weighting | From Gao et al. 2026; per-pixel loss weights |
| NLoS reweight | 4.0× / 5.0× / 6.0× on NLoS pixels | Expert-specific compensation for the LoS/NLoS imbalance |
| CutMix | p=0.45, Beta(1,1) | Strong regularizer for limited data (MICCAI 2024) |
| EMA decay | 0.9975 | Slightly slower averaging for stability |
| Batch size | 1 (grad accum 8) | 513×513 images; accumulation simulates a larger effective batch |
| base_channels | 40 / 44 / 48 by expert | Capacity scaled to the 3 city-morphology groups |
| hf_channels | 16 / 18 / 20 by expert | High-frequency Laplacian branch scales with expert width |
| Epochs | 800 | Long enough for plateau scheduling and early stopping |
| Dropout | 0.20 | Higher regularization for the current expert split |
| GAN | Disabled | Literature consensus; supervised regression is the default |
| Augmentation (train) | hflip, vflip, rot90 (p=0.5 each) | Verified safe: sync aug applies same transform to all spatial channels |
| Augmentation (test) | D4 TTA (8 transforms) | Free improvement at inference |

## Why single-stage 513×513 instead of coarse-to-fine

Tries 62-64 used a two-stage pipeline (256→513 with UNetResidualRefiner).
Try 67 keeps Stage 2 disabled in the default configs:

1. **Marginal gain:** Stage 2 consistently added only 0.2-0.5 dB across all
  experts and tries. This is within the uint8 quantization noise (1 dB).
2. **Quantization ceiling:** The refiner tries to correct errors of ~1-5 dB
   on a ground truth with 1 dB resolution. It cannot reliably learn corrections
   smaller than the quantization step.
3. **Complexity cost:** Two-stage doubles training time, requires careful
   checkpoint management (Stage 1 must finish before Stage 2), and propagates
   errors (bad Stage 1 → bad Stage 2 teacher).
4. **Direct training is now feasible:** With batch_size=1 + gradient
   accumulation (effective batch=4), 513×513 training fits in 4 GPUs and
   preserves high-frequency detail from the start.

## What was explicitly excluded and why

| Excluded idea | Why |
|---|---|
| GAN / adversarial training | Tried extensively (Try 1-41); literature says it is the exception (arXiv:2401.08976) |
| Stage 2 tail refiner | Code exists, but default Try67 configs keep it disabled; gain was only 0.2-0.5 dB and training cost doubled |
| Stochastic depth / DropPath | Requires deep architectural changes; CutMix + dropout suffice for regularization |
| Topology edge weighting | Tried in Try 27; informative but did not beat Try 22 |
| Bottleneck attention (global context) | Tried in Try 25; did not beat Try 22 |
| Pure grokking / no early stopping | Tried in Try 65; slow convergence, poor early results |
| Obstruction proxy channels | Mixed results in Try 62-63; adds complexity without clear gain |
| Radial loss | Tried in Try 29; did not beat Try 22 |
| Regime reweighting during training | Tried in Try 51, 61; replaced by NLoS pixel reweighting (simpler, more direct) |

**Re-included from previous exclusions:**
- **NLoS pixel reweighting**: Try 61 used *regime-level* reweighting (by city type),
  which was too coarse. Try 67 uses *pixel-level* NLoS reweighting (expert-specific
  weights on NLoS pixels identified by the LoS mask channel), which directly
  addresses the 10:1 LoS/NLoS gradient imbalance.

## Full source references

### Papers directly informing the new additions

0a. Ho et al., "Denoising Diffusion Probabilistic Models"
    - NeurIPS 2020
    - https://arxiv.org/abs/2006.11239
    - **Used for:** sinusoidal positional encoding of continuous scalar (height)

0b. Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis"
    - arXiv:2105.05233, NeurIPS 2021
    - https://arxiv.org/abs/2105.05233
    - **Used for:** per-layer FiLM conditioning pattern (every residual block
      modulated by the scalar embedding), architecture design of sinusoidal →
      MLP → FiLM pipeline

0c. Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer"
    - arXiv:1709.07871, AAAI 2018
    - https://arxiv.org/abs/1709.07871
    - **Used for:** FiLM affine modulation γ·x + β formulation

0d. Al-Hourani et al., "Optimal LAP Altitude for Maximum Coverage"
    - IEEE Wireless Communications Letters, 2014
    - https://ieeexplore.ieee.org/document/6863654
    - **Used for:** elevation-angle-dependent LoS probability model motivating
      the elevation angle map input channel

0e. 3GPP TR 38.901, "Study on channel model for frequencies from 0.5 to 100 GHz"
    - Table 7.4.2-1: height-dependent LoS probability
    - **Used for:** physics justification that LoS probability is fundamentally
      a function of elevation angle, not distance alone

0f. Borhani et al., "Millimeter-Wave UAV Channel Model with Height-Dependent
    Path Loss and Shadowing in Urban Scenarios"
    - arXiv:2511.10763, Nov 2025
    - https://arxiv.org/abs/2511.10763
    - **Used for:** evidence that NLoS PLE decreases from 4.5-5.4 at low
      altitude to 2.5-3 at high altitude, shadow fading reduces with height

0g. Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields
    for View Synthesis"
    - ECCV 2020
    - https://arxiv.org/abs/2003.08934
    - **Used for:** demonstrating that sinusoidal positional encoding of
      continuous variables enables neural networks to learn high-frequency
      functions (directly applicable to height conditioning)

1. Gao et al., "Effective outdoor pathloss prediction: A multi-layer
   segmentation approach with weighting map"
   - arXiv:2601.08436, Jan 2026
   - https://arxiv.org/abs/2601.08436
   - **Used for:** propagation corridor weighting map, Tx depth map input

2. Ribeiro et al., "Residual-based Adaptive Huber Loss (RAHL) — Design of an
   improved Huber loss for CQI prediction in 5G networks"
   - arXiv:2408.14718, Aug 2024
   - https://arxiv.org/abs/2408.14718
   - **Used for:** Huber loss justification for robust NLoS handling

3. Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts"
   - arXiv:1608.03983, ICLR 2017
   - https://arxiv.org/abs/1608.03983
   - **Used for:** cosine annealing with warm restarts LR schedule

4. Hu et al., "Squeeze-and-Excitation Networks"
   - arXiv:1709.01507, CVPR 2018
   - https://arxiv.org/abs/1709.01507
   - **Used for:** SE attention blocks in the PMHHNet encoder

5. Li et al., "TransPathNet: A Novel Two-Stage Framework for Indoor Radio Map
   Prediction"
   - arXiv:2501.16023, ICASSP 2025
   - https://arxiv.org/abs/2501.16023
   - **Used for:** coarse-to-fine two-stage justification, multiscale decoder

6. Zhong et al., "PathFinder: Advancing Path Loss Prediction for
   Single-to-Multi-Transmitter Scenario"
   - arXiv:2512.14150, Dec 2025
   - https://arxiv.org/abs/2512.14150
   - **Used for:** disentangled feature encoding philosophy, mask-guided attention

### Papers confirming the retained project choices

7. Lee et al., "PMNet: Robust Pathloss Map Prediction via Supervised Learning"
   - arXiv:2211.10527
   - https://arxiv.org/abs/2211.10527
   - **Used for:** PMNet backbone architecture, supervised regression as default

8. Lee & Molisch, "A Scalable and Generalizable Pathloss Map Prediction"
   - arXiv:2312.03950
   - https://arxiv.org/abs/2312.03950
   - **Used for:** transfer learning, scenario adaptation, warm-starting

9. Levie et al., "RadioUNet: Fast Radio Map Estimation with Convolutional
   Neural Networks"
   - arXiv:1911.09002
   - https://arxiv.org/abs/1911.09002
   - **Used for:** two-stage training is normal; MSE as default loss

10. Feng et al., "IPP-Net: A Generalizable Deep Neural Network Model for Indoor
    Pathloss Radio Map Prediction"
    - arXiv:2501.06414, ICASSP 2025
    - https://arxiv.org/abs/2501.06414
    - **Used for:** curriculum learning philosophy, UNet as strong baseline

11. Zhang et al., "Path Loss Prediction Based on Machine Learning: Principle,
    Method, and Data Expansion"
    - MDPI Applied Sciences 9(9):1908, 2019
    - https://www.mdpi.com/2076-3417/9/9/1908
    - **Used for:** hybrid physical + ML is standard, not a hack

12. Yapar et al., "The First Pathloss Radio Map Prediction Challenge"
    - arXiv:2310.07658
    - https://arxiv.org/abs/2310.07658
    - **Used for:** RMSE-based benchmarking is the field standard

13. Bakirtzis et al., "The First Indoor Pathloss Radio Map Prediction Challenge"
    - arXiv:2501.13698
    - https://arxiv.org/abs/2501.13698
    - **Used for:** challenge methodology, fair evaluation practices

14. Shrestha et al., "Radio Map Estimation: Empirical Validation and Analysis"
    - arXiv:2310.11036
    - https://arxiv.org/abs/2310.11036
    - **Used for:** warning against more architecture without better physics

15. Gupta et al., "Machine Learning-based Urban Canyon Path Loss Prediction
    using 28 GHz Manhattan Measurements"
    - arXiv:2202.05107
    - https://arxiv.org/abs/2202.05107
    - **Used for:** street-wise generalization, physical feature engineering

16. Qi et al., "ACT-GAN: Radio map construction based on generative adversarial
    networks with ACT blocks"
    - arXiv:2401.08976
    - https://arxiv.org/abs/2401.08976
    - **Used for:** evidence that GANs are the exception in radio-map work

17. Gao et al., (UNet + ASPP for UAV-assisted mmWave pathloss)
    - arXiv:2509.09606
    - **Used for:** ASPP bottleneck for multi-scale context is effective

18. SCSA: Exploring the Synergistic Effects Between Spatial and Channel
    Attention
    - arXiv:2407.05128, Jul 2024
    - https://arxiv.org/abs/2407.05128
    - **Used for:** confirming channel attention (SE-like) is still top-performing

19. RadioDiff-Loc: Diffusion Model Enhanced Scattering Cognition for NLoS
    - arXiv:2509.01875
    - https://arxiv.org/abs/2509.01875
    - **Used for:** NLoS is the main bottleneck everywhere; knife-edge diffraction theory motivation

20. Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits
    - arXiv:2411.18704, Nov 2024
    - https://arxiv.org/abs/2411.18704
    - **Used for:** EMA provides better generalization, calibration, and transfer

### Height-specific research context

21. Li et al., "A Fine-Grained 3D Radio Map Construction Paradigm with
    Ultra-Low Sampling Rates by Large Generative Models" (RadioLAM)
    - arXiv:2509.11571, Sep 2025
    - https://arxiv.org/abs/2509.11571
    - **Context:** Uses MoE + diffusion for 3D radio maps across heights,
      but treats each height as a separate generation target. Our approach
      conditions a single model on height continuously.

22. Yapar et al., "SenseRay-3D: Generalizable and Physics-Informed Framework
    for End-to-End Indoor Propagation Modeling"
    - arXiv:2511.12092, Nov 2025
    - https://arxiv.org/abs/2511.12092
    - **Context:** Uses SwinUNETR for 3D voxel-based path loss, achieving
      4.27 dB MAE on unseen environments. Indoor focused, not UAV A2G.

23. Li et al., "AIRMap: AI-Generated Radio Maps for Wireless Digital Twins"
    - arXiv:2511.05522, Nov 2025
    - https://arxiv.org/abs/2511.05522
    - **Context:** Single U-Net autoencoder for 2D radio maps from elevation
      data. Does not handle height as a continuous conditioning variable.

24. Cheng et al., "RadioDiff-3D: A 3D×3D Radio Map Dataset and Generative
    Diffusion Based Benchmark for 6G Environment-Aware Communication"
    - arXiv:2507.12166, Jul 2025
    - https://arxiv.org/abs/2507.12166
    - **Context:** UrbanRadio3D dataset with 7 height layers. Uses 3D
      convolutions rather than height conditioning.

25. Masrur et al., "Beyond Path Loss: Altitude-Dependent Spectral Structure
    Modeling for UAV Measurements"
    - arXiv:2601.02605, Jan 2026
    - https://arxiv.org/abs/2601.02605
    - **Context:** Altitude-Dependent Spectral Structure Model showing that
      spectral properties transition continuously over altitude ranges.

26. Lee et al., "CKMImageNet: A Dataset for AI-Based Channel Knowledge Map"
    - arXiv:2504.09849, Apr 2025
    - https://arxiv.org/abs/2504.09849
    - **Context:** CKM dataset for radio map prediction. Includes height as
      metadata but does not use continuous height conditioning.

### Project-internal references

- `PATH_LOSS_MODEL_TRAINING_PAPERS.md` — full literature review
- `PATH_LOSS_PRIORITY_NEXT_STEPS.md` — project roadmap
- `VERSIONS.md` — chronological evolution from Try 1 to Try 47
- `TRY67_DESIGN.md` — 3-city-morphology design rationale
- `TRY67_IMPLEMENTATION.md` — Try67 code-path details
- `SOA_IMPLEMENTATION_STATUS.md` — implemented vs missing features

## Current best reference numbers

From cluster outputs:

| Try | Setting | Overall RMSE | LoS RMSE | NLoS RMSE |
|---|---|---|---|---|
| 42 | PMNet + prior (single) | ~19.78 dB | ~3.86 dB | ~34.47 dB |
| 49 | PMNet stage 1 (w112) | ~18.96 dB | — | — |
| 55 | PMHHNet expert (open_sparse_lowrise) | ~10.53 dB | ~3.76 dB | ~35.82 dB |
| 64 | Coarse 128 + refiner (open_sparse_lowrise) | ~7.76 dB (128px) | ~2.97 dB | ~24.91 dB |

The target for Try 67 is to push per-expert RMSE lower than Try 64 while
using a more robust and physically motivated pipeline.

## Key files

- `train_partitioned_pathloss_expert.py` — Single-stage training (513×513)
- `model_pmhhnet.py` — PMHHNet + sinusoidal FiLM + SE attention
- `scripts/generate_try67_configs.py` — Config generator (3 experts + registry)
- `scripts/plot_try67_metrics.py` — Metric plotting (RMSE, LoS/NLoS, gain, loss, timing)
- `cluster/run_sixtyseventh_try67_4gpu.slurm` — Cluster entry
- `cluster/submit_try67_experts_4gpu_sequential.py` — Cluster submission (3-expert sequential run)
- `DATASET_ANALYSIS_AND_TRAINING_STRATEGY.md` — Full data analysis + training strategy documentation

## How to run

### 1. Generate all configs

```powershell
C:\TFG\.venv\Scripts\python.exe C:\TFG\TFGpractice\TFGSixtySeventhTry67\scripts\generate_try67_configs.py
```

### 2. Submit to cluster

```powershell
$env:SSH_PASSWORD = '***'
python C:\TFG\TFGpractice\TFGSixtySeventhTry67\cluster\submit_try67_experts_4gpu_sequential.py
```
