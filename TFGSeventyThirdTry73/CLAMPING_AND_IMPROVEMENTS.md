# Output Clamping and Path to Improvement — Try 73

---

## 1. Actual Path Loss Range in the Dataset

From the full scan of `CKM_Dataset_270326.h5` (16,180 samples, 66 cities,
~4.26 billion pixels total):

| Range (dB) | Pixels | % of valid ground pixels |
|---|---|---|
| < 60 | 0 | 0.00% |
| 60–70 | 3,080 | < 0.001% |
| 70–80 | 3,966,672 | 0.17% |
| 80–90 | 200,402,785 | 8.4% |
| **90–100** | **1,419,225,489** | **59.4%** |
| **100–110** | **686,961,650** | **28.8%** |
| 110–120 | 76,543,438 | 3.2% |
| 120–130 | 1,267,249 | 0.05% |
| 130–140 | ~120,000 | < 0.01% |
| 140–184 | ~37,000 | < 0.001% |

**Key facts:**
- **Minimum value observed:** ~65 dB (UAV at max altitude, closest receiver pixel, 7.125 GHz)
- **Maximum value observed:** ~184 dB (deep urban canyon NLoS at low altitude)
- **99.8% of all valid pixels: 75–125 dB**
- **99.99% of all valid pixels: 70–130 dB**
- The uint8 ground truth has 1 dB resolution — values are integers only

### Physical explanation of the extremes

**Minimum (~65 dB):** Free-space path loss at 7.125 GHz over ~1 m distance is ~49 dB.
At the closest receiver pixel to the Tx with UAV at 478 m height and direct LoS,
the actual minimum seen in the data is ~65 dB. This is physically correct for
~5–10 m 3D distance at 7 GHz.

**Maximum (~184 dB):** At UAV height ~15 m in dense urban highrise, a receiver
that is deep inside a building canyon with multiple walls between it and the Tx
can reach 150–180 dB. These are < 0.01% of pixels and are physically degenerate
cases (essentially no signal).

### Approximate percentile summary

The histogram above is more useful when converted into percentiles, because
clamping decisions should be driven by the **tail mass** rather than by the raw
minimum and maximum. From the binned full-dataset scan, the following percentiles
are a good approximation:

| Percentile | Approx. path loss |
|---|---|
| p0.01 | ~71 dB |
| p0.1 | ~76 dB |
| p1 | ~81 dB |
| p50 | ~97 dB |
| p99 | ~117 dB |
| p99.9 | ~120 dB |
| p99.99 | ~124 dB |

These are **bin-interpolated estimates** from the global histogram, not exact raw
percentiles from the HDF5 values. For the thesis, the stronger version of this
table should be reported separately for:
- all pixels
- LoS pixels only
- NLoS pixels only
- each topology expert

The key conclusion already holds: the dataset mass is concentrated in a very
narrow band around 90–110 dB, and the extreme tail above ~130 dB is tiny.

### Why uint8 precision is a real limitation

Our ground truth is effectively quantized at about **1 dB per level**. This is
not catastrophic for a 9 dB problem, but it is absolutely relevant once we start
discussing the last few dB of improvement.

With quantization step `Delta = 1 dB`, the irreducible uniform-quantization noise is:

```text
sigma_q = Delta / sqrt(12) = 1 / sqrt(12) = 0.2887 dB
```

This means:
- there is a hard floor below which the labels themselves are noisy
- sharp LoS/NLoS transitions are snapped to integer bins
- very small improvements can be invisible in validation even if the model is
  actually learning finer structure

This quantization does **not** explain the current 9.41 dB RMSE, but it does
matter for the final ceiling and for interpreting tiny gains late in training.

For comparison, the ICASSP 2023 challenge dataset also uses `uint8`, but over a
29 dB range, so its quantization step is only `29/255 = 0.1137 dB` per level,
which is far less restrictive.

---

## 2. Output Clamping Proposal

### Current configuration

```yaml
target_metadata:
  path_loss:
    scale: 180.0
    offset: 0.0
    clip_min: 0.0
    clip_max: 180.0
```

This means the model output head spans **0–180 dB** in physical space.
In normalized space (which the model actually uses): 0.0 to 1.0.
The residual added to the prior can theoretically push predictions
to any value in this range, including physically impossible ones.

### Proposed clamp: [50, 150] dB

| Clamp | Approx. GT coverage | Approx. GT excluded | Interpretation |
|---|---|---|---|
| [0, 180] (current) | 100.000% | 0.000% | Wide but physically wasteful |
| [50, 150] | ~99.999% | ~0.001% | Conservative global default |
| [60, 140] | ~99.998% | ~0.002% | Slightly tighter, still very safe |
| [65, 135] | ~99.996% | ~0.004% | Good compromise if we want stronger regularisation |
| [70, 130] | ~99.993% | ~0.007% | Still safe globally, but may be too tight for some dense/highrise tails |

**Recommendation: clamp to [50, 150] dB.**

This is conservative enough to miss no real data (not a single pixel below 60 dB
exists in the dataset) while cutting the model's prediction space from 180 dB to
100 dB — a 44% reduction in the output range.

### Recommended clamp ablation

The right way to defend clamping is not just to argue from histogram tails, but
to report an explicit ablation on existing predictions:

| Candidate clamp | Overall RMSE | LoS RMSE | NLoS RMSE | % predictions clipped | Notes |
|---|---|---|---|---|---|
| none / [0, 180] | baseline | baseline | baseline | 0% | Current setup |
| [50, 150] | TBD | TBD | TBD | TBD | Safest first test |
| [60, 140] | TBD | TBD | TBD | TBD | Moderate tightening |
| [65, 135] | TBD | TBD | TBD | TBD | Stronger global clamp |
| [70, 130] | TBD | TBD | TBD | TBD | Useful stress test |
| per-expert clamp | TBD | TBD | TBD | TBD | Best long-term option |

This ablation requires **no retraining** if saved predictions are already
available. It is the fastest experiment that can turn the clamping proposal
into a quantitative result rather than a qualitative argument.

### Implementation

The clamp should be applied **at inference time only** (validation + test), not
during training. This avoids gradient issues and allows the model to learn the
full distribution.

In `train_partitioned_pathloss_expert.py`, the final prediction after
`prior + residual` is:

```python
# Current (in normalized space, 180 dB scale):
output = torch.clamp(output, 0.0, 1.0)

# Proposed (physical range [50, 150] dB, normalized):
CLAMP_MIN_DB = 50.0
CLAMP_MAX_DB = 150.0
output = torch.clamp(output, CLAMP_MIN_DB / 180.0, CLAMP_MAX_DB / 180.0)
```

Or add two config keys to the YAML:
```yaml
target_metadata:
  path_loss:
    scale: 180.0
    clip_min_db: 50.0    # new
    clip_max_db: 150.0   # new
```

### Expected benefit

- **~0.1–0.3 dB improvement in overall RMSE** by eliminating tail mispredictions
- **~0.3–0.8 dB improvement in NLoS RMSE** (NLoS predictions sometimes
  extrapolate wildly beyond 150 dB when the model is uncertain)
- No cost: the clamp adds 1 line of code and zero training time
- Per-topology tuning is possible (e.g., `open_sparse_lowrise` never exceeds
  120 dB, so [60, 125] is valid for that expert)

### Residual-over-prior analysis

For the thesis, the strongest diagnostic is not the final target range alone,
but the range of the **learned residual**:

```text
residual_db = target_path_loss_db - formula_prior_db
```

If the residual distribution is narrow compared with the full 0–180 dB output
range, then the current head is wasting capacity on values it never needs.
This is especially relevant because Try 73 predicts a residual over a physical
prior, not path loss from scratch.

The minimum report that should be added is:
- residual histogram over all valid pixels
- residual percentiles (`p1`, `p50`, `p99`)
- residual percentiles split by LoS and NLoS
- residual percentiles split by expert

Why this matters:
- If LoS residuals are tightly concentrated around 0 dB, then LoS is already
  almost solved and the remaining gap is mostly structural NLoS error.
- If NLoS residuals show a long positive tail, that is direct evidence that the
  prior underestimates diffraction / shadowing loss.
- If the residual range is, for example, concentrated inside something like
  `[-20, +35] dB`, then both the loss design and the output parameterisation
  should be centred around that smaller range rather than around 0–180 dB.

---

## 3. Path to 5 dB RMSE — Roadmap

Current best: **9.41 dB** on `open_sparse_lowrise` at 513×513, epoch 96.
Target: **5 dB overall RMSE** (ambitious but physically achievable for LoS-dominant
experts per the dataset analysis in DATASET_ANALYSIS_AND_TRAINING_STRATEGY.md).

### Ceiling analysis

| Component | Current | Theoretical floor | Gap |
|---|---|---|---|
| LoS RMSE | 3.75 dB | ~2.0 dB (uint8 noise + model limit) | 1.75 dB |
| NLoS RMSE | 31.5 dB | ~8–12 dB (physics limit) | ~20 dB |
| Overall RMSE | 9.41 dB | ~3–4 dB (for lowrise) | ~5.4 dB |

LoS is 92.4% of pixels in `open_sparse_lowrise`. The overall RMSE is dominated by
the NLoS tail despite low NLoS fraction. Getting to 5 dB requires:
- LoS: 2.5 dB (from 3.75 dB), or
- NLoS: ~14 dB (from 31.5 dB, with same LoS)

### Why 5 dB is hard

The key limitation is **not** that the network cannot regress maps at all.
The real bottleneck is that a very small set of structurally difficult NLoS
pixels contributes a disproportionate share of the squared error:

- most LoS pixels are already in a relatively easy regime
- the prior is reasonably good in open areas
- the remaining error is concentrated near diffraction boundaries and deep shadow
  regions, where the prior is physically incomplete

This is why overall RMSE can look “close” to 5 dB while still being genuinely
hard to push down. In a LoS-dominant expert, the last few dB are not about broad
map quality; they are about eliminating the small number of high-error urban
shadowing failures that dominate the quadratic loss.

Supervisor-ready wording:

> The main limitation is not overall map regression capacity, but the absence of
> an explicit diffraction-aware prior in NLoS regions. The current model already
> performs competitively in LoS and on overall cross-city generalization; the
> remaining gap is concentrated in a small set of structurally hard urban
> shadowing cases.

### Priority improvements

#### Tier 1: Zero-cost / config-only

| Change | Expected gain | Status |
|---|---|---|
| **Output clamp [50, 150]** | 0.1–0.3 dB | Not yet implemented |
| **TTA at inference (D4, 8 transforms)** | 0.1–0.5 dB | Already configured |
| **Extend training** (patience=500, SGDR restarts) | 0.3–0.8 dB | Active (chain running) |
| **Lower EMA decay** (0.993 vs 0.995) | 0.1–0.2 dB | Active |

#### Tier 2: Training changes

| Change | Expected gain | Complexity |
|---|---|---|
| **Focal-Huber hybrid loss** (`w(e) = |e|^γ * Huber`) | 0.5–1.5 dB on NLoS | Medium |
| **Increase nlos_reweight_factor** to 8–10× | 0.3–0.8 dB on NLoS | Low (config) |
| **Curriculum: start Huber δ=8, decay to δ=3** | 0.3–0.6 dB | Medium |
| **Gradient clipping reduction** (1.0 → 0.5) for fine-tuning | 0.1–0.3 dB | Low |
| **Stochastic depth / DropPath** in encoder | 0.2–0.5 dB | High |

#### Tier 3: Model / data changes

| Change | Expected gain | Complexity |
|---|---|---|
| **Better NLoS prior** (knife-edge diffraction model) | 1–3 dB on NLoS | High |
| **Separate NLoS prediction head** | 1–2 dB on NLoS | High |
| **Expert ensemble at inference** (average 2-3 experts) | 0.3–0.7 dB | Medium |
| **Test-city fine-tuning** (10–50 samples from new city) | 1–3 dB | Medium |
| **Synthetic hard-NLoS augmentation** | 0.5–2 dB on NLoS | Very High |

### Realistic trajectory

Given current training (patience=500, cosine warm restarts, nlos_reweight=6×):

| Milestone | Est. epoch | Est. RMSE |
|---|---|---|
| Current best | 96 | 9.41 dB |
| After first SGDR restart cycle | ~120–140 | ~8.5–9.0 dB |
| After LR low point | ~200–240 | ~7.5–8.5 dB |
| With output clamp + TTA | inference | ~7.2–8.0 dB |
| With focal-Huber + nlos_reweight=8 | +100 epochs | ~6.5–7.5 dB |
| Theoretical floor for lowrise | — | ~3–4 dB |

**5 dB is achievable for lowrise but will require Tier 3 changes** (better NLoS
prior or separate NLoS head). The physics of single-edge diffraction at building
corners cannot be captured by a purely data-driven loss without explicit geometric
supervision.

### Canonical failure modes to discuss explicitly

These are the three most useful qualitative examples to include in the thesis:

| Failure mode | Why it is hard | What it suggests |
|---|---|---|
| Street canyon shadowing | Multiple tall facades create long NLoS corridors | Need diffraction-aware prior or dual-head decoding |
| Building-corner diffraction | Sharp transition from LoS to NLoS across edges | Need better boundary modelling and physics regularisation |
| Deep shadow behind dense blocks | Prior severely underestimates attenuation | Need residual analysis and higher-capacity NLoS branch |

These cases are more convincing than saying “NLoS is difficult” in general,
because they tie the model's error directly to known propagation mechanisms.

---

## 4. Per-Expert Clamping Reference

| Expert | LoS RMSE | NLoS RMSE | 99.9th percentile PL | Suggested clamp |
|---|---|---|---|---|
| open_sparse_lowrise | 3.75 dB | 31.5 dB | ~115 dB | [60, 120] |
| open_sparse_vertical | ~3.9 dB | ~41 dB | ~125 dB | [60, 128] |
| dense_block_highrise | ~4.1 dB | ~35 dB | ~135 dB | [55, 140] |
| mixed_compact_lowrise | TBD | TBD | ~120 dB | [58, 125] |
| mixed_compact_midrise | TBD | TBD | ~128 dB | [58, 132] |
| dense_block_midrise | TBD | TBD | ~132 dB | [55, 138] |

The dense/highrise experts have higher maximum values because: (a) more buildings
→ deeper NLoS → higher PL ceiling, and (b) lower Tx antenna below building tops →
longer shadow corridors. The clamp should be loosened for these.

---

## 5. Research Directions for Closing the 5.4 dB Gap

The gap between our 9.41 dB and the theoretical floor of ~3–4 dB is almost entirely
in the NLoS component (31.5 dB NLoS RMSE vs ~8–12 dB floor). Below are six concrete
research directions from recent literature, ranked by feasibility and expected impact.

### 5.1 Physics-Informed Loss — PDE Residual Regularisation

**Paper:** ReVeal (arXiv:2502.19646, Feb 2025) — "A Physics-Informed Neural Network
for High-Fidelity Radio Environment Mapping"

**Idea:** Derive a second-order PDE for the received signal field (based on wave
propagation models) and add the PDE residual as a **regularisation term** to the
standard loss. This forces the network to produce outputs that are not just accurate
on labelled pixels but also physically consistent across the entire map.

**Their result:** 1.95 dB RMSE on rural/suburban data with only 30 training points
over 514 km². The PDE acts as an infinite-data physics teacher — the model cannot
cheat by memorising samples.

**How to apply to Try 73:**
- Our model already uses a physics prior (`formula_prior`). The next step is to add
  a **Helmholtz-inspired PDE penalty** to the loss:
  ```
  L_physics = || ∇²ŷ + k²ŷ ||² (evaluated on predicted map)
  ```
  where k = 2π/λ. This penalises predictions that violate wave continuity — exactly
  the NLoS regions where the model currently produces noisy outputs.
- Does NOT require new data. Only a loss term. Medium complexity.

**Expected gain:** 0.5–2 dB on NLoS RMSE (most benefit in transition zones
between LoS and NLoS where the model currently shows sharp discontinuities).

**Link:** https://arxiv.org/abs/2502.19646

---

### 5.2 Diffraction-Aware Feature Extraction (Knife-Edge Prior)

**Paper:** "Diffraction and Scattering Aware Radio Map and Environment Reconstruction
using Geometry Model-Assisted Deep Learning" (arXiv:2403.00229, IEEE TWC 2024)

**Idea:** Use a **multi-screen knife-edge diffraction model** (ITU-R P.526) to
pre-compute diffraction features for every pixel based on building geometry. Feed
these as additional input channels to the neural network. The model learns
*residuals over the diffraction model* rather than predicting NLoS from scratch.

**Their result:** 10–18% accuracy improvement over SoA methods. 20% less training
data needed. 50% fewer epochs to converge when transferring to new environments.

**How to apply to Try 73:**
- We already compute building heights and topology maps. The knife-edge model
  only requires: (1) building edges along the Tx→pixel ray, (2) building heights
  relative to the Fresnel ellipsoid. This can be computed in the data pipeline.
- Add 1–2 input channels: `knife_edge_diffraction_loss` and `num_diffracting_edges`.
- The model's residual head then only needs to learn the gap between the KED
  prediction and reality, which is much smaller than learning NLoS from a
  free-space prior.

**Expected gain:** 1–3 dB on NLoS RMSE. This is the single highest-impact change
because it directly addresses the root cause: our current prior is free-space +
correction, which has no diffraction physics at all.

**Complexity:** High (requires a new data preprocessing step), but the KED
computation itself is O(N) per ray and well-understood.

**Link:** https://arxiv.org/abs/2403.00229

---

### 5.3 Diffusion-Based Refinement (RadioDiff)

**Papers:**
- RadioDiff (IEEE TCCN 2025): https://github.com/UNIC-Lab/RadioDiff
- RadioDiff-k² (IEEE JSAC 2026): Helmholtz PINN + diffusion
- RM-Gen (arXiv:2501.06604): Conditional DDPM for radio maps

**Idea:** Train a lightweight conditional diffusion model that takes our PMHHNet
output as the conditioning signal and iteratively refines it. Diffusion models
excel at generating spatially coherent, high-frequency detail — exactly what NLoS
shadow boundaries need.

**Architecture:**
```
PMHHNet prediction (8-channel) → Diffusion U-Net (5-10 steps) → Refined map
```

**Their results:** RadioDiff-k² achieves state-of-the-art radio map reconstruction
using physics-informed diffusion with Helmholtz PDE constraints.

**How to apply to Try 73:**
- Train PMHHNet as-is (stage 1). Freeze it.
- Train a small diffusion refinement head (stage 2) with the PMHHNet output as
  condition. Only 5–10 diffusion steps needed (not 1000 like image generation).
- The diffusion head specifically targets NLoS boundary sharpening and corridor
  filling, which is where most of the residual error lives.

**Expected gain:** 0.5–2 dB on overall RMSE (mainly NLoS improvement via better
boundary coherence).

**Complexity:** High (new training stage, diffusion infrastructure).

---

### 5.4 Test-Time Adaptation (RadioPiT / FM-RME)

**Papers:**
- RadioPiT (arXiv:2512.01451, Dec 2025): Pixel Transformer + TTA
- FM-RME (arXiv:2602.22231, Feb 2026): Foundation model + zero-shot generalisation

**Idea:** At inference time, adapt the model's batch normalisation / affine parameters
using self-supervised signals from the test sample itself. The key self-supervised
signal: **the physics prior should agree with the model output in LoS regions** (where
the prior is accurate). Any disagreement in LoS → the model's feature statistics
are miscalibrated for this city → adapt.

**RadioPiT result:** 21.9% RMSE reduction vs RadioUNet via TTA in real-world
scenarios with distribution mismatch.

**FM-RME result:** Zero-shot generalisation across diverse environments using
masked self-supervised pre-training + geometry-aware features.

**How to apply to Try 73:**
- At test time, for each unseen city batch:
  1. Forward pass → get prediction.
  2. Compute self-supervised loss: `L_tta = MSE(prediction[los_mask], prior[los_mask])`
  3. Backprop through BN affine parameters only (1–5 gradient steps).
  4. Forward pass again → refined prediction.
- Alternatively: use the physics prior as a pseudo-label for LoS pixels and adapt
  the FiLM layers (since those carry the height-specific statistics).

**Expected gain:** 0.3–1.0 dB on overall RMSE. Most benefit on unseen cities
that have different building statistics than training cities.

**Complexity:** Low–Medium (no retraining, just a few backward passes at test time).

---

### 5.5 NLoS-Specific Prediction Head (Dual-Head Architecture)

**Idea (not from a single paper, synthesised from multiple works):**

Currently PMHHNet produces a single residual map added to the physics prior.
The LoS and NLoS prediction tasks have fundamentally different statistical
properties:
- LoS: smooth, low-variance, well-predicted by free-space model
- NLoS: high-variance, spatially discontinuous, prior is nearly useless

**Proposal:** Split the decoder into two heads:
1. **LoS head**: predicts residual over prior (as now) — small corrections.
2. **NLoS head**: predicts absolute path loss directly from building geometry
   features — no prior anchor (because the prior is wrong in NLoS).

The `los_mask` (which we already have as input) selects which head's output
to use at each pixel. A soft blending in the transition zone (0.2–0.5 m around
building edges) avoids discontinuities.

**Expected gain:** 1–2 dB on NLoS RMSE. The NLoS head can allocate all its
capacity to the hard problem without being pulled towards the LoS distribution.

**Complexity:** Medium (architectural change, retraining from scratch or fine-tuning).

---

### 5.6 Foundation Model Pre-Training

**Paper:** FM-RME (arXiv:2602.22231, Feb 2026)

**Idea:** Pre-train a large encoder using **masked autoencoding** on all 16,180
samples (all 6 experts, not just one). The encoder learns general urban propagation
patterns. Then fine-tune the decoder for each expert. This is analogous to ImageNet
pre-training for vision tasks.

**How to apply to Try 73:**
- Stage 0 (self-supervised): Mask 50% of input channels, train the PMHHNet encoder
  to reconstruct the masked channels. No labels needed.
- Stage 1 (supervised): Attach expert-specific decoders and fine-tune with the
  standard loss on each topology partition.

**Expected gain:** 0.3–1.0 dB on overall RMSE, mainly from better feature
representations that transfer across topology types.

**Complexity:** Medium–High (new pre-training stage, may require > 50 epochs).

**Link:** https://arxiv.org/abs/2602.22231

---

### Summary: Priority Ranking for the 5.4 dB Gap

| Priority | Direction | Expected gain | Complexity | Ref |
|---|---|---|---|---|
| **1** | Knife-edge diffraction prior (§5.2) | 1–3 dB NLoS | High | arXiv:2403.00229 |
| **2** | Physics-informed PDE loss (§5.1) | 0.5–2 dB NLoS | Medium | arXiv:2502.19646 |
| **3** | Dual LoS/NLoS head (§5.5) | 1–2 dB NLoS | Medium | Synthesised |
| **4** | Test-time adaptation (§5.4) | 0.3–1.0 dB all | Low–Med | arXiv:2512.01451 |
| **5** | Diffusion refinement (§5.3) | 0.5–2 dB all | High | RadioDiff family |
| **6** | Foundation pre-training (§5.6) | 0.3–1.0 dB all | High | arXiv:2602.22231 |

**Most impactful single change:** Add knife-edge diffraction features as input
channels (§5.2). This directly addresses the #1 root cause of our NLoS error:
the physics prior has zero diffraction information. Every other improvement
works around this problem; §5.2 fixes it at the source.

**Quickest win for TFG timeline:** Test-time adaptation (§5.4) — requires no
retraining, can be prototyped in a few hours, and is publishable as a novel
contribution since no radio map paper applies TTA with height-conditioned FiLM.
