# Try 76 — Distribution-first, outlier-aware map synthesis

Built **from scratch** (no weight, loss or architecture inheritance from Tries 67–75; only
SLURM submission infrastructure is re-used). The 12-expert split contract of the
`try76_expert_registry.yaml` scaffold is preserved.

Companion docs:

- `docs/histogram_study.md` — auto-generated summary from
  `tmp_review/histograms_try74/histograms.csv` (run `scripts/study_histograms.py`).
  Contains per-topology-class tables for all 6 city types, including the
  `open_sparse_lowrise` and `open_sparse_vertical` classes the 12-expert scaffold
  covers at the tail of `try76_expert_registry.yaml`.
- `docs/distribution_classes.md` — auto-generated family-fit identification
  (run `scripts/classify_distributions.py`). Ranks `gaussian / laplace / skew_normal /
  lognormal / gamma / weibull / spike_plus_exp / gmm2 / gmm3 / gmm4 / gmm5` by
  KL divergence against the empirical histograms. **Not one** target histogram
  is best fit by a single Gaussian; across `path_loss|target_*` groups `gmm5`
  ties or narrowly beats `gmm3`/`gmm4` (usually within 1e-3 KL), so Try 76
  defaults to **K=5** mixture components. For LoS path-loss, which is already
  near-Gaussian, the extra components collapse (duplicate μ, small σ) and do
  no harm.
  - `path_loss|target_los` → `gmm3` (KL 0.003) ≻ `gmm2` (0.004) ≻ `skew_normal` (0.011).
  - `path_loss|target_nlos` → `gmm3` (0.039) ≻ `gmm2` (0.041) ≻ **`laplace`** (0.054).
    Laplace beating Gaussian into third place is the smoking gun that the NLoS
    bulk has heavier-than-Gaussian tails — exactly why a plain MSE head
    collapses.
  - `delay_spread|target` and `angular_spread|target` are dominated by a
    `spike_plus_exp` mixture (delta near 0 + exponential tail). They are
    extremely non-Gaussian (skew +5…+8, excess kurtosis +40…+80).
- `docs/histogram_study.json` / `docs/distribution_classes.json` — machine-readable versions for configs.
- `docs/distribution_classes_ranking.csv` — flat one-row-per-(scope, class, kind, family) ranking with KL and TVD, for downstream plotting.
- `docs/histograms_raw_by_class.csv` — long-form (city_type_6, metric, kind, bin_lo, bin_hi, count) dump of the aggregated target/prediction histograms across all 8 kinds, including delay and angular spreads. Try 77 (same structure as Try 76 but for delay/angular) will use this as its evidence.
- `docs/height_stratified.md` / `.json` / `.csv` — per-expert drift of the
  target distribution across four UAV-height buckets
  (`lowA` 0–40 m, `lowB` 40–80 m, `mid` 80–160 m, `high` 160–478 m). Key finding:
  **LoS path-loss mean drifts +5–11 dB from low to high UAV altitude** per
  topology class; NLoS mean is much flatter (±1 dB) but its GMM tail component
  still shifts. A single per-expert GMM without height conditioning would be
  biased on samples far from the expert's mean altitude — Try 76 avoids this
  by **making Stage-A's GMM parameters height-conditional** (see §3.2.1).

## 1. Problem recap & evidence

### 1.1 What is broken in Tries 67–75

From the aggregated target/prediction histograms on the Try 74/75 val+test splits
(`tmp_review/histograms_try74/histograms.csv`, 22 001 rows × 721 dB bins):

| kind (aggregated) | target mean / std | pred mean / std | note |
|---|---|---|---|
| `path_loss / target_los` | 97.0 / 5.3 | 96.4 / 3.8 | LoS is fine, slightly too narrow |
| `path_loss / target_nlos` | **107.1 / 4.3** | **36.5 / 22.4** | catastrophic mode collapse |
| `delay_spread / target` | 10.4 / 27.7 (skew +5.5) | 8.4 / 9.5 | tail is lost; only metric passes because GT is itself near-degenerate |
| `angular_spread / target` | 6.6 / 15.7 (skew +2.9) | 6.6 / 9.8 | same story |

Concretely, **predicting a uniform 110 dB on NLoS pixels would beat the current
Try 75 NLoS predictions.** The network is not learning the marginal target
histogram, let alone spatial structure.

### 1.2 Structural targets (from the study)

- **path_loss / LoS** — unimodal, Gaussian-ish, centred near 97 dB, std ≈ 5.
- **path_loss / NLoS** — near-Gaussian bulk at 106–108 dB with a faint LoS-leak
  tail. Bimodal GMM2 fit is tight: `(π=0.50, μ=106.5, σ=5.5)` + `(π=0.50, μ=107.7, σ=2.4)`.
- **delay_spread / target** — spike near 3 ns + heavy 14 % tail centred at 57 ns.
  GMM2 captures this well (`π≈0.86, μ=3, σ=1.2` + `π≈0.14, μ=57, σ=55`).
- **angular_spread / target** — spike at 0.5° + 16 % tail at 38° (`σ≈18°`).

The per-`city_type_6` tables show the same two-bump structure, but the bump
weights shift — denser / taller cities have **more tail mass** (more outliers).
This means the distribution has to be conditioned on topology / LoS context,
not fixed per-expert.

## 2. Design goals

1. **Get the marginal right, then worry about spatial layout.** The network
   should first commit to a full pixel-value histogram (as a mixture with a
   small number of components) before it paints the map.
2. **Explicitly model outliers.** Every expert should have an outlier branch
   whose mass and spread are regressed from the context — not a fixed clip.
3. **Work only on valid ground pixels** (`topology == 0`) for both loss and
   metric, matching the last ~50 tries and the VERSIONS.md masking rule.
4. **Inputs are minimal and physical** — no knife-edge / COST-231 prior, no
   formula-confidence channel, no Laplacian HF stem. We want to verify the
   distribution-first idea without confounding physics priors.

## 3. Architecture

A single Try 76 network per expert (12 experts, one checkpoint each):

```
         ┌──────────────────────────────┐
         │   Stage-A:  DIST HEAD        │
 Inputs  │   context encoder -> GAP     │
 ───────▶│   -> 3-component GMM head    │
         │   (π_k, μ_k, σ_k) for k=1..3 │
         └───────────┬──────────────────┘
                     │  (global shape parameters)
                     ▼  FiLM/cat into decoder
         ┌──────────────────────────────┐
         │   Stage-B:  MAP HEAD         │
         │   U-Net decoder ->           │
         │   per-pixel (p_k, z, σ̃)      │
         │   -> mixture-aware sampling  │
         └───────────┬──────────────────┘
                     ▼
                  ŷ(i,j) on ground pixels
```

### 3.1 Inputs (4 channels + 1 scalar)

| channel | dtype | normalization |
|---|---|---|
| `topology_map / 90.0` | float32 | building height in meters / 90 m (approx. P99) |
| `los_mask` | float32 | binary 0/1 (from HDF5 `los_mask`) |
| `nlos_mask` = `1 - los_mask` on ground | float32 | complementary binary |
| `ground_mask` = `topology == 0` | float32 | mandatory, no target outside |
| scalar `antenna_height_m` | float32 | **sinusoidal FiLM embedding**, 16 frequencies, log-spaced over [12, 478] m |

Masking convention: the **ground_mask** zeroes `topology`, `los`, `nlos` on
non-ground pixels during input construction, so the encoder never sees leaked
building values. Loss & metric are evaluated only where
`ground_mask & (target != no_data_sentinel)`.

**No** knife-edge, **no** COST-231 prior, **no** elevation map, **no** Tx-depth
map, **no** Laplacian HF — the whole point is to test whether the marginal
distribution can carry the heavy lifting by itself.

### 3.2 Stage-A — distribution head

**Why K=3 mixture.** `docs/distribution_classes.md` ranks `gmm3` as the best-fit
family for every `path_loss | target_los` and `path_loss | target_nlos`
aggregated histogram, with Laplace placing third for NLoS — evidence of
heavier-than-Gaussian tails. A single Gaussian is never in the top-3. The third
component is what lets Try 76 absorb the LoS-leak tail that Tries 67–75 smeared
into a 20 dB-wide modal-collapsed blob.

Shallow conv encoder (7 conv blocks, GroupNorm, width 48→192) with FiLM-conditioned
sinusoidal height embedding at two depths, followed by GAP. The GAP vector is
projected to **K=5-component GMM parameters on normalized path-loss** (K is a
config knob per expert, default 5):

- `π ∈ Δ²` via softmax (3 logits)
- `μ_k ∈ [0, 1]` via sigmoid, then rescaled to `[clamp_lo, clamp_hi]`
- `σ_k > 0` via softplus + 1 dB floor

#### 3.2.1 Height conditioning of Stage-A

`docs/height_stratified.md` shows that, per topology class, the LoS path-loss
mean drifts by **+5 to +11 dB** between the `lowA` (h<40 m) and `high` (h>160 m)
UAV-height buckets, while variance shrinks as h rises (less scatter in
free-space regime). NLoS mean drifts less (±1 dB) but the upper-tail GMM
component still moves ~2–5 dB. Key consequence: if Stage-A emitted a single
fixed (π, μ, σ) per expert, samples in the `high` bucket would sit systematically
biased vs a expert-pooled GMM.

So Stage-A does **not** produce one GMM per expert. The conv encoder feeding
the GMM MLP is FiLM-modulated at two depths by the *sinusoidal height embedding*
(16 log-spaced frequencies over [12, 478] m). The GAP vector that reaches
the MLP is therefore height-conditional by construction, and the emitted
`(π_k(h), μ_k(h), σ_k(h))` can track the drift reported in
`docs/height_stratified.md`. We additionally feed the concatenated GMM vector
back into Stage-B via a second FiLM path so the per-pixel `p_k(i,j)` respects
the per-sample shift.

#### 3.2.2 Clamps

`(clamp_lo, clamp_hi)` is per-expert and pulled from
`docs/histogram_study.json` (`nonzero_lo`/`nonzero_hi` plus a small margin).
Examples:

| expert | clamp_lo | clamp_hi |
|---|---:|---:|
| `open_sparse_lowrise_los` | 60 | 125 |
| `open_sparse_lowrise_nlos` | 90 | 140 |
| `dense_block_highrise_los` | 60 | 125 |
| `dense_block_highrise_nlos` | 90 | 140 |

(LoS experts always have a tighter lower bound; NLoS experts always sit ≥ 90 dB.
Exact clamps are materialised by `scripts/build_expert_clamps.py` from the JSON
summary.)

### 3.3 Stage-B — outlier-aware map head

A small U-Net (4 scales, width 64, GroupNorm) whose decoder receives the Stage-A
context vector **and** the predicted GMM parameters via FiLM. For each ground
pixel the decoder outputs:

- `p(i,j) ∈ Δ²` — soft assignment over the K=3 components.
- `z(i,j) ∈ ℝ` — a unit-Gaussian sample of where within that component the
  pixel should fall.
- `log σ̃(i,j)` — per-pixel residual variance used by the outlier branch.

Reconstruction (deterministic at inference):

```
ŷ(i,j) = Σ_k p_k(i,j) · ( μ_k + z(i,j) · σ_k )
```

At inference we can optionally draw K diverse samples using `z ~ 𝒩(0, 1)`;
the default is `z = 0` (conditional mean, lowest RMSE).

### 3.4 Why a third “outlier” mixture component

The study shows that the LoS-leak inside NLoS and the heavy right tail of
delay/angular spreads are concentrated in **<20 %** of pixels but dominate the
error budget. A third mixture component with a **deliberately high σ floor
(15 dB)** acts as an outlier reservoir: the soft assignment `p_3(i,j)` can be
arbitrarily small (≪ 10⁻³) on the easy bulk and light up on the hard pixels.

## 4. Losses

All losses are masked by `ground_mask`.

### 4.1 Distribution-matching NLL (Stage-A)

On the *empirical* per-image target histogram `q̂` (soft-binned to 64 bins on
the normalized dB range), the Stage-A head minimises the KL:

```
L_dist = KL( q̂ || p_A )           where p_A = Σ_k π_k 𝒩(μ_k, σ_k)
```

Soft-binning uses a Gaussian kernel with σ = 0.5 bin width to keep the KL
differentiable in `(π, μ, σ)`.

### 4.2 Mixture NLL for the map head (Stage-B)

Per ground pixel, under the predicted mixture:

```
L_map = - log Σ_k p_k(i,j) · 𝒩( y(i,j) ; μ_k, √(σ_k² + σ̃(i,j)²) )
```

The σ̃ term is what buys us heteroscedastic outlier tolerance; pixels with
high predicted σ̃ are down-weighted automatically (Kendall & Gal–style), but
unlike Try 71 the base component width `σ_k` is supplied by the Stage-A head
rather than emitted per pixel from scratch.

### 4.3 Moment-matching regulariser

```
L_mom = (E[ŷ] − E[y])² + (Var[ŷ] − Var[y])²
```

computed over valid pixels of each image, scaled so both terms are ~O(1).
This keeps the per-image mean and spread aligned even when individual pixels
are mis-placed.

### 4.4 Outlier budget

To stop the model from using the heavy component everywhere:

```
L_outlier = relu( mean p_3(i,j) − 0.25 )
```

with coefficient 0.1. 25 % is the loosest bound on the tail fraction seen in
`dense_block_highrise` targets (see study).

### 4.5 Masked RMSE on the reconstructed map

The **global thesis objective** is **RMSE < 5 dB** on the combined (LoS + NLoS)
path-loss map. Even though `L_map` already contains this information in
log-probability form, a direct `sqrt(mean_mask((ŷ-y)²))` term in dB units makes
the optimiser commit budget to the very quantity the thesis is judged on:

```
L_rmse = sqrt( masked_mean( (ŷ − y)² ) )   # dB
```

Weight `0.5`. Combined with the NLL it behaves like a *physical-units* guide
rail that keeps Stage-B from paying for a tighter distribution match by
accepting a worse map.

### 4.6 Total

```
L = L_map
  + 0.5 * L_dist
  + 0.1 * L_mom
  + 0.1 * L_outlier
  + 0.5 * L_rmse
```

No Huber, no MSE (the RMSE term is rooted — not squared — so it uses dB units,
not dB²), no corridor-weighted loss, no NLoS-focus loss, no Laplacian
PDE term, no multi-scale aux heads. We intentionally keep the objective minimal
so the distribution-first hypothesis is not muddied.

Early-stop score (val):
```
score = val.rmse_db + 0.25 * val.dist_kl + 0.1 * val.map_nll
```
RMSE in dB dominates, distribution terms act as tie-breakers.

## 5. Training protocol

- **Split:** `data.split_mode = city_holdout`, `val_ratio=0.15`, `test_ratio=0.15`,
  `split_seed=42`. This is the **same contract as Try 75**
  (`TFGpractice/TFGSeventyFifthTry75/data_utils.py::_split_hdf5_samples`).
  Try 76 re-implements the splitter from scratch (no imports from Try 75) and
  ships `tests/test_split_matches_try75.py` that asserts byte-for-byte
  equivalence on the CKM HDF5.
- **Routing:** 12 experts = 6 topology classes × {LoS, NLoS}. Each expert only
  sees:
  - samples where `city_type_6(topo) == expert.topology_class`
  - pixels where `expert.region_mode` says so (`los_only` or `nlos_only`, gated
    by `los_mask`).
- **Optimizer:** AdamW, lr `3e-4`, wd `0.01`, cosine with 1 warmup epoch.
- **Batch:** 1 image per GPU, grad-accum 8 (same as Try 68/75 post-patch).
- **Regularisation:** dropout 0.1 inside the Stage-B bottleneck only; no
  CutMix; no label noise; no EMA on stage-A (the GMM parameters are already
  global / low-variance).
- **Epochs:** 120 per expert on 4×RTX2080, ~2 h each. Early stop on val
  `L_map + L_dist` (patience 15).

## 6. Evaluation

Reported metrics (all masked by ground + valid target):

- `RMSE_dB` overall, LoS-only, NLoS-only (the key number).
- `Wasserstein-1` between per-image target and predicted histograms.
- `mean |Δmean|` and `mean |Δstd|` vs target.
- `KL( q̂_target || p_A )` — how close is the Stage-A prediction to the *true*
  per-image marginal? If this is large but `RMSE_dB` is small, the map head is
  overriding the distribution and we should shrink `L_dist`. If this is small
  but `RMSE_dB` is large, Stage-B is the bottleneck.

The `evaluate.py` script emits per-sample JSON + an aggregate markdown table,
plus a copy of the per-sample histograms in the same wide CSV schema as
`tmp_review/histograms_try74/histograms.csv` for apples-to-apples comparison.

## 7. Repository layout

```
TFGSeventySixthTry76/
├── DESIGN_TRY76.md                 # this file
├── README.md                        # scaffold note (pre-existing)
├── docs/
│   ├── histogram_study.md           # auto-generated
│   └── histogram_study.json
├── src/
│   ├── __init__.py
│   ├── model_try76.py               # Stage-A + Stage-B, GMM heads
│   ├── losses_try76.py              # all 4 loss components
│   ├── data_utils.py                # split + 12-expert routing + ground mask
│   ├── config_try76.py              # YAML loader with schema + clamp lookup
│   └── metrics_try76.py             # RMSE, Wasserstein, KL, histograms
├── scripts/
│   ├── study_histograms.py          # the analysis that shaped this design
│   ├── study_height_stratified.py   # height-bucket drift per expert
│   ├── classify_distributions.py    # family-fit identification + CSV dumps
│   ├── build_expert_clamps.py       # JSON -> per-expert clamp table
│   └── plot_history.py              # renders history.json -> curves PNG
├── train_try76.py
├── evaluate_try76.py
├── cluster/
│   ├── run_seventysixth_try76_1gpu.slurm
│   ├── run_seventysixth_try76_2gpu.slurm
│   ├── run_seventysixth_try76_4gpu.slurm
│   ├── prepare_runtime_config.py    # copy of Try 75's (SLURM-only reuse)
│   └── submit_try76_experts_4gpu_sequential.py
├── experiments/seventysixth_try76_experts/    # (pre-existing 12 YAMLs + registry)
└── tests/
    └── test_split_matches_try75.py
```

## 8. Non-goals (for this try)

- Multi-scale auxiliary heads (Try 70 territory).
- Heteroscedastic-only output (Try 71 territory) — we use σ̃ as an adjunct, not
  the primary spread.
- Knife-edge / COST-231 / two-ray priors.
- Domain adaptation on the target city.

If the distribution-first model still under-fits NLoS after Try 76 converges,
Try 77 can re-introduce knife-edge as a Stage-B channel and Try 78 can swap
the U-Net for a RadioTransformer trunk.
