# LoS 2-ray Ring Metric — Design Notes

Context: the LoS-only expert in Try 74 reaches ~3.68 dB overall RMSE on LoS
pixels, but RMSE alone does not tell us whether the model learns the 2-ray
interference pattern (concentric rings from direct + ground-reflected rays) or
just the smooth envelope underneath.

On a reference sample (`Dhaka/sample_05022`, Abidjan/sample_00001) the target
`path_loss` has:

- LoS-region total std ≈ 4–5.5 dB
- LoS-region **high-pass** std (after removing a σ=3 px Gaussian smooth) ≈
  4.3–5.7 dB

i.e. almost all of the LoS path-loss variance lives in the ring oscillations,
not in the smooth envelope. So a "getting LoS to ~3.7 dB" model could in
principle be doing this either well or very poorly — we currently can't tell.

## Why a purely radial metric is wrong

Looking at the target image, the pattern is *not* radially symmetric:

1. The primary concentric-ring system (2-ray) dominates but is modulated
   azimuthally.
2. There are brighter axial streaks (roughly at 0° / 90° / 180° / 270°) that
   are not explained by a simple 2-ray interference against a flat ground.
   These likely come from:
   - antenna-pattern directivity of the ray tracer,
   - grid-aligned sampling artifacts at map resolution,
   - or ground clutter close to the Tx.
3. There are *secondary* ring systems offset from the centre — consistent with
   multi-ray reflection (image-source from nearby walls), which appears at
   different-than-Tx centres.
4. The rings also deform and get clipped where LoS breaks.

An azimuthal-mean radial profile collapses (2–4) into a single curve and makes
any model that fits the *mean* radial profile look perfect, regardless of
whether it reproduces the azimuthal structure. That's exactly the failure
mode we would *not* want to hide.

So the previously-sketched `ring_rmse_radial_profile_db` and
`ring_rmse_radial_deriv_db` metrics are poor choices on this data.

## What to measure instead

Keep everything in (x, y); do not pre-project to polar. Restrict all of the
below to `(los_mask > 0.5) & valid_mask`.

### Metrics (evaluation)

1. **`rmse_highpass_db`** — RMSE of `pred - gauss_σ(pred)` versus
   `target - gauss_σ(target)` on LoS pixels. σ ≈ 3 px removes the smooth
   envelope and keeps the ring + axial-streak structure. *Already wired up.*
   Direct answer to "does the model produce the right ring amplitude at the
   right places, regardless of radial / azimuthal symmetry?"

2. **`rmse_smooth_db`** — RMSE of `gauss_σ(pred)` versus `gauss_σ(target)`.
   Decouples ring error from smooth-envelope error. *Already wired up.*

3. **`ring_amp_ratio = std(pred_highpass) / std(target_highpass)`** on LoS
   pixels. Tells us whether the model is *under-* or *over-*modulating. A
   model that collapses to a smooth envelope will score ~0; a perfect model
   scores 1.0. *Already wired up.*

4. **`grad_mag_rmse_db`** — RMSE of `|∇PL|` computed with a Sobel (or central
   finite difference) filter, on LoS pixels. The ring structure and axial
   streaks both produce strong local gradient; this probes whether the
   pixel-to-pixel gradient magnitude is reproduced. *Not yet wired up.*

5. **`grad_cos_similarity`** — cosine similarity between `(∂pred/∂x, ∂pred/∂y)`
   and `(∂tgt/∂x, ∂tgt/∂y)` averaged over LoS pixels with sufficient gradient
   magnitude. Probes *direction* of oscillation, not just amplitude.
   *Not yet wired up.*

6. **`ssim_los`** — standard SSIM (e.g. 11×11 window) over the LoS region of
   the dB map. Captures local structural similarity without assuming radial
   symmetry.

7. **Banded MSE** — decompose via a small set of Gaussian-pyramid (or DoG)
   levels and report RMSE per band on LoS pixels. This generalises
   `rmse_highpass_db` to "high / mid / low" and lets us see whether the
   model loses ring-scale detail specifically at a given spatial frequency.

8. *(optional, heavier)* **Polar-patch 2D FFT magnitude RMSE** — resample a
   thin annulus around the Tx to (r, θ), take a 2D FFT per annulus, and
   compare magnitudes. Phase-insensitive (so a small Tx-centre offset does
   not kill the metric) but spatially selective.

Recommended minimum set for the evaluation script:

- `rmse_highpass_db`, `rmse_smooth_db`, `ring_amp_ratio` (already there)
- `grad_mag_rmse_db`, `grad_cos_similarity`, `ssim_los` (add next)

Drop `rmse_radial_profile_db` and `rmse_radial_deriv_db`: they assume radial
symmetry which is wrong for this data.

### Losses (training, future Try 75)

Any loss added here should multiply a small weight (≈ 0.05–0.2 of the main
`full_map_rmse_only` term) and be applied only to LoS pixels.

- **High-pass L2** on the residual `(pred - gauss_σ(pred)) − (tgt - gauss_σ(tgt))`.
  Fully differentiable, trivially cheap, does not assume radial symmetry.
  This is probably the right default "ring-fidelity" loss.

- **Gradient-magnitude L2** with a Sobel kernel. Complements the high-pass
  term by forcing the pixel-level oscillation slopes to match rather than
  just their amplitudes.

- **Multi-scale Laplacian pyramid L2**. Same high-pass idea at several scales;
  useful if the high-pass at a single σ under-penalises coarser oscillations
  at large radii.

Not recommended as losses:

- **Radial-profile L2** — model can cheat by flattening rings; also assumes
  symmetry the data does not have.
- **Polar-FFT magnitude L2** — phase-insensitive, can match magnitude with
  wrong spatial placement. Useful as a diagnostic metric, not as a loss.

### Open questions before acting

- **Where is the antenna centre exactly?** If the Tx is not at pixel (256, 256)
  in all samples, any radial metric breaks hard. The current
  `_compute_distance_map_2d` assumes dead-centre; worth checking on 5–10
  random samples.
- **Which axial streaks are physics vs artifact?** If the 0°/90° streaks are a
  ray-tracer grid artifact, a model that *does not* learn them is arguably
  correct. Worth confirming with one of the source papers before penalising
  the model for smoothing them out.
- **σ of the high-pass**. σ = 3 px captures ring scales ≈ 2–10 px (a few m at
  1 m/px). If the ring period grows with distance from Tx (it does), a single
  σ is a compromise. A Laplacian-pyramid or multi-σ version fixes this.

## Status in the eval script

Currently implemented (in `eval_per_sample.py`, `compute_ring_metrics_db`):

- `rmse_highpass_db`, `rmse_smooth_db`
- `target_highpass_std_db`, `pred_highpass_std_db`, `ring_amp_ratio`
- `rmse_radial_profile_db`, `rmse_radial_deriv_db` (**keep in code as a
  numerical check but do not report as primary metrics** — the pattern is not
  radially symmetric)

Pending before running the full 273-sample eval:

- add `grad_mag_rmse_db` and `grad_cos_similarity`
- add `ssim_los`
- drop the two radial entries from the summary printout / primary rankings
- precompute obstruction features to a local HDF5 so the eval is not dominated
  by on-the-fly 720-ray ray-casting (~10 s / sample today)
