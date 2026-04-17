# NLoS Shadow-Structure Losses - Design Notes

Context: In the reviewed samples, NLoS error is dominated by shadow-structure mismatch:

- sharp transitions around LoS->NLoS breaks (knife-edge regions),
- sector/wedge attenuation patterns behind blockers,
- broad low-frequency attenuation envelopes over deep NLoS zones.

This differs from LoS ring fidelity. For NLoS we should emphasize boundary shape + mid/coarse spatial structure.

## Proposed NLoS training losses (all masked to NLoS + valid pixels)

Let:

- `nlos_mask = (los_mask <= 0.5) & valid_mask`
- `pred_db, tgt_db` be denormalized path-loss maps in dB.

### 1) `nlos_dog_l2_loss` (Difference-of-Gaussians L2)

Purpose: match shadow-sector spatial frequencies (mid-band), not only pointwise RMSE.

Definition:

- `DoG(x) = gauss_sigma_small(x) - gauss_sigma_large(x)`
- Minimize `MSE(DoG(pred_db), DoG(tgt_db))` on `nlos_mask`.

Defaults:

- `sigma_small_px = 2.0`
- `sigma_large_px = 6.0`
- `loss_weight = 0.08`

### 2) `nlos_gradmag_l2_loss` (Sobel gradient-magnitude L2)

Purpose: align knife-edge/shadow boundaries and local slope changes in NLoS.

Definition:

- `|grad(x)| = sqrt((Sobel_x(x))^2 + (Sobel_y(x))^2 + eps)`
- Minimize `MSE(|grad(pred_db)|, |grad(tgt_db)|)` on `nlos_mask`.

Boundary emphasis (optional):

- derive a soft boundary map from the LoS mask gradient,
- weight errors higher near LoS->NLoS boundaries.

Defaults:

- `loss_weight = 0.06`
- `boundary_boost = 2.0`
- `eps = 1e-6`

### 3) `nlos_laplacian_pyramid_l2_loss`

Purpose: preserve coarse and intermediate NLoS attenuation structure across scales.

Definition:

- Build Laplacian pyramid residuals across levels,
- minimize weighted L2 per level on pooled NLoS mask.

Defaults:

- `levels = 3`
- `min_valid_ratio = 0.25`
- `coarse_emphasis = true` (higher levels weighted more)
- `loss_weight = 0.08`

## Integration policy

- Keep existing `full_map_rmse_only` main objective unchanged.
- Add these as additive regularization terms with small weights.
- Apply only when LoS channel exists in inputs.
- Normalize by `(path_loss_scale_db)^2` for stable weighting.

## Recommended first-pass weights

- `nlos_dog_l2_loss.loss_weight = 0.08`
- `nlos_gradmag_l2_loss.loss_weight = 0.06`
- `nlos_laplacian_pyramid_l2_loss.loss_weight = 0.08`

Total additive pressure ~0.22 relative to normalized base terms; tune after first 10-20 epochs.

## Expected impact

- Lower error near knife-edge and moderate-depth break zones.
- Better preservation of sector-shaped deep-NLoS attenuation patterns.
- Reduced over-smoothing in NLoS while keeping RMSE objective intact.
