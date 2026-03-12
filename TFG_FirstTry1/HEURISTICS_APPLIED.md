# Heuristics Applied

This file documents the heuristics that are actually applied in the U-Net + GAN inference path of this repo.

Scope:
- cGAN + U-Net inference in [TFGpractice/TFG_FirstTry1/predict_cgan.py](TFGpractice/TFG_FirstTry1/predict_cgan.py)
- heuristic helpers in [TFGpractice/TFG_FirstTry1/heuristics_cgan.py](TFGpractice/TFG_FirstTry1/heuristics_cgan.py)
- config-driven post-processing in [TFGpractice/TFG_FirstTry1/configs/cgan_unet.yaml](TFGpractice/TFG_FirstTry1/configs/cgan_unet.yaml) and [TFGpractice/TFG_FirstTry1/configs/cgan_unet_hdf5.yaml](TFGpractice/TFG_FirstTry1/configs/cgan_unet_hdf5.yaml)

Important distinction:
- the plain U-Net path in [TFGpractice/TFG_FirstTry1/predict.py](TFGpractice/TFG_FirstTry1/predict.py) does not currently run the same explicit heuristic pass
- the explicit heuristic layer exists in the cGAN + U-Net inference path

## Goal

There are really two different notions of heuristics in this project, and they should not be mixed.

### 1. What is currently implemented

The current code applies conservative post-processing heuristics:
- suppress obviously invalid output values
- remove isolated image-to-image artifacts
- keep outputs numerically consistent with basic target ranges

These are lightweight sanity corrections, not physics-derived corrections.

### 2. What the project plan actually aims for

Based on the proposal and the notes in `docs/markdown`, the intended heuristic layer is stronger:
- use deterministic formulas from wireless propagation to correct or reinterpret raw model outputs
- enforce consistency between predicted maps and known physical relations
- derive secondary quantities from primary predicted quantities when the dataset does not store them directly

That is a different objective from simple clipping or median filtering. The current repo only implements the first category. The second category is the intended physics-aware direction, but it is mostly not implemented yet.

## Where they run

At inference time, [TFGpractice/TFG_FirstTry1/predict_cgan.py](TFGpractice/TFG_FirstTry1/predict_cgan.py) does this:

1. Loads the generator prediction.
2. Denormalizes each output with `target_metadata`.
3. Applies a heuristic pass if `postprocess.enable: true`.
4. Saves both raw and processed outputs.

The concrete heuristic functions are:
- `apply_regression_heuristics(...)`
- `apply_augmented_los_heuristics(...)`
- `apply_binary_mask_heuristics(...)`
- `derive_channel_power_from_path_loss(...)`
- `derive_snr_maps(...)`
- `derive_link_availability(...)`

Both live in [TFGpractice/TFG_FirstTry1/heuristics_cgan.py](TFGpractice/TFG_FirstTry1/heuristics_cgan.py).

## Heuristics currently applied

### 1. Physical-range clipping for regression targets

Applied by `apply_regression_heuristics(...)`.

Behavior:
- reads `clip_min` and `clip_max` from `target_metadata`
- clips denormalized predictions into that interval

In the original cGAN config [TFGpractice/TFG_FirstTry1/configs/cgan_unet.yaml](TFGpractice/TFG_FirstTry1/configs/cgan_unet.yaml), this means:
- `delay_spread` clipped to `[0, 1000] ns`
- `angular_spread` clipped to `[0, 180] deg`
- `channel_power` clipped to `[-140, 60] dB`
- `augmented_los` metadata also defines `[0, 1]`, but its dedicated soft-LoS path handles that target separately

In the HDF5 cGAN config [TFGpractice/TFG_FirstTry1/configs/cgan_unet_hdf5.yaml](TFGpractice/TFG_FirstTry1/configs/cgan_unet_hdf5.yaml), this means:
- `delay_spread` clipped to `[0, 1000] ns`
- `angular_spread` clipped to `[0, 180] deg`
- `path_loss` clipped to `[0, 180] dB`

Why:
- prevents physically invalid negative or out-of-range values after denormalization
- removes gross failures without changing the model itself

### 2. Median filtering on regression maps

Applied by `apply_regression_heuristics(...)` after clipping.

Behavior:
- applies a 2D median filter when `kernel_size > 1`
- current configs use `postprocess.regression_median_kernel: 3`

Applied to:
- original cGAN regression targets: `delay_spread`, `angular_spread`, `channel_power`
- HDF5 cGAN regression targets: `delay_spread`, `angular_spread`, `path_loss`

Why:
- removes isolated hot pixels and small speckle artifacts common in image-to-image predictions
- preserves edges better than a naive mean blur

### 3. Soft-field clamping for `augmented_los`

Applied by `apply_augmented_los_heuristics(...)` in the original cGAN route.

Behavior:
- clamps the denormalized soft map into `[0, 1]`

Formula:

$$
P_{clamped} = \mathrm{clip}(P, 0, 1)
$$

Why:
- `augmented_los` is treated here as a soft propagation / LoS confidence field
- values outside `[0, 1]` are not meaningful in that interpretation

### 4. Median filtering for `augmented_los`

Also applied by `apply_augmented_los_heuristics(...)` when enabled.

Behavior:
- uses `postprocess.augmented_los_median_kernel`
- default in [TFGpractice/TFG_FirstTry1/configs/cgan_unet.yaml](TFGpractice/TFG_FirstTry1/configs/cgan_unet.yaml) is `3`

Why:
- suppresses isolated blobs and unstable single-pixel confidence spikes in the soft LoS field

### 5. Optional consistency floor from binary LoS input

Supported in `apply_augmented_los_heuristics(...)`, but disabled by default in the shipped cGAN config.

Behavior when enabled:

$$
P_{out} = \max\left(P_{soft}, \text{binary\_los} \cdot \text{floor}\right)
$$

Config knobs in [TFGpractice/TFG_FirstTry1/configs/cgan_unet.yaml](TFGpractice/TFG_FirstTry1/configs/cgan_unet.yaml):
- `postprocess.enforce_binary_los_consistency`
- `postprocess.binary_los_consistency_floor`

Current default:
- `enforce_binary_los_consistency: false`
- `binary_los_consistency_floor: 0.5`

Why:
- if you trust the binary LoS input as a hard structural prior, this stops the refined soft LoS map from collapsing below a minimum confidence where binary LoS is present

### 6. Optional binary export for `augmented_los`

Supported in `apply_augmented_los_heuristics(...)`, also disabled by default.

Behavior when enabled:
- thresholds the postprocessed soft field using `postprocess.augmented_los_threshold`
- exports a hard binary map

Current default in [TFGpractice/TFG_FirstTry1/configs/cgan_unet.yaml](TFGpractice/TFG_FirstTry1/configs/cgan_unet.yaml):
- `export_augmented_los_binary: false`
- `augmented_los_threshold: 0.5`

Why:
- useful only if a downstream stage explicitly needs a hard mask
- not enabled by default because `augmented_los` is modeled as a soft field, not as a strict class mask

### 7. BCE outputs are converted to probabilities, not heuristically filtered

In the cGAN inference code, targets with `bce` loss are passed through a sigmoid and exported as probabilities.

That matters especially for the HDF5 cGAN route in [TFGpractice/TFG_FirstTry1/configs/cgan_unet_hdf5.yaml](TFGpractice/TFG_FirstTry1/configs/cgan_unet_hdf5.yaml):
- `los_mask` uses `bce`
- inference exports probability maps for `los_mask`
- no extra median or morphological cleanup is applied to that BCE target in the current code

Why:
- the current implementation treats BCE targets as probabilistic outputs and leaves any extra binarization or structural cleanup to later decisions

### 8. Path-loss-based link-budget derivation for the HDF5 route

This is now implemented as an explicit physics-aware derivation step in the cGAN HDF5 path.

Behavior:
- after `path_loss` is denormalized and cleaned, inference can derive a received-power map from a fixed link-budget assumption
- the implemented relation is

$$
P_r = P_t + G_t + G_r - L_{path} - L_{other}
$$

Current HDF5 cGAN configs enable this export with a default assumption of:
- `tx_power_dbm: 46.0`
- `tx_gain_dbi: 0.0`
- `rx_gain_dbi: 0.0`
- `other_losses_db: 0.0`

Saved outputs:
- `channel_power_derived_dbm.npy`
- `channel_power_derived_dbm.png`

Why:
- the dataset stores `path_loss` directly, not absolute `channel_power`
- this derivation gives a physically interpretable received-power map under explicit fixed assumptions
- it is closer to the project intent than pretending both tensors are identical

### 8b. Path-loss-prioritized training and post-processing

Because `path_loss` is the primary dB-domain quantity from which derived received power is reconstructed, the HDF5 cGAN configs now prioritize it explicitly.

Training change:
- `loss.target_loss_weights.path_loss: 3.0`
- other targets keep weight `1.0`
- `los_mask` is reduced to `0.5`

Inference change:
- the generic regression median filter remains available for all regression targets
- `path_loss` can now use its own kernel through `postprocess.path_loss_median_kernel`
- current HDF5 cGAN configs set `path_loss_median_kernel: 5`

Why:
- if `channel_power` is derived from `path_loss`, then improving `path_loss` is the most leverage-efficient way to improve the full dB-domain pipeline

### 9. Optional SNR derivation from derived received power

If `postprocess.link_budget.bandwidth_hz` is set, inference also derives SNR maps using a thermal-noise approximation:

$$
N_{dBm} = -174 + 10\log_{10}(B) + NF
$$

$$
SNR_{dB} = P_r - N_{dBm}
$$

Saved outputs when enabled:
- `noise_floor_derived_dbm.npy`
- `snr_derived_db.npy`
- `snr_derived_linear.npy`
- `snr_derived_db.png`

Why:
- the supervisor notes explicitly point to path loss being useful for averaged SNR calculations
- this keeps the neural net focused on predicting a primary propagation quantity while derived communication metrics are computed analytically

### 10. Optional reception-threshold availability map

If `postprocess.link_budget.reception_threshold_dbm` is set, inference exports a binary availability map defined by:

$$
\mathbb{1}(P_r \ge P_{th})
$$

Saved outputs when enabled:
- `link_available_binary.npy`
- `link_available_binary.png`

Why:
- it translates the derived received-power map into a simple coverage or service-feasibility map
- it is a formula-driven downstream interpretation rather than an arbitrary image cleanup rule

## Summary by route

### Original cGAN route

Config:
- [TFGpractice/TFG_FirstTry1/configs/cgan_unet.yaml](TFGpractice/TFG_FirstTry1/configs/cgan_unet.yaml)

Targets:
- `delay_spread`
- `angular_spread`
- `channel_power`
- `augmented_los`

Heuristics applied:
- regression clipping
- regression median filter
- `augmented_los` clamp to `[0, 1]`
- `augmented_los` median filter
- optional LoS consistency floor
- optional binary export

### HDF5 cGAN route

Config:
- [TFGpractice/TFG_FirstTry1/configs/cgan_unet_hdf5.yaml](TFGpractice/TFG_FirstTry1/configs/cgan_unet_hdf5.yaml)

Targets:
- `delay_spread`
- `angular_spread`
- `path_loss`
- `los_mask`

Heuristics applied:
- regression clipping for `delay_spread`, `angular_spread`, `path_loss`
- regression median filter for those same regression targets
- sigmoid probability export for `los_mask`
- optional binary export for `los_mask`
- derived `channel_power` export from `path_loss` under fixed link-budget assumptions
- optional SNR export when a bandwidth is configured
- optional link-availability export when a reception threshold is configured

Not applied in the current HDF5 cGAN route:
- no `augmented_los`-specific logic, because that target does not exist in `CKM_Dataset.h5`
- no explicit morphological cleanup for `los_mask`

## Formula-driven heuristics the project really points to

From the proposal and supervisor notes, the heuristic layer should mainly mean formula-aware corrections such as these:

### 1. Path loss to received-power reinterpretation

When transmit-side assumptions are fixed, received power can be reconstructed from path loss with a link-budget relation of the form

$$
P_r = P_t + G_t + G_r - L_{path} - L_{other}
$$

Implication for this repo:
- the current HDF5 route predicts `path_loss`
- if `P_t`, antenna gains, and extra losses are fixed externally, then a heuristic layer can derive a `channel_power`-like map from it
- this is much closer to the thesis intent than pretending `path_loss` and `channel_power` are literally the same tensor

### 2. SNR-oriented reinterpretation from path loss

Your supervisor explicitly noted that power/path loss is used for averaged SNR calculations.

That means a physics-aware heuristic layer could use:
- predicted `path_loss`
- fixed transmit-side assumptions
- optional receiver noise assumptions

to derive downstream SNR-related maps, rather than asking the neural net to predict every derived quantity directly.

### 3. Delay-spread corrections should respect bandwidth semantics

Your supervisor also noted that delay spread is calculated with effectively infinite bandwidth.

Implication:
- bandwidth should not be used as an ad hoc correction for `delay_spread` in the current HDF5 route
- any heuristic that tries to "fix" delay spread using a bandwidth parameter would be conceptually wrong for this dataset

### 4. LoS-related outputs should follow the actual dataset semantics

For the original manifest route, a soft `augmented_los` field can justify soft-valued correction rules.

For the HDF5 route:
- the target is `los_mask`
- it is a binary or probability-like LoS map, not a wave-aware soft field with richer semantics

So the formula-aware interpretation here should be stricter:
- thresholding or probability export is reasonable
- inventing a complex augmented-LoS correction model is not justified by the current HDF5 target itself

### 5. Centered-antenna geometry can justify future spatial consistency rules

Because the antenna is always at the map center, future heuristics could use that geometry.

But they should only be applied when tied to an actual propagation relation, not just because the center is known.

Good example:
- derive radial distance from the known center and use it inside a documented path-loss consistency check

Bad example:
- arbitrarily damp all predictions far from the center without a validated formula or empirical support

## What is not currently applied

These ideas are not implemented in the shipped U-Net + GAN path:
- morphology-based cleanup of `los_mask`
- connected-component filtering
- hole filling
- radial priors from antenna center
- building-height-aware suppression
- cross-target consistency rules such as forcing low delay spread where LoS is strong
- path-loss-to-channel-power conversion under fixed link-budget assumptions
- SNR-oriented derived-map export from predicted path loss
- formula-based consistency checks using explicit propagation equations

These may be useful later, but they need dataset-specific validation first.

## Practical conclusion

The current heuristic layer is intentionally conservative.

It does:
- clip impossible values
- smooth isolated local artifacts with median filtering
- optionally enforce limited LoS consistency when using the original `augmented_los` route

It does not:
- rewrite the prediction using heavy handcrafted rules
- impose strong physical priors that have not been verified against the dataset

That is the right tradeoff for the current U-Net + GAN path: enough cleanup to remove obvious garbage, without hiding model errors behind an overly rigid rule system.

## Dataset-informed revision

Given what is now known about `CKM_Dataset.h5`, the heuristic set is mostly acceptable, but it should be interpreted more narrowly for the HDF5 route.

Known facts from the dataset:
- targets are `delay_spread`, `angular_spread`, `path_loss`, and `los_mask`
- `los_mask` is a true binary target stored in `[0, 1]`, not a soft proxy such as `augmented_los`
- the antenna is always centered at `(0, 0)`
- all maps are already aligned to the same fixed `[-256, 256]` spatial frame
- there are no per-sample antenna, bandwidth, or frequency fields inside the HDF5

### What is still good

These heuristics remain justified for the HDF5 cGAN route:

1. Regression clipping is good.
It matches the observed target semantics and prevents impossible outputs for `delay_spread`, `angular_spread`, and `path_loss`.

2. A small `3x3` median filter on regression maps is reasonable.
It is a mild cleanup step and does not inject a strong physical prior.

3. Leaving `los_mask` as a probability map at inference is defensible.
Because it preserves uncertainty, it is safer than forcing a hard mask too early.

### What should be revised conceptually

These ideas should not be treated as equally appropriate for the HDF5 dataset:

1. `augmented_los`-style heuristics are not relevant to the HDF5 route.
The HDF5 dataset does not contain `augmented_los`; it contains a true binary `los_mask`. Any discussion of soft-wave LoS refinement belongs to the original manifest route, not to direct HDF5 training.

2. Binary-LoS consistency rules should not be copied into the HDF5 route.
In the original route, those rules made sense because `binary_los` could be used as an input prior while predicting a softer `augmented_los` field. In HDF5 mode, `los_mask` is itself the target and there is no separate trusted LoS input channel by default.

3. Radial priors from the centered antenna are tempting, but still not justified as post-processing.
Yes, the antenna is centered, but adding a hand-designed distance rule would risk hiding model mistakes behind an assumed propagation law that the dataset may not follow uniformly across cities.

### What would be the most defensible next revision

If the heuristics are revised further, the safest dataset-aware change would be this:

1. Keep the current regression clipping.
2. Keep the current `3x3` regression median filter.
3. Add only an optional export-time thresholded `los_mask_binary.npy` for HDF5 inference, while still saving `los_mask_probabilities.npy`.

That would be a modest improvement because:
- the target is genuinely binary
- many downstream uses may want a hard mask
- saving both soft and hard outputs avoids throwing away uncertainty

### What should still wait

These should still be held back until there is evidence from validation samples:
- connected-component cleanup of `los_mask`
- hole filling
- building-aware suppression
- cross-target constraints such as forcing low delay spread where predicted LoS is high
- explicit radial attenuation priors from the map center

## Final verdict

For the HDF5 dataset, the current heuristics are acceptable only as a conservative baseline.

My recommendation was:
- keep the regression heuristics as they are
- keep `los_mask` probabilistic by default
- if you revise anything next, add optional hard-mask export for `los_mask`
- do not add stronger geometry-driven or morphology-driven rules until you inspect prediction failures on real samples

But for the thesis intent, the more important next step is different:

1. Treat `path_loss` as the primary physically predicted dB-domain quantity.
2. Add derived outputs or evaluation helpers that reinterpret it through link-budget formulas.
3. Keep `delay_spread` free from fake bandwidth corrections.
4. Keep LoS heuristics aligned with whether the target is true binary `los_mask` or soft `augmented_los`.

This has now been implemented in the cGAN inference path for the HDF5-oriented configs:
- `predict_cgan.py` exports `los_mask_probabilities.npy` for `los_mask`
- the HDF5 cGAN configs enable `export_los_mask_binary: true`
- the binary export uses `postprocess.los_mask_threshold`, currently `0.5`

What this implementation does:
- clips the `los_mask` export to `[0, 1]`
- preserves the soft probability map
- optionally exports a thresholded binary map for downstream use

What it still does not do:
- morphology
- connected-component cleanup
- hole filling
- geometry-aware correction from antenna center
- link-budget-based reinterpretation of `path_loss`
- formula-driven correction of outputs based on documented propagation relations
