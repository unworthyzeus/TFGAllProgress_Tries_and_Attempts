# cGAN + U-Net heuristics applied

This file lists the **practical post-processing heuristics currently applied to the cGAN + U-Net predictor** in order to remove gross errors while keeping the model output mostly data-driven.

Important semantic note: **`augmented_los` is treated as a soft wave-aware LoS field, not as a truly binary map**. The binary LoS input is only a structural cue.

Related implementation files:
- [predict_cgan.py](predict_cgan.py)
- [heuristics_cgan.py](heuristics_cgan.py)
- [configs/cgan_unet.yaml](configs/cgan_unet.yaml)

## Goal
The goal of these heuristics is **not** to replace the model, but to suppress obviously invalid outputs such as:
- negative delay spread
- angular spread above physically valid range
- unrealistic channel-power spikes
- augmented LoS maps that contradict the binary LoS input
- isolated local artifacts in regression maps

## Currently applied heuristics

### 1. Physical range clipping for regression targets
Applied to:
- `delay_spread`
- `angular_spread`
- `channel_power`

Configured in [configs/cgan_unet.yaml](configs/cgan_unet.yaml) via `target_metadata`:
- `delay_spread`: clipped to `[0, 1000] ns`
- `angular_spread`: clipped to `[0, 180] deg`
- `channel_power`: clipped to `[-140, 60] dB`

### Why
This removes impossible or grossly implausible predictions directly after denormalization.

---

### 2. Median filtering on regression maps
Applied to:
- `delay_spread`
- `angular_spread`
- `channel_power`

Configured in [configs/cgan_unet.yaml](configs/cgan_unet.yaml):
- `postprocess.regression_median_kernel: 3`

### Why
A small `3x3` median filter removes isolated hot pixels and tiny local spikes that often appear in image-to-image models.

---

### 3. Augmented LoS soft-field clamping
Applied to:
- `augmented_los`

The `augmented_los` output is treated as a soft field and clipped into `[0, 1]`.

### Why
This guarantees numerically valid soft LoS / wave-propagation scores.

---

### 4. Binary LoS consistency enforcement
Applied to:
- `augmented_los`

If `binary_los` is provided as input, then the post-processing sets:

$$
P(\text{augmented\_los}) = \max(P(\text{augmented\_los}), \text{binary\_los})
$$

Configured in [configs/cgan_unet.yaml](configs/cgan_unet.yaml):
- `postprocess.enforce_binary_los_consistency: true`

### Why
If the input map already says LoS is present, the refined/augmented LoS output should not contradict that with a lower confidence.

---

### 5. Optional thresholding augmented LoS into a binary map
Applied to:
- `augmented_los`

Configured in [configs/cgan_unet.yaml](configs/cgan_unet.yaml):
- `postprocess.export_augmented_los_binary: false` by default
- `postprocess.augmented_los_threshold: 0.5`

The predictor now exports:
- `augmented_los_soft.npy`
- optionally `augmented_los_binary.npy`

### Why
This is optional because the main target is not truly binary. It is only useful if a later stage explicitly needs a hard mask.

## What is *not* yet applied
These were considered but are **not yet implemented** because they require dataset-specific validation:

### A. Height-aware occlusion suppression
Example idea:
- suppress improbable high-power / LoS islands behind very tall structures

### B. Distance-aware smoothing from antenna center
Example idea:
- use radial priors because the antenna is always centered

### C. Joint cross-target consistency rules
Examples:
- high delay spread should correlate with low direct-path dominance
- strong LoS should often imply smaller delay spread than severe NLoS zones

### D. Morphological cleanup for augmented LoS
Examples:
- removing tiny isolated connected components
- hole filling in large LoS regions

These are promising, but they should only be added once the dataset encoding is known.

## Recommendation
The current heuristic set is a good **safe first layer** because it:
- enforces basic physical plausibility
- removes isolated artifacts
- respects the binary LoS input
- does not hardcode too much domain structure yet

## Output files produced by cGAN inference
When using [predict_cgan.py](predict_cgan.py), the following are saved:
- `predictions_raw.npy`
- `<target>.png`
- `<target>_physical.npy` for regression targets
- `<target>_probabilities.npy` for BCE targets
- `augmented_los_binary.npy` for thresholded LoS output
