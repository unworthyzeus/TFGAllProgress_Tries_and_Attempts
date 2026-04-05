# Try 50 Prior Improvements

## Current Direction
- The main goal of Try 50 is to improve the NLoS prior formula itself.
- The active Try 50 prior now uses `hybrid_damped_coherent_two_ray_deepshadow_vinogradov_nlos`.
- The prior calibration JSON is the deepshadow train-only fit: `prior_calibration/regime_obstruction_train_only_deepshadow_directml_quick.json`.
- The cache version was bumped to keep the deepshadow run separate from the earlier cost231-based attempts.

## Why This Branch
- The plain `hybrid_two_ray_cost231_a2g_nlos` blend was still too generic for NLoS.
- The deepshadow branch keeps the LoS path separate but makes the NLoS side more conservative and obstruction-aware.
- That matches the goal for Try 50: improve the prior specifically where the NLoS tail is failing.

## Paper-Backed NLoS Update
- Keep the Try49 LoS backbone: deterministic geometry, LoS-aware path loss, and the confidence channel derived from distance and obstruction density.
- For NLoS, use the height-dependent Vinogradov/Saboor family from Proto1: the PLE should decay with ABS height instead of staying constant.
- Anchor the NLoS tail with the A2G elevation-angle excess-loss model from Proto1 as well, then let local obstruction features add the site-specific severity.
- The main takeaways from the papers are:
	- NLoS PLE drops from about 4.5-4.7 near the ground toward about 2.5-3 at higher altitude.
	- Shadow fading is much larger in NLoS than LoS, but it weakens with height.
	- Layout matters enough to justify city-type grouping, but not enough to replace the deterministic shadow/obstruction geometry.
- Local calibration dataset: `C:\TFG\TFGPractice\Datasets\CKM_Dataset_270326.h5`.

## Current Training Rules
- Stage 1 augmentation is disabled.
- Stage 2 augmentation is disabled.
- Stage 2 consumes frozen stage1 outputs from HDF5, so the prior has to stay deterministic and aligned.
