# Try 50 Prior Improvements

## Current Direction
- The main goal of Try 50 is to improve the NLoS prior formula itself.
- The practical active baseline is still the copied `Try 47` calibration:
  - `TFGFiftiethTry50/prior_calibration/regime_obstruction_train_only_from_try47.json`
- Most newer prior experiments were archived because they did not beat that
  baseline:
  - `TFGFiftiethTry50/prior_calibration/worse_experiments/`
- The current `Try 50` branch should be read as a prior-research sandbox, not
  yet as a production-ready replacement for the `Try 47` prior.

## Why This Branch
- The plain `hybrid_two_ray_cost231_a2g_nlos` blend was still too generic for NLoS.
- The deepshadow branch keeps the LoS path separate but makes the NLoS side more conservative and obstruction-aware.
- That matches the goal for Try 50: improve the prior specifically where the NLoS tail is failing.

## Current practical lesson

So far, the project has not found a `Try 50` prior that beats the copied
`Try 47` calibration in a convincing way.

The best archived `NLoS` value reached by the tabular-delta experiments is
still only around:

- `NLoS RMSE ~ 41.01 dB`

That result is archived in:

- `TFGFiftiethTry50/prior_calibration/worse_experiments/nlos_delta_hgbr_specialist_2pct_results.json`

This means the branch has been useful for diagnosis, but not yet for release.

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

## Where the failed runs went

The earlier `Try 50` formula redesigns, strict-LoS experiments, clipped-delta
tests, regime experts, and modern-vs-oldexact comparisons were moved to:

- `TFGFiftiethTry50/prior_calibration/worse_experiments/`

The main summary of those failures now lives in:

- `TFGFiftiethTry50/FAILED_PRIOR_EXPERIMENTS_SUMMARY.md`
