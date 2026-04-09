# Try 50 NLoS Prior Redesign

## Status

This note now serves mainly as a redesign log, not as the description of a
winning branch.

The redesign families proposed here were useful for diagnosis, but none of them
beat the copied `Try 47` calibration strongly enough to replace it as the
practical baseline.

Current practical baseline still in use:

- `prior_calibration/regime_obstruction_train_only_from_try47.json`

Archived redesign outputs:

- `prior_calibration/worse_experiments/`

## Why the redesign was attempted

The repeated pattern was:

- `LoS` around the right scale
- `NLoS` dramatically worse than target

That justified trying a larger redesign where:

- `LoS` stays fixed
- only `NLoS` changes

## Main redesign ideas that were tested

### 1. Strict hard-gated LoS / NLoS split

Instead of a soft blend:

- `LoS` pixels use the `LoS` branch exactly
- `NLoS` pixels use a separate `delta` branch

This was the right conceptual change, but by itself it did not solve the
`NLoS` bottleneck.

### 2. Structured additive NLoS loss

Tried the shape:

- `PL_NLoS = PL_LoS_reference + delta_break + delta_shadow_depth + delta_blocker + delta_context`

This improved interpretability, but not enough performance.

### 3. Minimal NLoS delta

Tried simplifying the extra attenuation to:

- `break_loss + clipped_shadow_depth + clipped_context`

This also did not materially reduce the `NLoS` RMSE.

### 4. Regime-specific NLoS experts

Tried heuristic gating into:

- shallow transition
- deep shadow / canyon
- high blocker / rooftop
- dense clutter

Again useful diagnostically, but still worse than the best tabular
`delta_nlos` specialist branch.

## What we learned

- the hard-gated `LoS/NLoS` split was conceptually correct;
- the main failure is still the `NLoS` object being modeled;
- simple formula redesigns are not enough;
- heuristic regime experts alone are not enough either.

The best archived `NLoS` result still came from a tabular specialist branch:

- `nlos_delta_hgbr_specialist_2pct_results.json`
- `NLoS RMSE ~= 41.01 dB`

That is still not good enough, but it remains better than the formula redesign
branches documented here.

## Practical conclusion

This redesign note remains useful as rationale, but it should be read as:

- a map of tested ideas,
- not as the current recommended production path.

For a more complete list of what was tried and how it failed, see:

- [FAILED_PRIOR_EXPERIMENTS_SUMMARY.md](C:/TFG/TFGpractice/TFGFiftiethTry50/FAILED_PRIOR_EXPERIMENTS_SUMMARY.md)
