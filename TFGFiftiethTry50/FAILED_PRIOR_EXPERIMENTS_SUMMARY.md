# Try 50 Failed Prior Experiments Summary

This note summarizes the main prior-calibration branches explored inside
`Try 50`, most of which did not beat the copied `Try 47`-family baseline.

Archived outputs live in:

- `prior_calibration/worse_experiments/`

The only calibration kept at the root of `prior_calibration` is the current
usable baseline:

- `regime_obstruction_train_only_from_try47.json`

## High-level picture

- `LoS` is not the main bottleneck anymore.
- `NLoS` remains the dominant failure mode.
- Formula redesigns repeatedly stalled around `NLoS ~= 42-46 dB`.
- The best archived `NLoS` result so far is still only about `41.01 dB`.

## Main families tried

### 1. Structured / strict formula redesigns

Examples:

- `regime_structured_nlos_directml_quick_*`
- `regime_structured_nlos_strict_los_*`
- `regime_minimal_nlos_strict_los_*`
- `regime_delta_nlos_*`

What they were trying to do:

- preserve `LoS`
- hard-gate `LoS/NLoS`
- model `NLoS` as additive attenuation over a `LoS` backbone

Why they were not enough:

- they did not materially push `NLoS` below the low-`42 dB` range
- they were better for diagnosis than for final performance

### 2. Tabular `delta_nlos` with `HistGradientBoostingRegressor`

Examples:

- `nlos_delta_hgbr_1pct_*`
- `nlos_delta_hgbr_specialist_2pct_*`
- `nlos_delta_hgbr_specialist_oldlos_3pct_*`
- `nlos_delta_hgbr_weighted_oldlos_25train_50val_*`

Best archived result:

- `nlos_delta_hgbr_specialist_2pct_results.json`
- `overall ~= 22.33 dB`
- `LoS ~= 4.81 dB`
- `NLoS ~= 41.01 dB`

Why it still was not enough:

- `NLoS` stayed far above the target regime
- more train / val did not clearly fix the bottleneck

### 3. `HGBoost + torch MLP` residual branch

Example:

- `nlos_delta_hgbr_torchmlp_exactold_5train_150val_*`

Outcome:

- worse than the simpler specialist branch
- `NLoS` got significantly worse

Main lesson:

- more complexity in the residual learner did not help
- it tended to over-push the `NLoS` correction

### 4. Heuristic obstruction-regime experts

Examples:

- `nlos_regime_experts_exactold_5train_150val_*`
- `nlos_regime_experts_exactold_12train_150val_*`

Regimes:

- shallow transition
- deep shadow / canyon
- high blocker / rooftop
- dense clutter

Outcome:

- useful as a conceptual test
- but worse than the best tabular specialist branch

### 5. Modern vs old-exact larger-sample comparison

Examples:

- `nlos_delta_hgbr_specialist_modern_8train_12val_*`
- `nlos_delta_hgbr_specialist_oldexact_8train_12val_*`

Outcome:

- `modern` collapsed badly:
  - `overall ~= 51.81 dB`
  - `NLoS ~= 91.22 dB`
- `old_exact` stayed more stable, but still poor:
  - `overall ~= 24.28 dB`
  - `NLoS ~= 42.52 dB`

Main lesson:

- the modern baseline branch appears unstable under this specialist recipe
- the old-exact baseline is safer, but still not enough

## Practical conclusion

The `Try 50` prior research did produce useful negative evidence:

- hard gating was correct;
- preserving `LoS` exactly was correct;
- but the current `NLoS` paradigm is still insufficient.

So the current working posture is:

- keep the copied `Try 47` calibration as the usable baseline;
- treat the archived `Try 50` prior branches as diagnostic evidence for the next
  bigger redesign.
