# Try 14 / Try 22 Reevaluation and Building-Mask Interpretation

This note records the retrospective reevaluation of older strong path-loss checkpoints using the modern ground-only mask.

## Why this reevaluation was needed

Older comparisons were mixing two different evaluation definitions:

- legacy evaluation:
  - no explicit exclusion of `topology != 0`
- modern evaluation:
  - only ground pixels are valid
  - building pixels are excluded from loss and error

That difference matters a lot, especially for `NLoS`.

## Main correction about Try 14

The previous intuition was:

- `Try 14 NLoS` reached about `23 dB`
- therefore the project used to solve `NLoS` much better

That statement was only true under the **legacy** metric.

### Try 14 NLoS, reevaluated

- legacy mask:
  - overall RMSE: `23.08 dB`
  - `LoS`: `8.20 dB`
  - `NLoS`: `26.62 dB`
  - file: [try14_nlos_legacy_reeval.json](C:/TFG/TFGpractice/analysis/try14_nlos_legacy_reeval.json)

- modern ground-only mask:
  - overall RMSE: `28.81 dB`
  - `LoS`: `8.20 dB`
  - `NLoS`: `37.46 dB`
  - file: [try14_nlos_masked_reeval.json](C:/TFG/TFGpractice/analysis/try14_nlos_masked_reeval.json)

## Why the masked result becomes much worse

The mask does **not** make the model worse.

What happens is:

- the modern mask removes a very large number of `NLoS` pixels
- those removed pixels were mostly invalid building locations
- many of those pixels were easier for the old model than the true ground-level `NLoS` receiver locations

So the masked metric is harder because it is more honest.

This is the key interpretation:

- excluding `topology != 0` does not degrade the model itself
- it removes artificially favorable pixels from the evaluation

## Important correction about LoS vs NLoS specialization in Try 14

Another possible misunderstanding was:

- maybe the old `NLoS` model was even better at `LoS` than the old `LoS` model

This is **not** true.

### Try 14 LoS, reevaluated

- legacy mask:
  - overall RMSE: `13.99 dB`
  - `LoS`: `5.16 dB`
  - `NLoS`: `25.50 dB`
  - file: [try14_los_legacy_reeval.json](C:/TFG/TFGpractice/analysis/try14_los_legacy_reeval.json)

- modern ground-only mask:
  - overall RMSE: `15.05 dB`
  - `LoS`: `5.16 dB`
  - `NLoS`: `36.21 dB`
  - file: [try14_los_masked_reeval.json](C:/TFG/TFGpractice/analysis/try14_los_masked_reeval.json)

Comparison on `LoS` pixels:

- `Try 14 LoS` on `LoS`: `5.16 dB`
- `Try 14 NLoS` on `LoS`: `8.20 dB`

So the dedicated `LoS` model was indeed better on `LoS`.

The earlier confusion came from the old split definition:

- `los_only` / `nlos_only` were sample-dominant splits
- they were not pure per-pixel regime splits

That means an old `nlos_only` split still contained many `LoS` pixels.

## Try 22, reevaluated the same way

`Try 22` had looked like a very strong path-loss baseline, and it still is strong in `LoS`.

### Try 22 reevaluation

- legacy mask:
  - overall RMSE: `17.33 dB`
  - `LoS`: `3.78 dB`
  - `NLoS`: `24.40 dB`
  - file: [try22_legacy_reeval.json](C:/TFG/TFGpractice/analysis/try22_legacy_reeval.json)

- modern ground-only mask:
  - overall RMSE: `19.94 dB`
  - `LoS`: `3.78 dB`
  - `NLoS`: `34.43 dB`
  - file: [try22_masked_reeval.json](C:/TFG/TFGpractice/analysis/try22_masked_reeval.json)

This shows the same pattern:

- `LoS` remains very strong
- the apparent old `NLoS` strength weakens substantially once buildings are excluded

## What this means for current priorities

The correct current reading is:

- the modern metric is harder, but also more meaningful
- `LoS` is already reasonably well solved
- the true unresolved bottleneck is modern, ground-only `NLoS`

## Clarification about Try 22 vs the newer prior-based line

The accurate statement is:

- `Try 22` under the modern masked evaluation is no longer as strong as it looked under the old metric
- and it is already slightly weaker than the best measured prior-based PMNet branch (`Try 42`) in overall RMSE

Current relevant comparison:

- `Try 22` masked overall: `19.94 dB`
- `Try 42` overall: about `19.78 dB`

So the fair conclusion is:

- `Try 22` is not clearly better than the newer prior-based line
- but the real unresolved problem is still masked, ground-only `NLoS`
