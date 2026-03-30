# Supervisor Summary of Tries 20-40

This note is intended for supervisor-facing discussion. It summarizes what was tested from `Try 20` onward, what each family meant in practical terms, what worked best, and why `Try 37-40` were opened on the newer dataset.

## Main stable conclusions so far

- `Try 22` remained the strongest clean `path_loss` baseline from the older dataset family.
- `Try 26` remained the strongest clean `delay_spread + angular_spread` baseline from the older dataset family.
- Several later tries were still useful even when they did not beat those baselines:
  - they clarified which bottlenecks were not dominant,
  - and they motivated a cleaner masked-supervision restart.

## Why the masked-supervision restart was needed

After reviewing both prediction maps and dataset interpretation, a simpler physical rule became important:

- pixels where `topology_map != 0` should not be treated as valid receiver points.

That rule matters in three places:

- training loss,
- validation/test metrics,
- and visual error maps.

The newer family therefore focused on ensuring that building pixels were truly excluded from supervision rather than only hidden during visualization.

## The new dataset branch

The older masked family was recreated on:

- `CKM_Dataset_270326.h5`

The new tries are:

- `Try 37`:
  - `path_loss`
  - equivalent in spirit to `Try 33`
  - same clean `Try 22`-style structure
  - building-mask exclusion only

- `Try 38`:
  - `path_loss`
  - equivalent in spirit to `Try 34`
  - hybrid `two_ray_ground + COST231` prior used as an extra input map
  - building-mask exclusion
  - model size raised relative to the tiny debug version of `Try 34`

- `Try 39`:
  - `delay_spread + angular_spread`
  - equivalent in spirit to `Try 35`
  - lighter 1-GPU masked spread branch

- `Try 40`:
  - `delay_spread + angular_spread`
  - equivalent in spirit to `Try 36`
  - cleaner 2-GPU masked spread branch

## Important implementation clarification

For the corrected masked branches, the building mask is applied in the actual optimization path:

- masked pixels do not contribute to the reconstruction loss,
- masked pixels do not contribute to validation/test error accumulation,
- and exported error maps mark them as invalid.

So the model is trained only on the remaining valid receiver points, even though it may still use topology as context in the input.

## Why this matters scientifically

This makes the newer branch easier to defend:

- it is not just another hyperparameter sweep,
- it is a cleaner statement about what the model is and is not being asked to predict.

The new family therefore improves interpretability even before considering whether it improves RMSE.
