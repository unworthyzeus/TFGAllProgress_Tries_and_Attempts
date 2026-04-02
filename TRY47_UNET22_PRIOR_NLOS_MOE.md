# Try 47: Try-22 U-Net + Calibrated Prior + NLoS MoE

## Why this branch exists

`Try 42` showed that the calibrated physical prior helps, especially in the harder dataset and in `NLoS`.

`Try 46` showed that explicitly separating `LoS` and `NLoS` is a sensible idea, but the PMNet-style trunk still did not recover the spatial stability that the older `Try 22` family had.

`Try 47` is therefore a deliberate synthesis:

- keep the calibrated prior/residual formulation from the recent path-loss branches,
- keep explicit `LoS / NLoS` specialization,
- but return to the stronger image-to-image backbone that worked best in the older family.

In short:

- `Try 42` contributed the prior/residual idea,
- `Try 46` contributed the regime split and `NLoS` specialization,
- `Try 22` contributed the stronger spatial backbone.

## Architectural recipe

`Try 47` uses the exact design principles that made `Try 22` strong:

- bilinear upsampling instead of transposed-convolution-heavy decoding,
- group normalization for `batch_size = 1`,
- scalar FiLM conditioning for antenna height,
- explicit distance-map input channel,
- multiscale path-loss supervision.

On top of that, it adds:

- calibrated physical prior input,
- final prediction defined as `path_loss = calibrated_prior + learned_residual`,
- one lightweight `LoS` residual head,
- one `NLoS` residual branch with mixture-of-experts,
- obstruction proxy channels used only as extra context for the difficult `NLoS` correction.

## Obstruction proxy channels

The model now receives four extra obstruction-oriented channels:

- `shadow_depth`
- `distance_since_los_break`
- `max_blocker_height`
- `blocker_count`

These are not exact ray-tracing outputs. They are angle-binned ray proxies computed from:

- `topology_map`
- `los_mask`
- transmitter-centered geometry

The goal is to expose the network to information that is much closer to the real `NLoS` mechanism than raw topology alone.

## Loss design

The loss keeps the standard path-loss objective but adds specialized supervision:

- global final prediction loss on the masked ground-valid pixels,
- residual loss relative to the calibrated prior,
- multiscale path-loss loss,
- `LoS`-only residual-head loss,
- `NLoS`-only residual-head loss,
- extra weighted `NLoS` combo losses for hard subsets such as:
  - `low_ant + deep_shadow`
  - `mid_ant + deep_shadow`
  - `dense_highrise` subsets

This is intentionally more targeted than simply making the backbone larger.

## Calibration pipeline

`Try 47` is not trained directly on the raw formula prior. It depends on a train-only calibrated JSON stored inside the try folder:

- `prior_calibration/regime_obstruction_train_only.json`

That calibration is generated before training through a separate cluster job. The training job is submitted with an `afterok` dependency so that the network starts only after the calibrated prior has been refreshed.

This matters because `Try 47` is designed around the calibrated-prior formulation, not around the raw formula.

In practice, the calibration job was also adjusted to be cluster-safe:

- it now logs progress explicitly,
- it uses sample-level subsampling for both `train` and `val`,
- and it keeps pixel-level subsampling inside the selected training samples.

This was necessary because the first CUDA calibration attempt timed out after four hours while producing almost no runtime visibility.

So the current `Try 47` calibration workflow is not just "run the same script on GPU".
It is:

- train-only,
- CUDA-enabled,
- progress-logged,
- and constrained to a reproducible subsampled calibration regime that can realistically finish within the cluster time limit.

## Important implementation note

The requested "warm-start from `Try 42`" cannot be performed as a direct weight load, because:

- `Try 42` uses a PMNet-style backbone,
- `Try 47` uses a U-Net-style backbone.

Those parameter shapes are not compatible.

So `Try 47` keeps the calibrated prior from the `Try 42` family, but the backbone itself is restarted in the `Try 22` architectural family.

## Expected behavior

The intended upside is:

- keep the already-good `LoS` behavior near the prior,
- recover the stronger spatial decoding behavior of `Try 22`,
- and push the `NLoS` correction harder through explicit specialization instead of one shared residual head.

The key question for this branch is therefore:

- does `Try 47` improve modern masked `NLoS` RMSE without sacrificing the already-strong `LoS` regime?
