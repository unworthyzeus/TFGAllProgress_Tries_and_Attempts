# Tries 33-36: Building-Mask Exclusion and Physical-Prior Branches

This note documents the new experiment family opened after reviewing the weak results of `Try 31` and `Try 32`.

## Main change in supervision policy

The key dataset-level change is:

- pixels where `topology_map != 0` are treated as non-ground / building pixels;
- they are excluded from training loss;
- they are excluded from validation/test error accumulation;
- and they should also be excluded from visual error maps.

The idea is simple: buildings are not valid receiver locations, so they should not be predicted as if they were measured path-loss or spread samples.

## Try 33

- target: `path_loss`
- base: `Try 22`
- decoder and loss recipe kept aligned with the strongest clean `path_loss` baseline
- building pixels excluded from loss and metrics
- no explicit physical prior
- no extra formula input channel

Purpose:

- isolate the effect of removing building pixels from supervision while keeping the rest of the successful `Try 22` formulation essentially unchanged.

## Try 34

- target: `path_loss`
- base: new path-loss physical-prior branch
- new prior/input formula: hybrid `two_ray_ground` for LoS-like behavior + `COST231-Hata` fallback for non-LoS / urban behavior
- extra conditioning: the formula map is also fed as an input channel
- building pixels excluded from loss and metrics

Purpose:

- test whether a more explicitly radial LoS prior helps the transmitter-centered structure while still keeping an urban empirical fallback.

## Try 35

- targets: `delay_spread`, `angular_spread`
- lighter 1-GPU spread-side branch
- kept separate from the clean 2-GPU rerun of the older spread baseline
- building pixels excluded from loss and metrics

Purpose:

- keep one lighter spread-oriented branch running while the cleaner 2-GPU baseline rerun is evaluated separately.

## Try 36

- targets: `delay_spread`, `angular_spread`
- base: `Try 26`
- same decoder and loss recipe as the strongest clean spread baseline
- building pixels excluded from loss and metrics

Purpose:

- isolate the effect of removing building pixels from supervision while keeping the rest of the successful `Try 26` spread recipe aligned with the older baseline.

## Evaluation and export implications

The new family also requires consistent downstream handling:

- evaluation must respect the building mask when accumulating errors;
- exported GT/prediction images must preserve the masked regions correctly;
- error visualizations should ignore building pixels instead of painting them as valid prediction errors.

That is why the support code was also updated alongside the new tries.
