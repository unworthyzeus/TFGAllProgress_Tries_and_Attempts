# Tries 37-40: New Dataset Refresh and Mask Verification

This note documents the restart of the masked-supervision family on the new dataset:

- `CKM_Dataset_270326.h5`

The goal was to preserve the experimental logic of `Try 33-36`, while removing any ambiguity about dataset versioning and confirming that masking is applied consistently in training, evaluation, and export.

## Core rule that was verified

Pixels where:

- `topology_map != 0`

are treated as building / non-ground pixels and therefore:

- they are excluded from the training loss,
- they are excluded from validation and test error accumulation,
- and they are exported as invalid in visual error maps.

This was checked directly in code and in sample-level tensor inspection.

## Where the mask is applied during training

For the corrected masked branches, the reconstruction loss receives a mask tensor and multiplies the per-pixel loss by that mask before reduction.

Relevant code locations:

- `TFGThirtyThirdTry33/train_cgan.py`
- `TFGThirtyFourthTry34/train_cgan.py`
- `TFGThirtySixthTry36/train_cgan.py`

The reconstruction loss uses the pattern:

- `msk = masks[:, i : i + 1]`
- `masked = raw * msk`

This means masked-out pixels do not contribute to the training objective.

## Important bug that was fixed for path loss

In the masked path-loss branch, there was an earlier bug where the saturation mask could overwrite the path-loss validity mask and accidentally re-enable building pixels.

That was corrected so that the path-loss mask keeps both constraints at once:

- valid non-building receiver pixels only,
- and non-saturated path-loss pixels only.

After the fix, direct tensor inspection on validation samples confirmed:

- building pixels remained invalid for `path_loss`,
- and export-style error maps correctly produced `NaN` on those building pixels.

## What each new try means

## Try 37

- new-dataset rerun of `Try 33`
- target: `path_loss`
- base logic: `Try 22`
- only supervision-policy change:
  - exclude building pixels from loss and metrics

Purpose:

- isolate the building-mask effect again, but now on `CKM_Dataset_270326.h5`

## Try 38

- new-dataset rerun of `Try 34`
- target: `path_loss`
- building-mask exclusion
- hybrid `two_ray_ground + COST231` formula map used as an extra conditioning input
- model enlarged relative to the tiny debug-sized `Try 34`

Purpose:

- test whether the explicit propagation prior helps more on the new dataset

## Try 39

- new-dataset rerun of `Try 35`
- targets:
  - `delay_spread`
  - `angular_spread`
- lighter 1-GPU spread branch
- building-mask exclusion

Purpose:

- keep the lighter spread-side reference branch available on the new dataset

## Try 40

- new-dataset rerun of `Try 36`
- targets:
  - `delay_spread`
  - `angular_spread`
- clean `Try 26`-style spread baseline
- building-mask exclusion

Purpose:

- keep the clearer 2-GPU spread comparison branch on the new dataset

## Practical conclusion

The main point of `Try 37-40` is not architectural novelty by itself. The main point is to remove two sources of ambiguity:

- older-dataset versus newer-dataset behavior,
- and whether building pixels were really excluded from optimization or only hidden in post-processing.

This family is therefore the cleaner masked-supervision restart on the new dataset.
