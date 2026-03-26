# Tries 15-19: Common Changes vs New Tries

This note records the decision to separate:

- changes that are reasonable to apply to all current tries, and
- changes that should remain isolated as new tries because they change the modeling hypothesis too much.

## Current status

The running jobs were cancelled before this cleanup so the current family can be kept consistent.

## Changes applied to all current tries (15-19)

These are considered low-risk or broadly useful enough to include everywhere in the current family.

### 1. Group normalization instead of batch normalization

Applied to all current tries.

Reason:

- the jobs run with very small effective batch sizes,
- the prediction panels showed local contrast instability and grid-like artifacts,
- `group norm` is usually safer than `batch norm` in this regime.

### 2. Antenna height taken from the HDF5 that actually contains it

Applied to all current tries.

Reason:

- previous notes already showed that using the wrong source can silently produce zero-filled height,
- this is not an experimental hypothesis, it is a data-correctness requirement.

### 3. Explicit distance-map channel

Applied to all current tries.

Reason:

- the GT path-loss maps show strong radial / distance-dependent structure,
- this geometric cue is cheap and physically meaningful,
- the network should not be forced to infer distance only implicitly from image texture.

### 4. Wider postprocess kernel search

Applied to all current tries with `path_loss_median_kernel_candidates: [1, 3, 5]`.

Reason:

- the current errors mix local noise and over-smoothed regions,
- a wider kernel search is a calibration aid, not a new modeling family.

### 5. 2-GPU launch support

Prepared for all current tries.

Reason:

- this is an execution detail, not a scientific hypothesis,
- if account limits allow it, the same family should be runnable with the same code path in DDP.

## Changes intentionally kept as try-specific

These remain separated across tries because they are part of the comparison.

### 1. GAN weight

Kept try-specific.

Reason:

- whether the adversarial term helps or only adds fake texture is one of the main questions,
- making it identical everywhere would destroy that comparison.

### 2. Learning-rate aggressiveness

Kept try-specific.

Reason:

- some tries are meant to be more conservative than others,
- LR interacts with GAN weight and regularization, so it is still part of the experiment.

### 3. Default postprocess kernel

Kept try-specific when useful, even if the candidate set is common.

Reason:

- the selected default can still express a hypothesis,
- the important universal part is that all of them are allowed to search the same small range.

## Changes that should become new tries, not silent edits

These are big enough that they should not be folded into the current family after the fact.

### New try family A: Decoder fix

Candidate change:

- replace `ConvTranspose2d` upsampling with `bilinear/nearest + conv`.

Why this must be a new try:

- it directly changes the source of the checkerboard artifacts,
- it alters the image formation behavior of the whole decoder.

### New try family B: Residual over physical prior

Candidate change:

- predict a residual over a physical / heuristic prior instead of predicting the full path-loss field directly.

Why this must be a new try:

- it changes what the network is learning,
- it is a major modeling assumption, not a small config tweak.

### New try family C: Multiscale / edge-aware loss

Candidate change:

- add coarse-scale loss,
- add gradient or edge-aware penalties,
- emphasize geometry transitions near buildings and shadows.

Why this must be a new try:

- the optimization target changes in a non-trivial way,
- improvements would no longer be attributable only to the current architecture family.

### New try family D: Larger-context backbone

Candidate change:

- residual blocks,
- dilated convolutions,
- stronger encoder / bottleneck context.

Why this must be a new try:

- this moves beyond parameter tuning into a different architecture capacity regime.

### New try family E: Explicit uncertainty head

Candidate change:

- add a head that predicts reliability / uncertainty of the map.

Why this must be a new try:

- it changes the output space and the way postprocessing would be used.

### New try family F: Regime routing / mixture of experts

Candidate change:

- route by LoS dominance, distance regime, or urban bucket.

Why this must be a new try:

- it changes the training decomposition itself,
- it is no longer a single-backbone comparison.

## Practical rule going forward

If a change is:

- about data correctness,
- a low-risk geometric cue,
- or a calibration detail that helps every run fairly,

then it can be applied to all current tries.

If a change:

- alters the decoder,
- changes the loss definition,
- changes the prediction target,
- or changes the model routing / architecture class,

then it should be introduced as a new try.
