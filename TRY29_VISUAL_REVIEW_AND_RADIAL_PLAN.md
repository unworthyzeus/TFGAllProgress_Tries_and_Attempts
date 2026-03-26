# Try 29 Visual Review and Radial Plan

This note records the visual review performed before opening `Try 29`, and explains why the next `path_loss` branch was designed around radial supervision.

## Scope of the review

Before defining `Try 29`, a manual review was performed on **20 composite panels**, each panel containing **15 images** (`alltogether2` layout).

The 20 reviewed samples were:

- `Abidjan/sample_00001`
- `Abidjan/sample_00088`
- `Abu_Dhabi/sample_00176`
- `Abu_Dhabi/sample_00728`
- `Alexandria/sample_01281`
- `Alexandria/sample_01403`
- `Athens/sample_01526`
- `Athens/sample_01713`
- `Bangalore/sample_01901`
- `Bangalore/sample_02101`
- `Bath/sample_02301`
- `Bath/sample_02318`
- `Berlin/sample_02336`
- `Berlin/sample_02471`
- `Bilbao/sample_02606`
- `Bilbao/sample_02631`
- `Bratislava/sample_02656`
- `Bratislava/sample_02668`
- `Brugge/sample_02681`
- `Brugge/sample_02711`

These samples span different cities, urban layouts, and UAV / antenna heights.

## Main visual findings

### 1. The dominant `path_loss` failure is radial

The most repeated issue is not only local blur or missing edges.

Across many samples, the prediction fails to reproduce the **transmitter-centered circular carrier pattern** that is clearly visible in the ground truth. Instead of reconstructing the concentric structure, the model often produces:

- a smooth central bowl or blob,
- a monotonic low-frequency field,
- and only part of the obstacle-induced modulation on top of that.

This means that the model is learning some local urban interaction, but not the full radial propagation structure that should already be present before obstacles modulate the field.

### 2. The error is not concentrated only around building edges

The review shows that some of the remaining error is indeed close to urban transitions and obstacle boundaries, but a large part of the `path_loss` error remains in relatively open regions.

That suggests that the current bottleneck is not just:

- "the model does not respect edges"

but also:

- "the model does not reconstruct the correct radial base field".

This is an important distinction, because it means that adding only topology-edge regularization is not sufficient.

### 3. Local building silhouettes are sometimes captured on top of the wrong global base

In multiple panels, the prediction seems to understand where strong obstructions are located, but the whole field is still built on a low-frequency map that is too smooth and too weakly radial.

In practical terms:

- the model can partially place shadowing where buildings exist,
- but the carrier field underneath is already wrong.

This is exactly the pattern expected when the network learns local corrections better than the global transmitter-centered structure.

### 4. `delay_spread` and `angular_spread` still underuse dynamic range

The review also confirms a repeated issue in the `Try 26` outputs:

- the predictions often collapse toward a dark or low-contrast background,
- sparse hot structures are only partially recovered,
- and the strongest streaks / hotspots are often underestimated.

So even though `Try 26` is more structured than earlier baselines, there is still a dynamic-range compression effect in `delay/angular`.

### 5. YCbCr composites are diagnostically useful but visibly unbalanced

The YCbCr panels are useful because they quickly show whether the three targets are aligned or not, but visually they are dominated by luminance and scale imbalance.

This is not itself a model failure, but the review suggests that:

- `path_loss` dominates the composite,
- `angular` and `delay` do not contribute with equally strong contrast,
- and green-dominant outputs should be interpreted carefully as diagnostics rather than as balanced physical visualizations.

### 6. Antenna height is present, but its effect is not obviously strong enough in the radial pattern

The review across low and high UAV / antenna heights does not show a consistently strong change in the radial structure spacing or amplitude in the predictions.

This does not prove that the scalar conditioning is useless, but it does suggest that the model may still be underusing height information relative to the much stronger local image cues.

## Why `Try 29` was opened

The visual review suggests that `Try 22` is still the strongest clean `path_loss` base, but it also leaves a clear unresolved issue:

- the model does not reconstruct the radial / circular transmitter-centered pattern strongly enough.

For that reason, `Try 29` is built on top of the **`Try 22` base**, not on top of `Try 28`.

That design choice is intentional:

- `Try 22` remains the stronger and cleaner baseline;
- `Try 28` added complexity but did not improve enough over `Try 22`;
- therefore the next branch should inject a **new physically targeted supervision signal** rather than stacking more architectural complexity.

## What `Try 29` adds

`Try 29` adds **two radial losses at the same time**:

### 1. Radial profile loss

This loss compares the predicted and ground-truth `path_loss` after averaging them over concentric radial bins.

Its job is to answer:

- "Is the model learning the right radial field profile as distance from the transmitter increases?"

This directly targets the missing circular carrier structure seen in the review.

### 2. Radial gradient loss

This loss compares the **change between adjacent radial bins** in the prediction and in the ground truth.

Its job is to answer:

- "Even if the absolute radial profile is imperfect, is the model at least learning the correct radial slope and oscillatory behavior?"

This is useful because the visual issue is not only wrong absolute brightness, but also wrong radial progression.

## Why these two losses were chosen together

Using both losses at once is more informative than using only one:

- the radial profile loss controls the large radial trend,
- the radial gradient loss controls how that trend evolves from one ring to the next.

Together they provide explicit supervision for the part of the field that the visual review suggests is still underlearned.

## Note on negative `path_loss` predictions

During export, some predicted `path_loss` values were observed below `0 dB`, for example:

- `pred min = -9.684 dB`

This can happen because the model output is not intrinsically constrained: after denormalization, the predicted value is simply:

- `prediction * scale + offset`

with `scale = 180` and `offset = 0`.

So:

- this does **not** mean normalization is broken,
- but it **does** mean the network is allowed to output physically implausible values unless an explicit clamp or penalty is added.

At the moment, this affects evaluation in the normal way:

- the metrics do not crash,
- but those negative values are counted as real error in dB space,
- and they can make both visualization and RMSE look worse.

This is another sign that the model is not yet sufficiently anchored to the physical `path_loss` structure.

## Short conclusion

The review of 20 composite panels suggests that the main unresolved `path_loss` bottleneck is now **radial structure**, not only decoder artifacts or topology edges.

That is the reason `Try 29` was defined as:

- `Try 22` base
- plus radial profile loss
- plus radial gradient loss

rather than as another purely architectural variant.
