# Try 30 Spread Visual Review and Plan

This note records the visual inspection performed on the current `delay_spread` and `angular_spread` outputs and proposes the next experiment for the spread branch.

## Scope of the review

The review was performed on composite `alltogether2` panels that combine:

- topology
- LoS mask
- antenna-height context
- `path_loss` GT / prediction / error
- `delay_spread` GT / prediction / error
- `angular_spread` GT / prediction / error
- YCbCr composites

The spread review was based on the same 20 manually checked samples used for the recent path-loss inspection, spanning multiple cities and urban layouts.

## Main visual findings for `delay_spread` and `angular_spread`

### 1. The dominant problem is dynamic-range compression

Across many samples, the predictions are not completely random. They often place responses in approximately the correct regions.

However, the dominant error is that the outputs are too weak, too dark, or too compressed in amplitude.

Typical pattern:

- the ground truth contains thin bright streaks or localized hotspots;
- the prediction keeps only a faint version of those structures;
- the rest of the map collapses toward a dark low-contrast background.

This suggests that the model is learning **where** something happens more easily than **how strong** it should be.

### 2. Sparse bright structures are systematically underestimated

The strongest visual errors are often attached to:

- elongated rays,
- bright thin bands,
- sparse concentrated peaks,
- and a few very intense local responses.

In the predictions, those regions are often:

- blurred,
- attenuated,
- or partially erased.

This means that the current loss is still too dominated by the large low-intensity background, so the model is rewarded for getting the average field right while missing the most informative high-value structures.

### 3. The model often preserves coarse support but loses contrast

In several reviewed panels, the prediction is not completely misplaced. The support is often roughly correct:

- road-like or corridor-like structures may still appear;
- some obstacle-aligned streaks survive;
- and the global orientation of the pattern is sometimes recognizable.

But the prediction looks like a low-energy version of the target.

This points to a supervision problem more than to a pure localization problem:

- the model knows part of the geometry,
- but it is not pushed hard enough to preserve strong spread responses.

### 4. The current gradient loss helps structure, but not enough amplitude

The current `Try 26` branch already adds a gradient-based loss, and the review suggests that this was a sensible idea:

- some transitions are cleaner than in older spread baselines;
- some thin structures are at least partially preserved.

But the visual evidence suggests that this is still insufficient.

Why:

- the gradient loss encourages transitions,
- but it does not necessarily force the network to reproduce the correct magnitude of the peaks;
- therefore the model can still produce a structurally plausible but too faint version of the map.

### 5. Delay and angular still look target-specific, but both suffer from the same imbalance

The review suggests that `delay_spread` and `angular_spread` are not failing in exactly the same visual way, but they share a common bottleneck:

- they are both sparse,
- they both contain informative high-value regions,
- and those high-value regions are both underweighted by the current recipe.

That makes it reasonable to design a joint follow-up try again, rather than splitting them immediately.

## Interpretation of the bottleneck

The spread branch does not appear to be primarily blocked by:

- lack of global context,
- or lack of topology information alone.

The more likely bottleneck is:

- **loss imbalance between the sparse high-value structures and the broad low-value background**.

In other words, the current model is still encouraged too strongly to become conservative.

That is a reasonable behavior under standard MSE-style supervision:

- background occupies many more pixels,
- high-value spread structures are rare,
- so the cheapest way to reduce total error is often to predict a muted field.

## Proposed direction for Try 30

The strongest next hypothesis is:

- keep the current structural recipe from `Try 26`,
- but add supervision that explicitly protects the **rare, high-value spread responses**.

## Recommended Try 30 design

### Base

Use `Try 26` as the base:

- bilinear decoder
- multiscale loss
- gradient-aware loss

This is preferable to going back to an older spread branch, because `Try 26` already addresses structure better than the earlier variants.

### New loss 1: value-weighted spread regression

Add a target-aware weighting term for `delay_spread` and `angular_spread`.

Core idea:

- pixels with larger target values should contribute more strongly to the reconstruction loss.

Simple implementation options:

- weight proportional to normalized target intensity;
- or weight using a clipped power law, for example `weight = 1 + alpha * target_norm^gamma`.

Why this is the most direct match to the review:

- the recurring error is underestimation of strong responses;
- this loss directly tells the model that those regions matter more than the dark background.

### New loss 2: high-percentile / hotspot emphasis

Add a second loss that only acts on the strongest target regions.

Practical formulation:

- compute a mask from the target using a percentile threshold per map, for example top `10%` or top `5%`;
- apply an extra MSE or L1 term only inside that hotspot mask.

Why this is useful:

- the review shows that the most visually meaningful failures are often on sparse bright streaks and hotspots;
- a percentile-based mask prevents the model from ignoring those areas just because they are rare.

## Why this is preferable to other ideas

### Why not a transformer-style spread branch first

The current visual evidence does not suggest that the main problem is missing context.

The model often places structures in roughly the right regions, but with insufficient amplitude.

So adding more global modeling before fixing the loss imbalance seems less targeted.

### Why not a multitask `path_loss + spread` branch first

That may still be worth testing later, but it introduces a second source of uncertainty:

- maybe the encoder sharing helps,
- or maybe `path_loss` dominates again.

For the next step, it is cleaner to stay focused on the spread bottleneck itself.

### Why not topology-edge weighting first

Topology-edge weighting helped make sense for `path_loss`, but the spread review suggests that:

- the dominant issue is not only edge placement;
- it is that bright responses are too weak.

So the next change should target amplitude imbalance before obstacle-edge weighting is expanded further.

## Recommended Try 30 wording

The clearest description is:

- `Try 30 = Try 26 + value-weighted spread loss + hotspot-focused spread loss`

This says exactly what the experiment is trying to fix:

- not just structure,
- but the systematic underprediction of high-intensity spread responses.

## How to explain this to a supervisor

A concise and honest explanation would be:

- "After reviewing the composite prediction panels, I observed that the spread branch often gets the rough support right but systematically underestimates the strongest delay and angular responses."
- "That suggests the current loss is still too dominated by the low-intensity background."
- "So the next try should keep the structural improvements from Try 26, but add two new losses that explicitly protect high-value spread regions: one value-weighted regression loss and one hotspot-focused loss."

## Short conclusion

The next spread experiment should probably not be more architectural first.

The visual review suggests that the most targeted next step is:

- keep `Try 26` as the base,
- and redesign the supervision so that strong spread responses are no longer treated as a small detail inside a mostly dark map.
