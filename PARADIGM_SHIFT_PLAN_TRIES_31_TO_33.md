# Paradigm Shift Plan After Tries 29 and 30

This note summarizes why `Try 29` and `Try 30` should be treated as evidence that the project is reaching the limit of the current formulation, and proposes the next experiments as **formulation changes**, not just more auxiliary losses.

## 1. Why the current formulation looks saturated

Recent results suggest that the project is no longer mainly limited by:

- decoder choice alone,
- one more auxiliary loss,
- one more attention block,
- or one more weighting tweak.

The stronger interpretation is that the current one-shot dense regression setup is reaching its limit.

## 2. Evidence from the latest tries

## Path loss

Reference branch:

- `Try 22`
- best `path_loss.rmse_physical = 16.669 dB` at epoch `75`

Recent follow-up:

- `Try 29`
- best `path_loss.rmse_physical = 17.106 dB` at epoch `58`

Meaning:

- radial auxiliary losses did **not** outperform the stronger clean `Try 22` base;
- the gap is around `+0.44 dB` in RMSE, which is too large to dismiss as noise.

## Delay / angular spread

Reference branch:

- `Try 26`
- best `delay_spread.rmse_physical = 23.497 ns`
- best `angular_spread.rmse_physical = 8.384 deg`

Recent follow-up:

- `Try 30`
- best `delay_spread.rmse_physical = 23.609 ns`
- best `angular_spread.rmse_physical = 8.446 deg`

Meaning:

- the value-weighted and hotspot-focused losses also failed to beat the stronger `Try 26` baseline;
- the gap is smaller than in path loss, but still negative.

## 3. Interpretation

The lesson is not that all recent ideas were useless. The lesson is:

- the project now needs a **different problem decomposition**, not just a richer version of the same objective.

In other words, the current formulation still asks a single image-to-image model to learn:

- the global physical carrier field,
- local obstacle modulation,
- sparse strong spread responses,
- and dynamic range,

all in one direct regression output.

That is likely the real bottleneck.

## 4. Paradigm shift for path loss: Try 31

## Proposed idea

Move from:

- direct full-map `path_loss` regression

to:

- **physical prior + learned residual**

### Core formulation

Predict:

- `final_path_loss = clipped_physical_prior + learned_residual`

where the prior is a simple transmitter-centered field such as:

- free-space path loss,
- log-distance path loss,
- or another calibrated radial baseline.

### Why this is a paradigm shift

This changes the question from:

- “Can the network learn the entire field from scratch?”

to:

- “Can the network learn how the real environment deviates from a physically meaningful baseline?”

### Why it fits the current evidence

The reviewed outputs suggest that the model often captures some obstacle modulation while still underlearning the radial carrier field.

That is exactly the situation where residual learning over a prior should help.

### Suggested Try 31 design

- base: clean `Try 22` path-loss branch
- inputs:
  - topology
  - LoS mask
  - distance map
  - physical prior map
  - antenna-height conditioning
- outputs:
  - residual map
  - optional confidence / uncertainty map
- losses:
  - reconstruction on final path loss
  - reconstruction on residual
  - optional small radial consistency term only on the final output
- output constraint:
  - bounded final path-loss range

## 5. Paradigm shift for spread: Try 32

## Proposed idea

Move from:

- direct dense regression of `delay_spread` and `angular_spread`

to:

- **two-stage support + amplitude prediction**

### Core formulation

For each spread target:

- head A predicts a support / hotspot map
- head B predicts the amplitude conditioned on that support

Then the final prediction is the masked or support-guided amplitude field.

### Why this is a paradigm shift

This separates two different tasks that are currently entangled:

- where strong responses exist
- how large those responses are

### Why it fits the current evidence

The recent visual review suggests that spread prediction often gets the rough support approximately right but underestimates the strongest responses.

That is exactly the situation where a support-first formulation is more appropriate than one-shot regression.

### Suggested Try 32 design

- base: clean `Try 26` spread branch
- outputs per target:
  - support / hotspot logits
  - amplitude map
- losses:
  - BCE / focal loss on support
  - masked regression on amplitude inside target support
  - lighter full-map regression term for stability
- optional target space:
  - predict spread in log-space to reduce dynamic-range compression

## 6. Optional downstream shift: Try 33

If `Try 31` works, the next step could be:

- **path-loss-informed spread prediction**

Idea:

- first obtain a better physically structured path-loss field;
- then condition spread prediction on that field or on its latent representation.

Why this might help:

- spread responses and path loss are not identical,
- but they are both shaped by the same propagation geometry,
- so a cleaner path-loss branch may become a useful structural prior for spread.

## 7. Why this is better than more incremental tries

At this point, more tries of the form:

- “same model + one more auxiliary loss”
- “same model + one more weighting”
- “same model + slightly more attention”

are unlikely to produce a decisive jump.

The most promising next step is to change the task decomposition itself.

## 8. Recommended order

1. `Try 31`: path loss as physical prior + residual
2. `Try 32`: spread as support + amplitude
3. `Try 33`: optional path-loss-informed spread branch if `Try 31` is genuinely better

## 9. Supervisor-facing explanation

A clear explanation would be:

- “The latest tries suggest that the current image-to-image formulation is saturating.”
- “For path loss, the model still struggles to learn the physical carrier field from scratch.”
- “For delay and angular spread, the model still entangles support detection with amplitude regression.”
- “So the next step should not be another small tuning branch, but a reformulation of the prediction problem itself.”
