# Try76 LoS vs NLoS Note

## Observation

In the newest Try76 runs, the apparent regime ordering has flipped compared to
the older branches:

- `open_sparse_lowrise_los` is currently worse than `open_sparse_lowrise_nlos`
- `open_sparse_vertical_los` is currently worse than `open_sparse_vertical_nlos`

This is the opposite of what happened in the older model families, where NLoS
was worse by a large margin.

## Current run numbers

Best validation RMSE among the currently available Try76 runs:

| expert | best epoch | best val RMSE (dB) | val map_nll | val dist_kl | val moment_match | score |
|---|---:|---:|---:|---:|---:|---:|
| `open_sparse_lowrise_los` | 40 | **4.266** | 2.722 | 0.052 | 1.632 | 4.551 |
| `open_sparse_lowrise_nlos` | 25 | **2.298** | 2.197 | 0.304 | 1.603 | 2.594 |
| `open_sparse_vertical_los` | 14 | **4.698** | 2.835 | 0.035 | 2.560 | 4.990 |
| `open_sparse_vertical_nlos` | 9 | **2.947** | 2.477 | 0.364 | 3.057 | 3.286 |
| `mixed_compact_lowrise_los` | 14 | **4.041** | 2.742 | 0.063 | 2.139 | 4.331 |

So the reversal is not subtle in the current partial run set:

- `open_sparse_lowrise`: LoS is worse by about **+1.97 dB**
- `open_sparse_vertical`: LoS is worse by about **+1.75 dB**

## Why this happens

The old ordering was strongly influenced by a **model failure on NLoS**, not
just by intrinsic task difficulty.

From the Try76 design study:

- old NLoS predictions had severe mode collapse
- the new Try76 architecture was designed specifically to fix that
- the new distribution-first head, KL term, and mixture NLL directly target the
  NLoS marginal-distribution problem

The actual study numbers are the key:

| aggregated group | target mean / std | old prediction mean / std | interpretation |
|---|---|---|---|
| `path_loss / target_los` | `97.0 / 5.3` | `96.4 / 3.8` | LoS was already fairly reasonable |
| `path_loss / target_nlos` | `107.1 / 4.3` | `36.5 / 22.4` | NLoS was catastrophically collapsed |

The design note says this explicitly: predicting a uniform **110 dB** on NLoS
pixels would beat the old Try75 NLoS predictions.

Once that failure is fixed, NLoS can become easier than LoS inside a topology
expert because:

- NLoS within an expert is often fairly concentrated around a narrow bulk
- LoS has stronger height drift
- LoS therefore needs more accurate conditioning and spatial placement

Again, the numeric evidence supports that:

- the histogram study describes `path_loss | target_los` as roughly `97 dB ± 5.3`
- it describes `path_loss | target_nlos` as roughly `107 dB ± 4.3`
- in `distribution_classes.md`, `path_loss | target_nlos` is described as a
  **narrow peak around 107 dB**

So after the model starts fitting the right marginal family, NLoS is no longer
the wildly unstable target it used to be in the older branches.

LoS also has a stronger dependence on UAV height:

- LoS mean drift across height buckets is reported as about **+5 to +11 dB**
- NLoS mean drift is much flatter, about **±1 dB**

This matters because it means the LoS expert has to learn a stronger
height-conditional shift than the NLoS expert.

So the new ordering does **not** mean LoS suddenly became intrinsically harder
than NLoS in a universal sense. It means the Try76 family is much better aligned
with the NLoS failure mode than the old families were.

## Practical recommendation

For the current branch, the evidence suggests:

- use the **Try76 distribution-first model** for **NLoS**
- revert to an **older LoS model family** for **LoS**

In other words:

- **NLoS** benefits from the new distribution-first formulation
- **LoS** may still be better served by the older prior-heavy / older-head
  architecture until the Try76 LoS branch is improved

## Interpretation

This should be treated as a **hybrid regime recommendation**, not as a claim
that Try76 is globally better or worse.

Recommended current policy:

- **LoS:** use the older LoS-specialist model family
- **NLoS:** use Try76-style distribution-first experts

With the current numbers, the practical reading is:

- for `open_sparse_lowrise`, the Try76 NLoS expert at **2.298 dB** is already in
  a very different quality regime from the LoS expert at **4.266 dB**
- for `open_sparse_vertical`, the Try76 NLoS expert at **2.947 dB** still beats
  the LoS expert at **4.698 dB** by a large margin

That is enough evidence to justify a regime-wise split in model choice.

## Caveat

The clamp discussion should now be separated into two parts:

1. what the old completed runs used
2. what the branch should use going forward

The currently reported completed runs in this note were trained with the older
resolved clamp range:

- `clamp_lo = 60.0`
- `clamp_hi = 125.0`

After reviewing the GT histogram support in
`tmp_review/histograms_try74/histograms.csv`, the branch defaults were widened
to:

- `clamp_lo = 30.0`
- `clamp_hi = 178.0`

That wider range is more consistent with the observed GT support:

- overall `path_loss|target_los`: **67 .. 170**
- overall `path_loss|target_nlos`: **74 .. 168**
- `open_sparse_lowrise target_nlos`: **79 .. 168**
- `open_sparse_vertical target_nlos`: **80 .. 154**

So for future retraining, the new `30 .. 178` clamp is the intended setting.

However, this still does **not** change the interpretation of the current
results in the note:

- the LoS/NLoS reversal is already visible in runs trained with the old clamp
- that reversal is still more plausibly explained by the architectural fix to
  the old NLoS collapse and by stronger LoS height drift

In short:

- **for old finished runs:** remember they used `60 .. 125`
- **for future runs:** use `30 .. 178`
- **for interpretation:** the main story is still architectural, not just clamp choice
