# Tries 24-28

This document summarizes what has been implemented for the post-`Try 22` / post-`Try 23` family.

## What is treated as a common baseline when it makes sense

- bilinear decoder instead of `ConvTranspose2d`
- `group norm` with `batch_size=1`
- `distance_map_channel: true`
- `antenna_height_m` as scalar conditioning
- `lambda_gan: 0.0`
- `250 epochs`
- `early_stopping.enabled: false`

The idea is to avoid spending whole tries on changes that are already clearly common and reasonable for this family.

## What should still open a new try

- real architecture changes
- real target changes
- real loss changes
- real target-combination changes

## Try 24

- Folder: `TFGTwentyFourthTry24`
- Status: local only, not uploaded
- Targets: `path_loss + delay_spread + angular_spread`
- Hypothesis:
  - a shared encoder may exploit the global structure of `path_loss` to help `delay/angular`
- Risk:
  - `path_loss` may dominate training and suppress the other two outputs

## Try 25

- Folder: `TFGTwentyFifthTry25`
- Status: launched first in cluster with `1 GPU`, later removed from remote scratch
- Target: `path_loss`
- New change:
  - lightweight attention in the bottleneck
- Hypothesis:
  - improve global context without redesigning the whole model

## Try 26

- Folder: `TFGTwentySixthTry26`
- Status: active cluster candidate, now prepared and launched in `2 GPU`
- Targets: `delay_spread + angular_spread`
- New change:
  - spatial gradient loss
- Hypothesis:
  - reduce over-smoothed maps and improve transitions

## Try 27

- Folder: `TFGTwentySeventhTry27`
- Status: implemented as a standalone idea, not kept as the active branch
- Target: `path_loss`
- New change:
  - topology-edge-guided regularization
- Hypothesis:
  - penalize errors more strongly where physically meaningful map failures often appear

## Try 28

- Folder: `TFGTwentyEighthTry28`
- Status: active cluster branch in `2 GPU`
- Target: `path_loss`
- New change:
  - combine `Try 25` and `Try 27`
  - lightweight bottleneck attention
  - topology-edge-weighted path-loss regularization
- Hypothesis:
  - global context and physics-aware local regularization may be complementary rather than redundant
- Risk:
  - combining two good ideas does not guarantee a better result;
  - optimization may become slower or less stable;
  - the added regularization may partially interfere with the benefit of attention.
