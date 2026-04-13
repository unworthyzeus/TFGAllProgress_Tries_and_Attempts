# Try 61 Research Sources and Current Reading

Date: `2026-04-12`

This note collects the main papers and result pages that are most relevant to
the current `Try 61` direction.

## Why this note exists

`Try 61` adds:

- stronger `LoS/NLoS` reweighting
- an explicit `NLoS` loss
- a composite checkpoint-selection proxy
- a split of `open_sparse_vertical` into `los` and `nlos` experts

However, the current curves do not show a clear breakthrough. The practical
pattern is:

- `overall val RMSE` improves early and then flattens
- `LoS RMSE` improves much more than `NLoS RMSE`
- the auxiliary `no_data` head reaches high `accuracy/IoU`, so it does not
  look like the main bottleneck

The goal of this note is to compare that behavior against what the literature
and challenge winners are actually doing.

## Main Sources

### 1. The First Indoor Pathloss Radio Map Prediction Challenge

- Overview paper:
  - https://arxiv.org/abs/2501.13698
- Official challenge website:
  - https://indoorradiomapchallenge.github.io/
- Official results page:
  - https://indoorradiomapchallenge.github.io/results.html

Why it matters:

- this is the closest benchmark to our current indoor path-loss setting
- it tells us what the strongest public methods were optimizing for
- it gives a realistic target scale for RMSE

Relevant reading for `Try 61`:

- the benchmark is centered on direct path-loss map quality
- the top public scores are much lower than our current expert-level results
- that makes it useful as a sanity check against overly complex objectives

### 2. TransPathNet: A Novel Two-Stage Framework for Indoor Radio Map Prediction

- Paper:
  - https://arxiv.org/abs/2501.16023
- Project page:
  - https://lixin.ai/TransPathNet/

Why it matters:

- it is one of the strongest public methods from the ICASSP 2025 challenge
- it uses a clear two-stage `coarse-to-fine` design instead of many manual
  auxiliary losses

Relevant reading for `Try 61`:

- the main signal here is architectural and stage-based, not "add more loss
  terms"
- the paper supports revisiting a simpler `stage1 + refiner stage2` path
- the emphasis is on representation and refinement, not on manually forcing
  every failure mode through separate weighted objectives

### 3. IPP-Net

- Paper:
  - https://arxiv.org/abs/2501.06414

Why it matters:

- another high-performing challenge method
- useful as a second reference point so we do not overfit our reasoning to one
  winner only

Relevant reading for `Try 61`:

- it reinforces the idea that better structure and signal representation matter
  more than stacking several handcrafted losses
- it is another argument for keeping the optimization target close to the real
  radio-map metric

### 4. Vision Transformers for Efficient Indoor Pathloss Radio Map Prediction

- Paper:
  - https://www.mdpi.com/2079-9292/14/10/1905

Why it matters:

- directly studies indoor path-loss prediction
- explicitly evaluates augmentation, feature engineering, and architecture
  choices

Relevant reading for `Try 61`:

- extensive augmentation improves generalization
- feature engineering is especially important in low-data regimes
- manually cropping maps to reduce distribution shift did not help in their
  experiments
- this points more toward better data treatment and richer physical channels
  than toward adding more objective terms

### 5. Radio Map Prediction from Aerial Images and Application to Coverage Optimization

- Paper:
  - https://arxiv.org/abs/2410.17264

Why it matters:

- although it is not the same indoor setup, it is directly relevant to radio-map
  prediction strategy
- it is useful for understanding how much accuracy can come from better spatial
  representation rather than more elaborate losses

Relevant reading for `Try 61`:

- supports keeping the learning objective close to the final map-quality target
- supports investing effort in feature design and model structure first

### 6. Multi-Task Learning Using Uncertainty to Weigh Losses

- Paper:
  - https://arxiv.org/abs/1705.07115

Why it matters:

- if we want multitask behavior, this is one of the standard references for
  learning task weights instead of hand-tuning them

Relevant reading for `Try 61`:

- `Try 61` currently uses several manually weighted terms
- if we keep multitask or multi-objective training, the literature supports
  learned weighting more than repeated manual retuning

### 7. Joint Modeling of Received Power, Mean Delay, and Delay Spread for Wideband Radio Channels

- Paper:
  - https://arxiv.org/abs/2005.06808

Why it matters:

- it shows that received power, mean delay, and delay spread are correlated and
  should not be treated as fully independent variables

Relevant reading for `Try 61` and later tries:

- this supports a shared representation or multitask setup
- it does not support leaking the ground-truth target of one task into another
  task as an input during training
- if we want path loss, delay spread, and angular spread to help each other,
  the principled direction is shared latent structure, not target leakage

## Current Reading of Try 61 Against the Literature

My current interpretation is:

1. `Try 61` is not obviously failing because of lack of supervision.
   It already has plenty of loss terms.
2. `Try 61` looks more likely to be misaligned with the strongest public
   strategies.
3. The literature gives stronger support to:
   - simpler primary objectives
   - better physical and geometric inputs
   - stronger augmentation
   - two-stage coarse-to-fine refinement
   - optionally multitask learning with learned weighting
4. The literature gives weaker support to:
   - adding many manually weighted auxiliary losses
   - expecting a separate `no_data` objective to rescue the hard `NLoS` regime
   - trying to fix a hard expert mainly through more objective engineering

## What this suggests after Try 61

The most literature-aligned next direction would be:

- keep the main objective close to direct `path_loss` map quality
- reduce manual auxiliary losses
- return to a clean `stage1 + stage2 refiner` setup
- add richer physically meaningful channels if available
- use stronger but physics-safe augmentation
- if path loss, delay spread, and angular spread are learned together, prefer
  a shared encoder or uncertainty-weighted multitask training over feeding
  ground-truth side targets as inputs

## Short Conclusion

`Try 61` is valuable because it tells us that "more losses and stronger manual
reweighting" is not automatically giving better generalization. The papers
above suggest that the next gains are more likely to come from:

- better representation
- better physically meaningful inputs
- stronger augmentation
- cleaner stage design

than from adding even more handcrafted loss terms.
