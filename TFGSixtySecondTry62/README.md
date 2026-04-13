# Try 62

`Try 62` is the paper-like reset after `Try 61`.

The main idea is to move back toward what the strongest public radio-map papers
are actually doing:

- a simpler supervised `stage1`
- a `stage2` refiner
- stronger physically meaningful inputs
- fewer manual auxiliary losses

## What changes versus Try 61

- back to `6` topology experts instead of splitting `open_sparse_vertical`
- restore the formula-prior input channel
- enable obstruction proxy channels:
  - `shadow_depth`
  - `distance_since_los_break`
  - `max_blocker_height`
  - `blocker_count`
- remove `no_data` as a meaningful objective
- remove explicit `NLoS` focus loss
- remove manual regime reweighting
- keep `stage1` close to direct `path_loss` RMSE
- use `stage2` as the main coarse-to-fine refinement step

## Stage 1

`stage1` is intentionally simple:

- PMHHNet expert
- formula prior enabled
- obstruction features enabled
- augmentation enabled
- `generator_objective = full_map_rmse_only`
- selection by `path_loss.rmse_physical`

This is meant to stay closer to the direct target we care about instead of
spending capacity on several weighted side objectives.

## Stage 2

`stage2` uses the tail refiner again:

- on-the-fly `stage1` teacher
- refiner predicts residual correction over the `stage1` output
- small high-frequency term enabled
- no gate, no oversample, no manual regime weighting

So the refinement signal is still there, but the objective remains compact.

## Main Goal

The point of `Try 62` is to test whether a cleaner coarse-to-fine pipeline with
better physical inputs generalizes better than the more heavily hand-shaped
objective from `Try 61`.

## Key Files

- `train_partitioned_pathloss_expert.py`
- `train_pmnet_tail_refiner.py`
- `scripts/generate_try62_configs.py`
- `cluster/run_sixtysecondtry62_4gpu.slurm`
- `cluster/submit_try62_stage1_stage2_4gpu.py`
