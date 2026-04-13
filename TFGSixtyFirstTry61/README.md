# Try 61

`Try 61` keeps the `Try 60` no-prior setup but changes the training objective to attack the hard regime directly.

What changes versus `Try 60`:

- strong LoS/NLoS regime reweighting during training
- explicit `NLoS` loss term in the generator objective
- checkpoint selection based on a composite proxy:
  - `overall_rmse + alpha * nlos_rmse`
- `open_sparse_vertical` is split into 2 experts:
  - `open_sparse_vertical_los`
  - `open_sparse_vertical_nlos`

So the expert family is now 7 experts:

- `open_sparse_lowrise`
- `open_sparse_vertical_los`
- `open_sparse_vertical_nlos`
- `mixed_compact_lowrise`
- `mixed_compact_midrise`
- `dense_block_midrise`
- `dense_block_highrise`

## Main Goal

The point of this try is to stop rewarding the easy `LoS` majority so much and force optimization pressure onto the difficult `NLoS` pixels that are still dominating RMSE.

## Literature Note

See `TRY61_RESEARCH_SOURCES.md` for the papers, challenge results, and the current interpretation of why `Try 61` is not clearly improving despite the extra losses.

## Inputs

Each expert receives:

- `topology_map`
- `los_mask`
- `distance_map`
- `antenna_height_m` through FiLM scalar conditioning

There is still no formula-prior channel in this try.

## Outputs

Each expert outputs 2 channels:

- channel `0`: direct `path_loss`
- channel `1`: auxiliary `no_data` logit

## Losses

The training objective now combines:

- final reconstruction loss on `path_loss`
- multiscale path-loss loss
- auxiliary `no_data` BCE
- explicit `NLoS`-only RMSE term

And the training mask is reweighted with:

- `los_weight = 0.5`
- `nlos_weight = 4.0`

## Key Files

- `train_partitioned_pathloss_expert.py`
- `predict.py`
- `scripts/generate_try61_configs.py`
- `scripts/plot_try61_metrics.py`
- `cluster/run_sixtyfirsttry61_partitioned_expert_4gpu.slurm`
- `cluster/submit_try61_experts_4gpu.py`
