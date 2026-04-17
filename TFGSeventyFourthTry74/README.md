# Try 74

`Try 74` is the first stage of the new 3-expert curriculum.

## Goal

Learn the morphology-to-path-loss mapping at an almost fixed transmitter height, before asking the model to generalize across the full altitude distribution.

## Design

- **3 experts**, reusing the city-type split from `Try 67`:
  - `open_lowrise`
  - `mixed_midrise`
  - `dense_highrise`
- **No prior**:
  - `data.path_loss_formula_input.enabled = false`
  - `data.path_loss_formula_input.include_confidence_channel = false`
  - `prior_residual_path_loss.enabled = false`
- **Tight height band** around 50 m:
  - `antenna_height_m_min = 47.5`
  - `antenna_height_m_max = 52.5`
- **No explicit height modulation**:
  - `scalar_feature_columns = []`
  - `model.use_scalar_channels = false`
  - `model.use_scalar_film = false`
  - `tx_depth_map_channel = false`
  - `elevation_angle_map_channel = false`
- **Building mask kept on** as an input/support channel.

## Files

- Config generator:
  - `scripts/generate_try74_configs.py`
- Registry:
  - `experiments/seventyfourth_try74_experts/try74_expert_registry.yaml`
- 2-GPU chain:
  - `cluster/submit_try74_experts_2gpu_sequential.py`
  - `cluster/run_seventyfourth_try74_2gpu.slurm`
- 4-GPU chain:
  - `cluster/submit_try74_experts_4gpu_sequential.py`
  - `cluster/run_seventyfourth_try74_4gpu.slurm`
- Plotters:
  - `scripts/plot_try74_metrics.py`
  - `scripts/plot_try73_metrics.py`

## Notes

- `Try 74` is intentionally **not** the final all-height model.
- The point is to pretrain a stable geometry/morphology mapper under near-constant height.
- `Try 75` is the continuation stage that reopens the full height range and turns FiLM conditioning back on.
