# Try 75

`Try 75` is the second stage of the new 3-expert no-prior curriculum.

## Goal

Start from the near-fixed-height representation learned in `Try 74`, reopen the full antenna-height distribution, and let FiLM learn the height-dependent deformation of the mapping.

## Design

- **Same 3 experts as Try 74**:
  - `open_lowrise`
  - `mixed_midrise`
  - `dense_highrise`
- **No prior**:
  - `data.path_loss_formula_input.enabled = false`
  - `data.path_loss_formula_input.include_confidence_channel = false`
  - `prior_residual_path_loss.enabled = false`
- **All heights enabled again**:
  - `partition_filter` uses only `city_type`
- **Height conditioning restored**:
  - `scalar_feature_columns = ["antenna_height_m"]`
  - `model.use_scalar_channels = true`
  - `model.use_scalar_film = true`
- **Default continuation from Try 74**:
  - each expert `runtime.resume_checkpoint` points to the matching `Try 74` best checkpoint
  - the registry checkpoint field also points to the matching `Try 74` checkpoint

## Files

- Config generator:
  - `scripts/generate_try75_configs.py`
- Registry:
  - `experiments/seventyfifth_try75_experts/try75_expert_registry.yaml`
- 4-GPU chain:
  - `cluster/submit_try75_experts_4gpu_sequential.py`
  - `cluster/run_seventyfifth_try75_4gpu.slurm`
- Plotters:
  - `scripts/plot_try75_metrics.py`
  - `scripts/plot_try73_metrics.py`

## Notes

- `Try 75` is meant to answer a clean question:
  - can a model first learn the near-fixed-height morphology mapping,
  - and then learn height generalization more stably as a second stage?
- The current scaffold keeps the FiLM mechanism simple on purpose.
- If this curriculum works, the next ablation is to compare the current sinusoidal height embedding against a stronger conditioner rather than changing both the curriculum and the embedding at once.
