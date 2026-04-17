# Try 73

`Try 73` is the no-prior direct-prediction rerun of the stable `Try 66/68` line.

## Goal

Measure whether the expert PMHHNet family can learn `path_loss` directly, without inheriting the residual-learning bias or failure modes of the analytic prior.

## Design

- **6 topology experts**:
  - `open_sparse_lowrise`
  - `open_sparse_vertical`
  - `mixed_compact_lowrise`
  - `mixed_compact_midrise`
  - `dense_block_midrise`
  - `dense_block_highrise`
- **No prior**:
  - `data.path_loss_formula_input.enabled = false`
  - `data.path_loss_formula_input.include_confidence_channel = false`
  - `prior_residual_path_loss.enabled = false`
  - `prior_residual_path_loss.use_formula_input_channel = false`
- **Direct path-loss prediction**, not residual-over-prior prediction.
- **Building mask kept on**.
- **Per-expert output clamps removed** from the target metadata.
- **EMA safety guard** added so non-finite source weights do not poison validation EMA snapshots.

## Files

- Config generator:
  - `scripts/generate_try73_configs.py`
- Registry:
  - `experiments/seventythird_try73_experts/try73_expert_registry.yaml`
- 2-GPU chain:
  - `cluster/submit_try73_experts_2gpu_sequential.py`
  - `cluster/run_seventythird_try73_2gpu.slurm`
- 4-GPU chain:
  - `cluster/submit_try73_experts_4gpu_sequential.py`
  - `cluster/run_seventythird_try73_4gpu.slurm`
- Plotter:
  - `scripts/plot_try73_metrics.py`

## Notes

- This branch was opened after the first `Try 73` attempt showed that a prior-free run can become unstable if config drift accumulates on top of the no-prior change.
- The current generator intentionally keeps the mature `Try 66/68` data and training layout while removing the prior path.
- `Try 74` and `Try 75` continue the no-prior direction with a simpler 3-expert curriculum.
