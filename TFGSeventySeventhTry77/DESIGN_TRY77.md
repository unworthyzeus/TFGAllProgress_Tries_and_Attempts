# Try 77 - Distribution-first spread experts

Try 77 extends the distribution-first idea from Try 76 to **spread prediction only**:

- `delay_spread`
- `angular_spread`

The experiment is intentionally split into **12 experts**:

- 6 topology classes
- 2 target metrics per class: `{delay_spread, angular_spread}`

The goal is to test whether a topology-partitioned expert family can model the
very non-Gaussian spread targets better than a plain regression head.

## 1. Scope

Try 77 predicts spread maps in their **native units**:

- `delay_spread` in nanoseconds
- `angular_spread` in degrees

The architecture and losses are implemented from scratch inside this try:

- `src/model_try77.py`
- `src/losses_try77.py`
- `src/config_try77.py`
- `src/metrics_try77.py`
- `src/data_utils.py`

The only thing conceptually inherited from earlier tries is the **split contract**:

- `city_holdout`
- `val_ratio = 0.15`
- `test_ratio = 0.15`
- `split_seed = 42`

This is verified explicitly by:

- `tests/test_split_matches_try75.py`

## 2. Expert structure

Experts are defined over the same 6 topology classes used in Try 76:

- `open_sparse_lowrise`
- `open_sparse_vertical`
- `mixed_compact_lowrise`
- `mixed_compact_midrise`
- `dense_block_midrise`
- `dense_block_highrise`

For each topology class, Try 77 creates:

- one `delay_spread` expert
- one `angular_spread` expert

Registry:

- `experiments/seventyseventh_try77_experts/try77_expert_registry.yaml`

## 3. Model summary

Try 77 keeps the same high-level philosophy as Try 76, adapted to spread targets:

1. **Stage A** predicts a compact global distribution for the sample.
2. **Stage B** reconstructs a per-pixel map conditioned on that distribution.

The model uses:

- a spike-plus-mixture parameterization
- FiLM height conditioning
- a U-Net style decoder
- GroupNorm throughout

Why this design:

- delay and angular spread histograms are highly skewed
- they often contain a narrow spike near zero plus a heavy tail
- a direct MSE-style regressor tends to wash out the rare tail pixels

## 4. Losses

Try 77 optimizes a weighted combination of:

- `map_nll`
- `dist_kl`
- `moment_match`
- `outlier_budget`
- `rmse`
- `mae`

All losses are masked by:

- ground pixels only
- finite and non-negative target pixels only

This matches the implementation in `src/losses_try77.py`.

## 5. Input and masking

Per-sample inputs are built from:

- normalized `topology_map`
- `los_mask`
- complementary `nlos_mask`
- explicit `ground_mask`
- scalar UAV height embedding

Targets:

- `delay_spread`
- `angular_spread`

Invalid targets are masked out if they are:

- non-finite
- negative

The current Try 77 implementation therefore assumes **non-negative spread targets**.

## 6. Training and evaluation entry points

Training:

- `train_try77.py`

Evaluation:

- `evaluate_try77.py`

The trainer supports:

- single GPU
- CPU fallback
- `torchrun` / DDP launching via SLURM
- checkpoint resume
- `history.json`
- `best_model.pt`

Evaluation writes:

- `summary.json`
- `per_sample.json`
- `histograms.csv`

under `outputs/.../eval_val` or `outputs/.../eval_test`.

## 7. Cluster workflow

Cluster runtime helpers are in:

- `cluster/prepare_runtime_config.py`
- `cluster/run_seventyseventh_try77_1gpu.slurm`
- `cluster/run_seventyseventh_try77_2gpu.slurm`
- `cluster/run_seventyseventh_try77_cleanup_sert2001_1gpu.slurm`
- `cluster/submit_try77_experts_2gpu_sequential.py`

The sequential submitter is designed for:

- upload once
- train expert 1
- cleanup
- train expert 2
- cleanup
- ...

This mirrors the Try 76 chained submission style.

## 8. Experiment configs

All 12 experiment configs live in:

- `experiments/seventyseventh_try77_experts/`

Naming convention:

- `try77_expert_<topology>_delay_spread.yaml`
- `try77_expert_<topology>_angular_spread.yaml`

The default clamps are metric-specific:

- delay spread: `0 .. 400`
- angular spread: `0 .. 90`

These can be refined later if the histogram study suggests tighter expert-wise ranges.

## 9. Utilities

Training-curve plotting:

- `scripts/plot_history.py`

This reads `history.json` and renders `history_curves.png` next to each run.

## 10. Current status

Try 77 is now scaffolded as a complete experiment package:

- source modules
- trainer
- evaluator
- cluster scripts
- 12 YAML expert configs
- registry
- plotting utility
- split-consistency test

The remaining work is experimental rather than structural:

- train all 12 experts
- compare delay vs angular difficulty by topology
- inspect whether clamp ranges should be tightened per expert
- decide if a future Try 78 should add explicit physics priors back into the spread pipeline
