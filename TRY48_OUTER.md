# Try 48 Outer Notes

Date: 2026-04-03

## Goal
- Run Try 48 as a separated 2-network pipeline:
	- Stage 1: PMNet base model for residual correction against the prior.
	- Stage 2: frozen PMNet base + UNet refiner trained with adversarial detail pressure.
	- Inference: prior + base residual + refiner residual.

## Current Training Setup (2026-04-03)
- Stage 1 config:
	- Residual-focused PMNet training (`optimize_residual_only: true`, `lambda_gan: 0.0`).
	- Output folder: `outputs/fortyeighthtry48_pmnet_prior_stage1_t48_stage1_sep_base_4gpu`.
- Stage 2 config:
	- Separated refiner mode enabled (`separated_refiner.enabled: true`).
	- Frozen base checkpoint is loaded from Stage 1 best checkpoint.
	- Refiner architecture is UNet (`separated_refiner.refiner_arch: unet`).
	- GAN enabled for refinement (`lambda_gan: 0.015`).
	- Output folder: `outputs/fortyeighthtry48_pmnet_prior_stage2_t48_stage2_sep_4gpu`.

## Active Cluster Chain (Latest)
- Job `10019343`: Stage 1 (running).
- Job `10019344`: Stage 2 (pending with dependency `afterany:10019343`).
- This preserves the 4-hour slot constraint while keeping the 2-stage workflow automated.

## Scalar Antenna Height Clarification
- `scalar_feature_columns: [antenna_height_m]` defines the feature the model expects.
- `hdf5_scalar_specs` maps that feature to HDF5 source `uav_height`.
- `constant_scalar_features: {}` means no duplicate constant value is injected.
- This is a single feature with an explicit source mapping, not a duplicate feature.

## Loader/Runtime Changes
- Data loader workers increased earlier for throughput.
- `persistent_workers` disabled per preference.
- `prefetch_factor` now set to `8` per request.

## Cluster Resources (Current 4-GPU run)
- 4x RTX2080 GPUs.
- CPUs per task: 32.
- Memory: 120G.
- Time limit per job: 4 hours.

## Operational Notes
- Each stage writes to a different output folder to avoid mixing checkpoints/metrics when downloading.
- Stage 2 starts only after Stage 1 exits, so the base checkpoint exists before refiner training begins.
- Throughput should be re-checked after each relaunch because early log windows can be noisy.

## 4-Hour Limit Bypass
- Cluster imposes a 4-hour time limit for the 4-GPU partition.
- For chained runs, submission uses dependency scheduling:
  - Job A: normal `sbatch`.
  - Job B: `sbatch --dependency=afterany:<jobA_id> ...`.
- Current Stage 1 -> Stage 2 chain follows this dependency pattern (`10019343` -> `10019344`).

## Duration Estimate (Current Stage 1, 4 GPUs)
- Current train speed snapshot: around `1.14 s/it` at `140/2832` iterations.
- Approximate train-only epoch time:
	- `2832 * 1.14 s` ≈ `3228 s` ≈ `53.8 min`.
- With validation and checkpoint overhead, practical epoch time is roughly `60-75 min`.
- Approximate coverage per 4-hour job:
	- Around `3-4` epochs (environment-dependent).
