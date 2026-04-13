# Try 63

`Try 63` is the coarse-to-fine follow-up to `Try 62`.

The main change is to make the coarse stage genuinely cheap:

- `stage1` trains at `128x128`
- `stage2` still refines at full `513x513`
- the `stage2` teacher upsamples the `stage1` prediction back to full resolution

## Main Idea

Instead of asking the first network to solve the whole high-resolution problem,
we let it learn a coarse physically informed path-loss map quickly and cheaply.

Then the refiner works at high resolution and learns only the missing detail.

## Stage 1

- PMHHNet expert
- `image_size = 128`
- `base_channels = 32`
- formula prior enabled
- obstruction proxy channels enabled
- augmentation enabled
- direct supervised `path_loss` objective

## Stage 2

- full-resolution refiner at `513x513`
- `stage1` teacher runs at `128x128`
- teacher prediction is bilinearly upsampled before refinement
- refiner corrects the coarse teacher at full resolution

## Why

This is the version that matches the earlier idea:

1. solve an easier low-resolution problem first
2. upscale
3. refine detail only where it matters

## Key Files

- `train_partitioned_pathloss_expert.py`
- `train_pmnet_tail_refiner.py`
- `scripts/generate_try63_configs.py`
- `scripts/plot_try63_metrics.py`
- `cluster/run_sixtythirdtry63_4gpu.slurm`
- `cluster/submit_try63_stage1_stage2_4gpu.py`
