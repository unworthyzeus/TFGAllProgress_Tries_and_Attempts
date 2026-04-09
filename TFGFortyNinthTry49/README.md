# Try 49

`Try 49` is the current two-stage PMNet branch built on top of the usable
formula prior copied from the `Try 47` family.

The repo has also been cleaned so the active path stays visible:

- active files stay in `prior_calibration/` and `experiments/`
- older prior trials and superseded configs were moved to
  `failed_experiments/`

## Current structure

- `stage1`: PMNet prior+residual generator
- `stage2`: tail refiner trained on top of a frozen `stage1` teacher

The main practical path is now:

- `stage1 init`
- `stage1 resume`
- `stage2`

The old `export -> stage2` route still exists in code, but it is no longer the
preferred path because large HDF5 exports were hitting quota and operational
friction.

## Stage 1

Main training script:

- `train_pmnet_prior_gan.py`

New robust-loss branch:

- `experiments/fortyninthtry49_pmnet_prior_gan_fastbatch/fortyninthtry49_pmnet_prior_stage1_widen112_initial_mae_dominant.yaml`
- `experiments/fortyninthtry49_pmnet_prior_gan_fastbatch/fortyninthtry49_pmnet_prior_stage1_widen112_resume_mae_dominant.yaml`

That new branch keeps the same architecture, but shifts the loss to an
`MAE`-dominant setting:

- `mse_weight = 0.25`
- `l1_weight = 1.0`

Best observed stage-1 validation result so far:

- widened `112`-channel branch: about `18.96 dB` RMSE

Superseded non-`mae_dominant` stage-1 configs were moved to:

- `failed_experiments/configs/`

## Stage 2

Main training script:

- `train_pmnet_tail_refiner.py`

Current config:

- `experiments/fortyninthtry49_pmnet_tail_refiner_fastbatch/fortyninthtry49_pmnet_tail_refiner_stage2.yaml`

Recent stage-2 changes:

- no mandatory intermediate HDF5 export;
- frozen `stage1` teacher runs on-the-fly inside each batch;
- `oversample` was disabled because startup cost was too high;
- validation now uses `DistributedSampler`;
- validation uses `teacher.predict_only()` instead of computing unnecessary
  stage-1 error maps;
- `val_batch_size` stays at `1` because `2` produced `OOM`;
- refiner width was reduced from `96` to `84` channels;
- high-frequency loss was added:
  - stronger Laplacian term
  - smaller gradient term

Current high-frequency settings:

- `laplacian_weight = 0.06`
- `gradient_weight = 0.02`

Important operational detail:

- `stage2` keeps reusing the same output folder;
- it is not being split into a separate branch per relaunch.

## Checkpoint adaptation

The old `96`-channel stage-2 checkpoint can be adapted to the new `84`-channel
refiner with:

- `scripts/widen_pmnet_checkpoint.py`

That script now supports both:

- `stage1`
- `tail_refiner`

There is also a dedicated cluster entry point for the one-off shrink step:

- `cluster/run_fortyninthtry49_tail_refiner_stage2_shrink_84_1gpu.slurm`

## Cluster entry points

- `cluster/run_fortyninthtry49_pmnet_prior_stage1_4gpu.slurm`
- `cluster/run_fortyninthtry49_pmnet_prior_stage1_resume_4gpu.slurm`
- `cluster/run_fortyninthtry49_tail_refiner_stage2_4gpu.slurm`
- `cluster/run_fortyninthtry49_tail_refiner_stage2_shrink_84_1gpu.slurm`

Helper for uploading `Try 49` without cleaning outputs and submitting the
current shrink/`stage2`/new-`stage1` chain:

- `cluster/submit_try49_mae_stage_chain.py`

## Prior calibration

Active prior calibration file:

- `prior_calibration/regime_obstruction_train_only_from_try47.json`

Archived prior calibration attempts:

- `failed_experiments/prior_calibration/`

Those archived files include the structural, shadowed-ripple, and deep-shadow
calibrations that were explored but are not part of the current training path.

## Current practical lessons

- `stage1` is the strongest branch so far.
- `stage2` does improve the architecture story, but its validation path is much
  more expensive because it still runs:
  - frozen `stage1 teacher`
  - refiner forward
  - metric breakdowns
- The biggest operational failures recently were not modeling errors, but:
  - stale GPU processes on the cluster node
  - overly expensive validation
  - and bad dependency / checkpoint plumbing

Diagram:

- [try49_full_system.mmd](C:/TFG/TFGpractice/diagram/try49/try49_full_system.mmd)
