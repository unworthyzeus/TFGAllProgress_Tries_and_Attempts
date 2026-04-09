# Try 49 Failed Experiments

This file tracks the `Try 49` artifacts that were moved out of the active path
so the current branch stays easier to follow.

## Active path

- Prior calibration kept active:
  - `prior_calibration/regime_obstruction_train_only_from_try47.json`
- Stage 1 configs kept active:
  - `experiments/fortyninthtry49_pmnet_prior_gan_fastbatch/fortyninthtry49_pmnet_prior_stage1_widen112_initial_mae_dominant.yaml`
  - `experiments/fortyninthtry49_pmnet_prior_gan_fastbatch/fortyninthtry49_pmnet_prior_stage1_widen112_resume_mae_dominant.yaml`
- Stage 2 config kept active:
  - `experiments/fortyninthtry49_pmnet_tail_refiner_fastbatch/fortyninthtry49_pmnet_tail_refiner_stage2.yaml`

## Archived prior experiments

Moved to `failed_experiments/prior_calibration/`:

- `regime_obstruction_train_only_deepshadow_directml_quick.*`
- `regime_shadowed_ripple_train_only_directml_quick.*`
- `regime_structural_train_only_*`

These correspond to earlier prior redesign attempts that were explored but are
not the prior used by the current `Try 49` training path.

## Archived config variants

Moved to `failed_experiments/configs/`:

- `fortyninthtry49_pmnet_prior_stage1_widen112_initial.yaml`
- `fortyninthtry49_pmnet_prior_stage1_widen112_resume.yaml`

These are the older non-`mae_dominant` stage-1 configs. They are still useful
as references, but the current branch uses the newer robust-loss configs.
