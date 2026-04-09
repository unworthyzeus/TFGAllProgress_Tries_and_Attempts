# Try 52

`Try 52` is the cleaned paper-routed branch that grows out of the `Try 51`
line. The goal of this folder is to keep a compact, city-type-aware branch
with consistent names and without stale `Try 51` paths mixed into configs,
cluster scripts, and outputs.

## What is active here

The current checked-in launch path is:

- supervised `stage1` residual regression with the calibrated prior;
- `city_holdout` validation instead of random splits;
- automatic `city_type` routing via density/height thresholds;
- a simpler `stage2` residual refiner on top of the `Try 52 stage1` teacher;
- and a prepared `stage3` that only refines `NLoS` using a small global-context
  model on top of `stage2`.

This stays closer to the common pattern in the literature:

- strong prior;
- supervised regression;
- geography-aware holdout;
- transfer from a previously trained branch;
- regime-aware weighting instead of memorizing city names.

Main references:

- [PATH_LOSS_MODEL_TRAINING_PAPERS.md](C:/TFG/TFGpractice/PATH_LOSS_MODEL_TRAINING_PAPERS.md)
- [TRY52_PAPER_BACKED_NEXT_STEPS.md](C:/TFG/TFGpractice/TRY52_PAPER_BACKED_NEXT_STEPS.md)

## Cleaned naming

The main configs are now:

- [fiftysecondtry52_pmnet_prior_stage1_w112_initial_paper_routed.yaml](C:/TFG/TFGpractice/TFGFiftySecondTry52/experiments/fiftysecondtry52_pmnet_prior_gan_fastbatch/fiftysecondtry52_pmnet_prior_stage1_w112_initial_paper_routed.yaml)
- [fiftysecondtry52_pmnet_prior_stage1_w112_resume_paper_routed.yaml](C:/TFG/TFGpractice/TFGFiftySecondTry52/experiments/fiftysecondtry52_pmnet_prior_gan_fastbatch/fiftysecondtry52_pmnet_prior_stage1_w112_resume_paper_routed.yaml)
- [fiftysecondtry52_pmnet_tail_refiner_stage2_paper_routed.yaml](C:/TFG/TFGpractice/TFGFiftySecondTry52/experiments/fiftysecondtry52_pmnet_tail_refiner_fastbatch/fiftysecondtry52_pmnet_tail_refiner_stage2_paper_routed.yaml)

The names now reflect what is actually in the files:

- `w112` for the current `stage1` width;
- `paper_routed` because the branch prefers morphology/city-type routing;
- no stale `t51`, `literature`, or misleading `small_*` launch names in the
  main files.

## Stage 1

Main script:

- [train_pmnet_prior_gan.py](C:/TFG/TFGpractice/TFGFiftySecondTry52/train_pmnet_prior_gan.py)

Current launch config:

- [fiftysecondtry52_pmnet_prior_stage1_w112_resume_paper_routed.yaml](C:/TFG/TFGpractice/TFGFiftySecondTry52/experiments/fiftysecondtry52_pmnet_prior_gan_fastbatch/fiftysecondtry52_pmnet_prior_stage1_w112_resume_paper_routed.yaml)

Current behavior:

- `CityTypeRoutedNLoSMoERegressor`
- `112` base channels
- one learned `NLoS` expert map per automatic city type:
  - `open_lowrise`
  - `mixed_midrise`
  - `dense_highrise`
- `GAN` disabled (`lambda_gan = 0.0`)
- `MSE`-dominant loss with a smaller `L1` term
- `city_holdout` split
- regime reweighting that boosts:
  - `NLoS`
  - low antenna
  - `dense_highrise`

Warm start:

- the init path adapts from the `Try 51` stage-1 checkpoint at epoch `36`
- the cleaned cluster output path is:
  - `outputs/fiftysecondtry52_pmnet_prior_stage1_moe112_paper_routed_t52_stage1_moe112_4gpu`

## Stage 2

Main script:

- [train_pmnet_tail_refiner.py](C:/TFG/TFGpractice/TFGFiftySecondTry52/train_pmnet_tail_refiner.py)

Current launch config:

- [fiftysecondtry52_pmnet_tail_refiner_stage2_paper_routed.yaml](C:/TFG/TFGpractice/TFGFiftySecondTry52/experiments/fiftysecondtry52_pmnet_tail_refiner_fastbatch/fiftysecondtry52_pmnet_tail_refiner_stage2_paper_routed.yaml)

Current behavior:

- frozen on-the-fly `Try 52 stage1` teacher
- `UNet` tail refiner
- `72` refiner base channels
- no gate branch
- no oversample / tail-focus / high-frequency extras in the launch config
- regime-aware weighting still favors the hard cases

The cleaned stage-2 output path is:

- `outputs/fiftysecondtry52_tail_refiner_stage2_teacher_moe112_4gpu`

## Stage 3

Prepared config:

- [fiftysecondtry52_pmnet_stage3_nlos_global_context.yaml](C:/TFG/TFGpractice/TFGFiftySecondTry52/experiments/fiftysecondtry52_pmnet_tail_refiner_fastbatch/fiftysecondtry52_pmnet_stage3_nlos_global_context.yaml)

Current idea:

- teacher = `stage2`
- apply correction only on `NLoS`
- keep `LoS` untouched
- use a small `GlobalContextUNetRefiner`
- rely on lightweight token-mixing/global context instead of a big model

This is the most explicitly paper-inspired refinement stage in the branch:

- cascaded residual refinement
- `NLoS` specialization
- extra global context for hard urban cases

Planned output path:

- `outputs/fiftysecondtry52_stage3_nlos_global_context_4gpu`

## Prior routing

Active calibration file:

- [regime_obstruction_train_only_from_try47.json](C:/TFG/TFGpractice/TFGFiftySecondTry52/prior_calibration/regime_obstruction_train_only_from_try47.json)

Important setting:

- `prefer_threshold_city_type: true`

This keeps the useful train-only calibrated prior, but routes it by inferred
morphology instead of relying mainly on a hard-coded city-name mapping.

## Cluster entry points

- [run_fiftysecondtry52_stage1_init_4gpu.slurm](C:/TFG/TFGpractice/TFGFiftySecondTry52/cluster/run_fiftysecondtry52_stage1_init_4gpu.slurm)
- [run_fiftysecondtry52_stage1_resume_4gpu.slurm](C:/TFG/TFGpractice/TFGFiftySecondTry52/cluster/run_fiftysecondtry52_stage1_resume_4gpu.slurm)
- [run_fiftysecondtry52_stage2_4gpu.slurm](C:/TFG/TFGpractice/TFGFiftySecondTry52/cluster/run_fiftysecondtry52_stage2_4gpu.slurm)
- [run_fiftysecondtry52_stage3_4gpu.slurm](C:/TFG/TFGpractice/TFGFiftySecondTry52/cluster/run_fiftysecondtry52_stage3_4gpu.slurm)
- [run_fiftysecondtry52_cleanup_sert2001_1gpu.slurm](C:/TFG/TFGpractice/TFGFiftySecondTry52/cluster/run_fiftysecondtry52_cleanup_sert2001_1gpu.slurm)
- [submit_try52_chain.py](C:/TFG/TFGpractice/TFGFiftySecondTry52/cluster/submit_try52_chain.py)

## Notes

- This branch is cleaned and consistently named, but it is still experimental.
- The target is still much lower RMSE, especially in `NLoS`.
- The cleanup here is meant to make the next iterations less error-prone and
  easier to relaunch on the cluster.
