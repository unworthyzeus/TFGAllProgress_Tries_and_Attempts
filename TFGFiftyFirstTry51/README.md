# Try 51

`Try 51` is a literature-aligned continuation of the `Try 49` PMNet path-loss
line.

The main goal is not to add more ad-hoc tricks, but to make the training setup
look more like what strong path-loss and radio-map papers actually do:

- supervised dense regression;
- strong physics-guided prior;
- explicit `LoS / NLoS` awareness;
- geography-aware held-out validation;
- automatic morphology grouping by city type;
- residual correction instead of full replacement;
- transfer from an already trained branch instead of restarting from zero.

## Core idea

`Try 51` keeps the useful parts of `Try 49`, but changes the training logic in
two important ways:

1. it treats **automatic city type** as more important than memorizing a city
   name;
2. it adds **regime-aware loss reweighting** during training instead of only
   reporting regime-aware metrics afterwards.

In practice, this means:

- the prior calibration is forced to prefer `density + height -> city_type`
  thresholds;
- train / val / test are split with `city_holdout` instead of random mixing;
- no fixed `city -> city_type` lookup is required for the intended main path;
- `NLoS`, low-antenna samples, and dense-highrise morphology receive more loss
  weight.

## Why this branch exists

The literature review in:

- [PATH_LOSS_MODEL_TRAINING_PAPERS.md](C:/TFG/TFGpractice/PATH_LOSS_MODEL_TRAINING_PAPERS.md)

suggests that the most common serious training recipe is:

- supervised regression;
- ray-tracing or physics-rich prior supervision;
- transfer learning or warm start;
- geography-aware held-out validation;
- `MSE / NMSE`-style optimization first, optional robust terms second;
- and often explicit `LoS / NLoS` decomposition.

That is much closer to this branch than to a GAN-first pipeline.

## Stage 1

Main script:

- [train_pmnet_prior_gan.py](C:/TFG/TFGpractice/TFGFiftyFirstTry51/train_pmnet_prior_gan.py)

Main configs:

- [fiftyfirsttry51_pmnet_prior_stage1_widen112_initial_literature.yaml](C:/TFG/TFGpractice/TFGFiftyFirstTry51/experiments/fiftyfirsttry51_pmnet_prior_gan_fastbatch/fiftyfirsttry51_pmnet_prior_stage1_widen112_initial_literature.yaml)
- [fiftyfirsttry51_pmnet_prior_stage1_widen112_resume_literature.yaml](C:/TFG/TFGpractice/TFGFiftyFirstTry51/experiments/fiftyfirsttry51_pmnet_prior_gan_fastbatch/fiftyfirsttry51_pmnet_prior_stage1_widen112_resume_literature.yaml)

Current stage-1 design:

- `PMNetResidualRegressor`
- `112` base channels
- supervised residual learning with `GAN` terms disabled
- `MSE`-dominant reconstruction:
  - `mse_weight = 1.0`
  - `l1_weight = 0.25`
- `Adam` optimizer with zero weight decay, closer to the common setups in the
  baseline papers
- multiscale path-loss supervision enabled
- prior-residual optimization enabled
- regime reweighting enabled

Regime reweighting in stage 1:

- `LoS` pixels keep base weight
- `NLoS` pixels get higher weight
- low-altitude antenna cases get an extra boost
- automatic city type changes the sample importance:
  - `open_lowrise`
  - `mixed_midrise`
  - `dense_highrise`

The branch is meant to warm-start from the best usable `Try 49` stage-1 model,
not from the discarded older `mae` branch.

Validation is deliberately stricter than in older tries:

- `data.split_mode = city_holdout`
- full cities are held out together for validation and test
- this is meant to reward morphology transfer instead of city memorization

## Automatic city type

This branch deliberately prefers **automatic morphology inference** over a
hard-coded `city name -> class` map.

The active intended path is:

- estimate non-ground density from topology;
- estimate mean obstacle height from topology;
- infer:
  - `open_lowrise`
  - `mixed_midrise`
  - `dense_highrise`

This is used both for:

- prior calibration routing;
- and training-time regime reweighting.

The key change is that:

- `Try 51` is meant to generalize to unseen cities that share morphology,
- not to memorize the historical calibration city names.

## Stage 2

Main script:

- [train_pmnet_tail_refiner.py](C:/TFG/TFGpractice/TFGFiftyFirstTry51/train_pmnet_tail_refiner.py)

Main config:

- [fiftyfirsttry51_pmnet_tail_refiner_stage2.yaml](C:/TFG/TFGpractice/TFGFiftyFirstTry51/experiments/fiftyfirsttry51_pmnet_tail_refiner_fastbatch/fiftyfirsttry51_pmnet_tail_refiner_stage2.yaml)

Current stage-2 design:

- frozen on-the-fly `stage1` teacher from `Try 51`;
- residual refiner with `84` base channels;
- no gate branch;
- no tail-focus heuristic weighting;
- no extra high-frequency objective;
- additional regime-aware reweighting for:
  - `NLoS`
  - low antenna
  - dense-highrise morphology

The intended role of `stage2` is not to relearn the whole map. It is a
weighted residual corrector for the hard tail left by `stage1`, kept
deliberately simpler than the more engineered `Try 49` versions.

## Prior path

Active calibration file:

- [regime_obstruction_train_only_from_try47.json](C:/TFG/TFGpractice/TFGFiftyFirstTry51/prior_calibration/regime_obstruction_train_only_from_try47.json)

Important setting:

- `prefer_threshold_city_type: true`

This keeps the useful train-only calibrated prior, but routes it through
automatic city-type thresholds instead of relying on explicit city-name mapping
as the main strategy.

## Supervised-only training

Even though the training script name still contains `gan`, the active `Try 51`
setup is now genuinely supervised:

- `loss.lambda_gan = 0.0`
- the script does not build or optimize a discriminator in that case
- checkpoints therefore stay lighter and the training path is closer to the
  supervised setups described in RadioUNet, PMNet and the challenge baselines

## Cluster entry points

- [run_fiftyfirsttry51_pmnet_prior_stage1_4gpu.slurm](C:/TFG/TFGpractice/TFGFiftyFirstTry51/cluster/run_fiftyfirsttry51_pmnet_prior_stage1_4gpu.slurm)
- [run_fiftyfirsttry51_pmnet_prior_stage1_resume_4gpu.slurm](C:/TFG/TFGpractice/TFGFiftyFirstTry51/cluster/run_fiftyfirsttry51_pmnet_prior_stage1_resume_4gpu.slurm)
- [run_fiftyfirsttry51_tail_refiner_stage2_4gpu.slurm](C:/TFG/TFGpractice/TFGFiftyFirstTry51/cluster/run_fiftyfirsttry51_tail_refiner_stage2_4gpu.slurm)

The stage-1 init job is designed to adapt from the best usable `Try 49` stage1
checkpoint.

## Target and realism

The branch target is:

- lower overall RMSE than the current active branches;
- stronger `NLoS` behavior than `Try 49/50`;
- and better cross-city generalization.

The requested target of `< 5 dB` overall RMSE is extremely ambitious for the
current dense `LoS + NLoS` dataset mix, so this branch should be treated as a
better-founded attempt, not as a guaranteed shortcut to that number.

## Related docs

- [TRY51_LITERATURE_ALIGNED_SUPERVISED_PLAN.md](C:/TFG/TFGpractice/TRY51_LITERATURE_ALIGNED_SUPERVISED_PLAN.md)
- [PRIOR_AGGREGATION_NOTE.md](C:/TFG/TFGpractice/TFGFiftyFirstTry51/PRIOR_AGGREGATION_NOTE.md)
- [PATH_LOSS_MODEL_TRAINING_PAPERS.md](C:/TFG/TFGpractice/PATH_LOSS_MODEL_TRAINING_PAPERS.md)
