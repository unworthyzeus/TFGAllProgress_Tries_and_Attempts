# Implemented New Tries 20 to 23

This file records the path-loss follow-up tries that were actually implemented, what happened once they started to train, and why Try 23 was opened afterwards.

## What we implemented first

### Try 20

Folder:

- `TFGTwentiethTry20`

Goal:

- remove checkerboard-style decoder artifacts by replacing transposed convolutions with bilinear upsampling plus projection.

Main code change:

- `model_unet.py`: decoder upsampling changed through `upsample_mode: bilinear`.

Config:

- `experiments/twentiethtry20_cityregime/twentiethtry20_cityregime.yaml`

### Try 21

Folder:

- `TFGTwentyFirstTry21`

Goal:

- improve large-scale path-loss consistency with an explicit multiscale loss.

Main code change:

- `train_cgan.py`: added multiscale path-loss loss.

Config:

- `experiments/twentyfirsttry21_multiscale/twentyfirsttry21_multiscale.yaml`

### Try 22

Folder:

- `TFGTwentySecondTry22`

Goal:

- combine the decoder fix from Try 20 with the multiscale loss from Try 21.

Main code changes:

- `model_unet.py`: bilinear upsampling decoder.
- `train_cgan.py`: multiscale path-loss loss.

Config:

- `experiments/twentysecondtry22_decoder_multiscale/twentysecondtry22_decoder_multiscale.yaml`

## What happened once 20 to 22 were running

Early cluster comparison on the same HDF5 route showed:

- `Try 22` was the strongest path-loss run.
- `Try 21` was second.
- `Try 20` was clearly behind the other two.

Observed best validation trend before the next decision:

- `Try 20`: around `path_loss.rmse_physical ~= 17.40`
- `Try 21`: around `path_loss.rmse_physical ~= 17.06`
- `Try 22`: around `path_loss.rmse_physical ~= 16.72`

Decision taken from that ranking:

- keep `Try 22` running,
- cancel `Try 20` and `Try 21`,
- open a new line of work instead of spending all GPUs on three very similar path-loss jobs.

## Why Try 23 was opened

While reviewing the old `delay_spread` and `angular_spread` prediction panels in:

- `D:\Dataset_Imagenes\predictions_secondtry2_delay_angular`

the same failure pattern kept appearing:

- predictions were too smooth in regions where the GT has thin discontinuous structures,
- interiors of blocks were filled as broad soft blobs,
- weak grid-like texture was still visible in the predicted maps,
- postprocessing from older runs likely amplified smoothing instead of helping.

That made the next step more interesting than another path-loss-only variant.

### Try 23

Folder:

- `TFGTwentyThirdTry23`

Goal:

- revisit `delay_spread` and `angular_spread` with the two most useful lessons from the path-loss branch:
  - bilinear decoder instead of transpose-conv decoder,
  - multiscale regression loss, but now applied to the delay/angular targets.

Main code changes:

- `train_cgan.py`: generalized the multiscale loss into `compute_multiscale_regression_loss(...)`.
- `model_unet.py` and `model_cgan.py`: inherited the bilinear decoder path from Try 22.

Config:

- `experiments/twentythirdtry23_delay_angular_multiscale/twentythirdtry23_delay_angular_multiscale.yaml`

Key design choices:

- targets are only `delay_spread` and `angular_spread`,
- `lambda_gan: 0.0` to avoid reintroducing fake texture,
- `regression_median_kernel: 1` to avoid extra blur in postprocess,
- checkpoint selection uses `delay_spread.rmse_physical` and `angular_spread.rmse_physical`,
- distance map and antenna height conditioning are kept because they are cheap and physically meaningful cues.

Cluster launch file:

- `cluster/run_twentythirdtry23_delay_angular_multiscale_2gpu.slurm`

Submission helper:

- `python cluster/upload_and_submit_experiments.py --preset twentythird --gpus 2 --skip-datasets`

## Relaunch to 250 epochs

After the first comparison, the cluster plan was updated again:

- current jobs `20`, `21`, and `22` were cancelled,
- remote cluster folders for `Try 20` and `Try 21` were deleted to free scratch space,
- `Try 22` was relaunched in `2 GPU`, `2 days`, `250 epochs`, with `early_stopping.enabled: false`,
- `Try 23` was launched in `2 GPU`, `2 days`, `250 epochs`, with `early_stopping.enabled: false`.

Cluster state after relaunch:

- `Try 22` keeps the path-loss line alive as the strongest of `20/21/22`.
- `Try 23` opens the new delay/angular line under the same runtime budget.
