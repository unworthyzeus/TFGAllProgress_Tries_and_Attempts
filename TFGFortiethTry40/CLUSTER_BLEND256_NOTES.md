# Cluster Blend 256 Notes

## Status

This note reflects the earlier planning stage.

For the **real confirmed cluster workflow**, use:

- `COMPUTE_REAL_CLUSTER_GUIDE.md`

Important:

- the real cluster maximum confirmed for this setup was `6` GPUs, not `8`
- the real partition/account/QOS details are documented in `COMPUTE_REAL_CLUSTER_GUIDE.md`

## Goal

Prepare a cluster run for `TFGThirdTry3` using:

- hybrid path-loss model
- soft confidence fallback (`blend`)
- very wide generator (`base_channels: 256`)
- 8 GPUs on the UPC cluster

## Files Prepared

- Config: `configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max256_blend.yaml`
- 8-GPU launcher: `cluster/run_train_cgan_hdf5_8gpu.slurm`

## What Changed

### New CUDA config

`configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max256_blend.yaml` was created with:

- `runtime.device: cuda`
- `model.base_channels: 256`
- `path_loss_hybrid.fallback_mode: blend`
- `runtime.output_dir: outputs/cgan_unet_hdf5_pathloss_hybrid_cuda_max256_blend`
- `training.batch_size: 1`
- `training.auto_batch_by_vram.enabled: true`
- AMP enabled

Notes:

- `disc_base_channels` was left at `96` instead of scaling to `256`.
- This keeps the generator large while avoiding an unnecessarily huge discriminator.
- Auto batch sizing is conservative because a 256-channel model can be heavy.

### New 8-GPU SLURM script

`cluster/run_train_cgan_hdf5_8gpu.slurm` was created to:

- request `8` GPUs
- use partition `big_gpu`
- request `32` CPUs
- request `128G` RAM
- run for `48:00:00`
- launch one independent training process per visible GPU
- create one runtime config per GPU with:
  - output suffix `_gpu0`, `_gpu1`, ...
  - seed offsets `0..7`

Important:

- This is **not DDP**.
- It is a multi-run sweep inside one 8-GPU allocation.
- Each GPU trains its own independent run.

## What Was Confirmed

### SSH access

Confirmed cluster host:

- `ssh gmoreno@sert.ac.upc.edu`

What happened from this machine:

- host key was accepted successfully
- password authentication works when using a programmable client
- plain non-interactive `ssh` from the local shell initially failed because this environment cannot answer password prompts directly

### Remote home

Confirmed current remote home directory:

- `/homeB/g/gmoreno`

Observed host during login:

- `sert-entry-3`

### Old paths that appear stale

The project currently contains older cluster defaults such as:

- `/scratch/nas/g/gmoreno/tf_env/bin/activate`

But during remote probing, these old paths did **not** resolve as expected:

- `/scratch/nas/g/gmoreno`
- `/scratch/1/gmoreno`
- `/home/gmoreno`

So the cluster scripts may still contain stale path defaults copied from earlier notes.

## What Was Not Yet Confirmed

Before launching for real, these still need verification on the remote side:

- exact remote path of the synced project containing `TFGpractice/TFGThirdTry3`
- exact remote virtual environment path to activate
- whether the dataset file `CKM_Dataset.h5` is already present in that remote project location

## Recommended Remote Checks

Once logged into the cluster, verify:

```bash
pwd
echo $HOME
find ~ -maxdepth 4 -type d -name TFGThirdTry3 2>/dev/null
find ~ -maxdepth 5 -name CKM_Dataset.h5 2>/dev/null
```

Then verify the environment:

```bash
find ~ -maxdepth 5 -path '*bin/activate' 2>/dev/null | grep tf_env
```

If the old venv path is wrong, override it when submitting:

```bash
sbatch --export=ALL,CONFIG_PATH=configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max256_blend.yaml,VENV_PATH=/real/path/to/tf_env/bin/activate cluster/run_train_cgan_hdf5_8gpu.slurm
```

## Recommended Launch Command

If the project is already synced remotely and the venv path is correct:

```bash
cd <remote-project-root>/TFGpractice/TFGThirdTry3
sbatch cluster/run_train_cgan_hdf5_8gpu.slurm
```

Explicit version:

```bash
cd <remote-project-root>/TFGpractice/TFGThirdTry3
sbatch --export=ALL,CONFIG_PATH=configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max256_blend.yaml cluster/run_train_cgan_hdf5_8gpu.slurm
```

## Why `blend` Was Chosen

The earlier local hybrid run with hard fallback (`replace`) improved only briefly, then became unstable.

Observed pattern:

- best early epoch around `21.47 dB` RMSE
- later epochs drifted back upward
- confidence metrics and final fused RMSE did not move together

Main reason:

- training uses a binary confidence target derived from current error
- evaluation with `replace` performs a hard switch at the confidence threshold
- small confidence changes can flip whole pixels from DL prediction to heuristic prior

`blend` should make the fusion smoother and reduce this instability.

## Secret Handling Note

The cluster password is **not repeated in this file**.

It was intentionally omitted here because you said it is already documented elsewhere in another markdown file.
