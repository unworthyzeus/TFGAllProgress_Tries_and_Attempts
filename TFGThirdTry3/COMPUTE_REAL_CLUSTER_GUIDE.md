# Compute Real Cluster Guide

## Scope

This file records the **real, confirmed** cluster workflow used for `TFGThirdTry3`, replacing older assumptions that turned out to be stale.

The target experiment is:

- `TFGThirdTry3`
- hybrid path-loss model
- `blend` fallback
- `base_channels: 256`
- config: `configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max256_blend.yaml`

## Confirmed Connection

SSH access:

```bash
ssh gmoreno@sert.ac.upc.edu
```

Confirmed remote home:

```bash
/homeB/g/gmoreno
```

Observed login host:

```bash
sert-entry-3
```

Important:

- Host key acceptance worked normally.
- Password authentication works.
- Non-interactive plain `ssh` from the local shell was awkward, so programmable SSH/SFTP was used to automate remote work.

## Real Storage Layout

### Home

Confirmed home path:

```bash
/homeB/g/gmoreno
```

This home has a **very small quota**.

Confirmed quota snapshot:

- `/homeB`: `100M` quota
- it was already effectively full because `.cache` alone had grown to about `100M`

### Scratch

Confirmed persistent scratch path for this user:

```bash
/scratch/nas/3/gmoreno
```

This is the correct place for:

- project copy
- virtual environment
- temporary files
- pip cache
- dataset

Confirmed scratch quota snapshot:

- `/scratch/nas/3`: about `30G` quota for this user
- enough space for the project, dataset, and Python environment

## Paths That Were Wrong

These older paths appeared in earlier notes or scripts, but were **not** valid for the current account setup:

```bash
/scratch/nas/g/gmoreno
/scratch/1/gmoreno
/home/gmoreno
```

Do not rely on them for this account.

## Real Remote Project Location

The project was uploaded to:

```bash
/scratch/nas/3/gmoreno/TFGpractice/TFGThirdTry3
```

## Real Remote Python Environment

The working environment was created at:

```bash
/scratch/nas/3/gmoreno/tf_env
```

Why it had to be here:

- the user home quota was too small
- pip and torch downloads need a lot of temporary space
- using `homeB` for cache/temp caused failures

## Required Scratch-Based Environment Variables

For reliable installs and runtime behavior, these should point to scratch:

```bash
export HOME=/scratch/nas/3/gmoreno/home
export TMPDIR=/scratch/nas/3/gmoreno/tmp
export PIP_CACHE_DIR=/scratch/nas/3/gmoreno/pip_cache
```

These directories were used because:

- `pip` failed when writing cache/state to the real home
- the initial install hit quota issues until `HOME`, `TMPDIR`, and `PIP_CACHE_DIR` were redirected

## Real Cluster GPU / SLURM Facts

### What `sinfo` actually showed

Confirmed partitions:

- `production`
- `interactive`
- `fpga`
- `vivado`
- `gpu`
- `memory`

For GPU work, the relevant partition is:

```bash
gpu
```

Confirmed GPU resources exposed there:

```bash
gpu:rtx2080:6
```

That means:

- the practical maximum on this cluster path is **6 GPUs**
- not 8 GPUs

## Real Account / QOS Facts

Confirmed association for this user:

- account: `gpu`
- partition: `gpu`

Confirmed useful QOS values from `sacctmgr`:

- `small_gpu`: `3:00:00`
- `medium_gpu`: `2-00:00:00`
- `big_gpu`: `04:00:00`

Important real behavior:

- submitting with the wrong partition failed
- submitting without the right account failed
- asking `big_gpu` for more than `04:00:00` failed with walltime limit issues
- asking `medium_gpu` for too few GPUs caused `QOSMinGRES`

## Real Working Submission Pattern

The large job that was accepted used:

- account: `gpu`
- partition: `gpu`
- qos: `big_gpu`
- GPUs: `6x rtx2080`
- CPUs: `24`
- RAM: `96G`
- walltime: `04:00:00`

Submission shape:

```bash
cd /scratch/nas/3/gmoreno/TFGpractice/TFGThirdTry3
sbatch -A gpu -p gpu --qos=big_gpu --gres=gpu:rtx2080:6 --cpus-per-task=24 --mem=96G --time=04:00:00 \
  --export=ALL,CONFIG_PATH=configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max256_blend.yaml,VENV_PATH=/scratch/nas/3/gmoreno/tf_env/bin/activate,HOME=/scratch/nas/3/gmoreno/home,TMPDIR=/scratch/nas/3/gmoreno/tmp,PIP_CACHE_DIR=/scratch/nas/3/gmoreno/pip_cache \
  cluster/run_multi_gpu_hdf5.slurm
```

## Real Max-GPU Launcher

This file was updated to reflect the real maximum:

```bash
cluster/run_train_cgan_hdf5_8gpu.slurm
```

Even though the filename still says `8gpu`, it now targets the real maximum available here:

- partition `gpu`
- `--gres=gpu:rtx2080:6`
- `24` CPUs
- `96G` RAM
- `04:00:00`

The historical filename was kept to avoid breaking existing references.

## How Files Were Uploaded

### Recommended target

Upload to:

```bash
/scratch/nas/3/gmoreno/TFGpractice/TFGThirdTry3
```

### What to upload first

The fastest useful order is:

1. all Python code
2. `configs/`
3. `cluster/`
4. docs if needed
5. `CKM_Dataset.h5` last

Reason:

- the codebase becomes runnable much earlier
- then only the large dataset transfer remains

### What to exclude

Exclude local junk such as:

- `outputs/`
- `__pycache__/`

### Dataset path

Remote dataset file path:

```bash
/scratch/nas/3/gmoreno/TFGpractice/TFGThirdTry3/CKM_Dataset.h5
```

## Environment Installation That Worked

The environment needed:

- `torch`
- `torchvision`
- `pandas`
- `numpy`
- `pillow`
- `tqdm`
- `pyyaml`
- `h5py`

The reliable install strategy was:

```bash
export HOME=/scratch/nas/3/gmoreno/home
export TMPDIR=/scratch/nas/3/gmoreno/tmp
export PIP_CACHE_DIR=/scratch/nas/3/gmoreno/pip_cache

python3 -m venv /scratch/nas/3/gmoreno/tf_env
source /scratch/nas/3/gmoreno/tf_env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
python -m pip install pandas numpy pillow tqdm pyyaml h5py
```

## Important Runtime Finding

Importing the new environment on the **login node** was misleading and failed with:

```text
ImportError: libnccl.so.2: failed to map segment from shared object
```

But that did **not** mean the environment was unusable.

A real test job on a GPU node succeeded and confirmed:

- `torch 2.5.1+cu121`
- `torch.cuda.is_available() == True`
- project imports worked

So for validation, trust a short GPU-node sanity job more than login-node imports.

## Proven Sanity Test Pattern

A short GPU-node validation job was submitted and completed successfully before launching the big run.

Result confirmed:

```text
2.5.1+cu121
True
IMPORT_OK
```

That means:

- CUDA torch worked on the compute node
- project imports were valid there

## CRLF / LF Issue

Uploaded `.slurm` files initially had Windows line endings.

This caused:

```text
sbatch: error: Batch script contains DOS line breaks (\r\n)
```

Fix:

- convert remote `.slurm` scripts to Unix LF before submitting

Affected files:

- `cluster/run_multi_gpu_hdf5.slurm`
- `cluster/run_train_cgan_hdf5_8gpu.slurm`

## Real Job IDs Observed

Sanity import test:

- `10007341`
- completed successfully

Main multi-GPU training submission:

- `10007343`

At submission time it was:

```text
PD (Priority)
```

## Monitoring Commands

Check queue:

```bash
squeue -u gmoreno
```

Detailed job info:

```bash
scontrol show job 10007343
```

Accounting summary:

```bash
sacct -j 10007343 --format=JobID,State,Elapsed,NodeList,ExitCode
```

Worker logs from the multi-run launcher:

```bash
ls logs_multigpu_10007343_*.out
ls logs_multigpu_10007343_*.err
sed -n '1,120p' logs_multigpu_10007343_0.out
```

Main SLURM stdout/stderr:

```bash
ls logs_multigpu_hdf5_10007343.out
ls logs_multigpu_hdf5_10007343.err
```

## Why `blend` Was Kept

Local hybrid results with hard `replace` improved only briefly and then became unstable.

The fused metric behaved too discontinuously because:

- training used a binary confidence target
- evaluation used hard switching at the fallback threshold

So this cluster experiment keeps:

```yaml
path_loss_hybrid:
  fallback_mode: blend
```

## Current Recommendation

For this cluster and account, the reliable default is:

- use `/scratch/nas/3/gmoreno` for everything heavy
- use account `gpu`
- use partition `gpu`
- use `big_gpu` only up to `04:00:00`
- use **6 GPUs max**, not 8
- validate the environment on a GPU node, not on the login node

## Secret Handling

This file intentionally does **not** repeat the password.

Connection method, paths, quotas, SLURM settings, and operational steps are documented here, but the credential itself should stay documented elsewhere if needed.
