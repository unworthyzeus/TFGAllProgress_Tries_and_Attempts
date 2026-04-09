# RunPod Try53 Operations Log

Last updated: 2026-04-08 (Europe/Madrid)

## Purpose

This file documents, in one place and with operational detail, how `Try 53` is being run on:

- RunPod
- UPC cluster

It also records:

- how the connection is made,
- which scripts are used,
- what was uploaded,
- where logs and outputs live,
- which stages/jobs are active right now,
- and the important caveat around the misleading `resolved_batch_size=1` line in the RunPod logs.

## Local Workspace

Main local project path:

- `C:\TFG\TFGpractice`

Current try:

- `C:\TFG\TFGpractice\TFGFiftyThirdTry53`

Important local helper scripts:

- `C:\TFG\TFGpractice\cluster\upload_runpod_try53.py`
- `C:\TFG\TFGpractice\cluster\upload_runpod_try53_shell.py`
- `C:\TFG\TFGpractice\TFGFiftyThirdTry53\cluster\run_runpod_try53_sequential.sh`
- `C:\TFG\TFGpractice\TFGFiftyThirdTry53\cluster\submit_try53_cyclic_chain.py`

## RunPod Connection

### Pod

- Pod name: `fuzzy_coral_marlin`
- Pod ID: `27wguc8c92rdnx`

### SSH over exposed TCP

The working SSH endpoint is:

- host: `38.65.239.14`
- port: `35484`
- user: `root`

The private key used locally is:

- `C:\TFG\TFGpractice\runpod_ssh`

### Direct SSH command used

```bash
ssh root@38.65.239.14 -p 35484 -i C:\TFG\TFGpractice\runpod_ssh
```

### Programmatic access used from this machine

Most checks were done with Paramiko from:

- `C:\TFG\.venv\Scripts\python.exe`

Typical connection pattern:

```python
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    "38.65.239.14",
    port=35484,
    username="root",
    key_filename=r"C:\TFG\TFGpractice\runpod_ssh",
    timeout=20,
)
```

## What Was Uploaded To RunPod

The uploader was rewritten to use direct SSH/SFTP instead of the older brittle upload path.

Uploader script:

- `C:\TFG\TFGpractice\cluster\upload_runpod_try53.py`

What is intentionally uploaded now:

- `Try 53` code
- datasets when needed on a fresh pod

What is intentionally **not** uploaded now:

- pretrained outputs/checkpoints

Reason:

- the current RunPod policy for this run is to start from epoch 0 and avoid uploading large historical outputs.

Remote workspace root:

- `/workspace/TFGpractice`

Remote try root:

- `/workspace/TFGpractice/TFGFiftyThirdTry53`

RunPod launcher log:

- `/workspace/TFGpractice/TFGFiftyThirdTry53/runpod_try53_chain3.log`

Important note:

- in the current fresh rerun, the background process is alive even when that log file is temporarily missing or still empty
- when that happens, the reliable sources of truth are:
  - `ps`
  - `nvidia-smi`
  - the temporary YAMLs under `/tmp/try53_stage*.yaml`

## RunPod Launch Strategy

RunPod is **not** using Slurm. It runs a sequential bash chain.

Launcher:

- `/workspace/TFGpractice/TFGFiftyThirdTry53/cluster/run_runpod_try53_sequential.sh`

The chain currently runs:

1. `Stage 1 bootstrap`
2. `Stage 2`
3. `Stage 3`
4. `Stage 1 feedback`
5. `Stage 2 (feedback teacher)`
6. `Stage 3 (feedback teacher)`

And it repeats this chain:

- `CHAIN_REPEATS=3`

The current RunPod launch was started with:

- `START_FROM_SCRATCH=1`
- `RUNPOD_STAGE1_BATCH_SIZE=3`
- `RUNPOD_STAGE1_VAL_BATCH_SIZE=3`
- `RUNPOD_STAGE23_BATCH_SIZE=3`
- `RUNPOD_STAGE23_VAL_BATCH_SIZE=3`

That means:

- `stage1` starts from scratch on RunPod,
- `stage1`, `stage2`, and `stage3` are all intended to run with batch size `3`,
- validation batch size is also `3` in RunPod.
- the current RunPod execution is meant to start from scratch and not reuse uploaded historical outputs.

## Important Note About `resolved_batch_size=1`

There is a misleading line in the RunPod log:

```text
resolved_batch_size=1
```

This does **not** mean the final RunPod runtime config is using batch size `1`.

What is happening:

1. `cluster/prepare_runtime_config.py` prints `resolved_batch_size=...`
2. That happens **before** the RunPod-specific overrides are applied
3. Then `run_runpod_try53_sequential.sh` rewrites the temporary YAML with the RunPod batch sizes

Verified current temporary stage1 runtime YAML on RunPod:

- `/tmp/try53_stage1_bootstrap_KzjG.yaml`

Confirmed values inside that YAML:

- `training.batch_size: 3`
- `training.homogeneous_city_type_batches: true`
- `data.val_batch_size: 3`

So the actual intended RunPod batch size is `3`, even though the earlier preparation log still prints `1`.

This confusion was important enough that the launcher was patched locally to print the **post-override** effective config too:

- `C:\TFG\TFGpractice\TFGFiftyThirdTry53\cluster\run_runpod_try53_sequential.sh`

## Stage1 Batch Logic On RunPod

User requested:

- batch size `3` for `stage1`,
- but with batches containing only one `city_type`.

This was implemented in:

- `C:\TFG\TFGpractice\TFGFiftyThirdTry53\train_pmnet_prior_gan.py`

Key addition:

- `HomogeneousCityTypeBatchSampler`

What it does:

- groups samples by inferred `city_type`,
- builds per-city-type batches,
- feeds DDP with batches that are homogeneous in `city_type`,
- pads the batch list so all ranks see the same number of batches.

This is specifically meant to keep the routed/expert behavior coherent while still raising batch size.

## Current RunPod Status

### Host / machine

Current container hostname:

- `b5c9dfd0a57f`

### GPU status observed

Observed from:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

Latest observed values:

- GPU 0: `20057 MiB / 24564 MiB`, utilization snapshot `0%`
- GPU 1: `20057 MiB / 24564 MiB`, utilization snapshot `0%`
- GPU 2: `20057 MiB / 24564 MiB`, utilization snapshot `0%`
- GPU 3: `20057 MiB / 24564 MiB`, utilization snapshot `0%`

Interpretation:

- the pod is genuinely running the job right now,
- memory usage is high but still below 24 GB,
- batch size `3` is clearly not instantly OOMing,
- a single `nvidia-smi` utilization snapshot can show `0%` even while the workers are alive; in this run the stronger signal is that:
  - `torchrun` is alive,
  - 4 worker processes are alive,
  - each worker is consuming significant CPU,
  - and each GPU is holding ~20 GB of allocations.

### Current active stage

The current active RunPod stage is:

- `Stage 1 bootstrap`

Confirmed from the live process tree:

```text
bash cluster/run_runpod_try53_sequential.sh
/usr/bin/python /usr/local/bin/torchrun --standalone --nnodes=1 --nproc-per-node=4 --master-port 29731 train_pmnet_prior_gan.py --config /tmp/try53_stage1_bootstrap_KzjG.yaml
/usr/bin/python -u train_pmnet_prior_gan.py --config /tmp/try53_stage1_bootstrap_KzjG.yaml
```

And from the effective runtime YAML:

- `training.batch_size: 3`
- `data.val_batch_size: 3`
- `model.base_channels: 112`
- `runtime.output_dir: outputs/fiftythirdtry53_pmnet_prior_stage1_literature_cyclic_t53_stage1_literature_4gpu`

Current status summary:

- cycle `1/3`
- `stage1 bootstrap` is the stage in progress
- this is the fresh rerun from scratch
- no new `validate_metrics*.json` has been written yet for this fresh rerun
- the output directory exists already:
  - `/workspace/TFGpractice/TFGFiftyThirdTry53/outputs/fiftythirdtry53_pmnet_prior_stage1_literature_cyclic_t53_stage1_literature_4gpu`
- but at the last check it was still empty, so the job is still in the early part of the first epoch / initialization path

### Live process snapshot

Latest process snapshot:

- launcher bash PID elapsed ~`300s`
- `torchrun` PID elapsed ~`300s`
- 4 Python worker ranks alive
- per-rank CPU around `31%` to `36%`

That is enough to conclude that the fresh RunPod chain is alive and actively computing, even before the first validation JSON appears.

### RunPod train speed observed

Recent live progress lines from the same log show:

- around `1.02 it/s`
- with periodic slower bursts around `1.30s/it` to `1.49s/it`

Typical pattern seen:

- fast zone: `~1.02 to 1.03 it/s`
- slow zone: `~1.30 to 1.49 s/it`

Practical summary:

- `stage1` on RunPod is currently behaving like roughly `~1.0 it/s`

### Effective epoch size on RunPod stage1

The current stage1 train epoch has:

- `904` iterations

That is much smaller than the older `2710`-iteration pattern we had elsewhere.

At roughly `~1.0 it/s`, that implies:

- train-only epoch time around `~15 minutes`

Validation has not been timed cleanly yet in this current RunPod chain because:

- `stage2` and `stage3` have not been reached yet in this current run.

## Current RunPod Processes Observed

Observed process pattern:

- the outer launcher shell is still alive
- `torchrun` is active
- 4 main Python workers are active for DDP

Representative command observed:

```text
/usr/bin/python /usr/local/bin/torchrun --standalone --nnodes=1 --nproc-per-node=4 --master-port 29731 train_pmnet_prior_gan.py --config /tmp/try53_stage1_bootstrap_IT8R.yaml
```

This confirms:

- RunPod is currently training `stage1`,
- using `torchrun`,
- on 4 GPUs,
- from the temporary runtime YAML.

## Temporary Runtime YAMLs Seen On RunPod

Observed temp files:

- `/tmp/try53_stage1_bootstrap_IT8R.yaml`
- `/tmp/try53_stage1_feedback_final_F6AH.yaml`
- `/tmp/try53_stage1_feedback_runtime_IkpW.yaml`
- `/tmp/try53_stage2_runtime_ZRDX.yaml`
- `/tmp/try53_stage3_runtime_sklN.yaml`

Current real parsed content confirmed only for the active bootstrap YAML:

- `output_dir=outputs/fiftythirdtry53_pmnet_prior_stage1_moe112_cyclic_t53_stage1_moe112_4gpu`
- `resume_checkpoint=` empty
- `training.batch_size=3`
- `data.val_batch_size=3`
- `training.homogeneous_city_type_batches=true`

The other temp files existed but were empty placeholders at the moment they were inspected, which is expected before those stages are actually prepared/populated.

## UPC Cluster Status

### How we connect

Host:

- `sert.ac.upc.edu`

User:

- `gmoreno`

Typical authentication used during this thread:

- password via environment or direct Paramiko password auth

### Current queue snapshot

Observed from:

```bash
squeue -u gmoreno
```

Current running job:

- `10021758` is `RUNNING` on `sert-2001`

Elapsed when checked:

- `44:03`

Everything else currently visible in this chain is pending on dependency.

### Current UPC chain

The current `Try 53` UPC chain was launched 3 times with cleanup jobs interleaved.

The active chain structure is:

1. cleanup
2. `stage1 bootstrap`
3. cleanup
4. `stage2`
5. cleanup
6. `stage3`
7. cleanup
8. `stage1 feedback`
9. cleanup
10. `stage2`
11. cleanup
12. `stage3`
13. cleanup

Repeated 3 times.

### Current UPC job ids

First repeat:

- `10021757` cleanup
- `10021758` stage1 bootstrap
- `10021759` cleanup
- `10021760` stage2
- `10021761` cleanup
- `10021762` stage3
- `10021763` cleanup
- `10021764` stage1 feedback
- `10021765` cleanup
- `10021766` stage2
- `10021767` cleanup
- `10021768` stage3
- `10021769` cleanup

Second repeat:

- `10021770` cleanup
- `10021771` stage1 bootstrap
- `10021772` cleanup
- `10021773` stage2
- `10021774` cleanup
- `10021775` stage3
- `10021776` cleanup
- `10021777` stage1 feedback
- `10021778` cleanup
- `10021779` stage2
- `10021780` cleanup
- `10021781` stage3
- `10021782` cleanup

Third repeat:

- `10021783` cleanup
- `10021784` stage1 bootstrap
- `10021785` cleanup
- `10021786` stage2
- `10021787` cleanup
- `10021788` stage3
- `10021789` cleanup
- `10021790` stage1 feedback
- `10021791` cleanup
- `10021792` stage2
- `10021793` cleanup
- `10021794` stage3
- `10021795` cleanup

### sacct snapshot

Observed state at check time:

- `10021758` running
- `10021760` pending
- `10021762` pending
- `10021764` pending
- `10021766` pending
- `10021768` pending
- `10021770` and onward pending

So at the moment:

- UPC is still at the first `stage1 bootstrap`
- RunPod is also currently at `stage1 bootstrap`

## Stage3 Size Change

The current `Try 53` `stage3` was intentionally enlarged.

Config file:

- `C:\TFG\TFGpractice\TFGFiftyThirdTry53\experiments\fiftythirdtry53_pmnet_tail_refiner_cyclic\fiftythirdtry53_pmnet_stage3_nlos_cycle.yaml`

Important current values:

- `model.base_channels: 64`
- `tail_refiner.refiner_base_channels: 64`

This change was made because the previous `stage3` checkpoint size looked too small and likely underpowered.

## What Is Still Unknown Right Now

Not yet measured in the **current** RunPod chain:

- actual `stage2` it/s on 4090s with batch size `3`
- actual `stage3` it/s on 4090s with batch size `3`
- whether validation at batch size `3` remains stable across all stages

Those measurements can only be taken once:

- `stage1 bootstrap` finishes,
- and the chain reaches `stage2` and `stage3`.

## Commands Used To Inspect Status

### RunPod

GPU usage:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

Processes:

```bash
ps -eo pid,etimes,cmd | grep -E 'run_runpod_try53_sequential|train_pmnet|torchrun' | grep -v grep
```

Log tail:

```bash
tail -n 120 /workspace/TFGpractice/TFGFiftyThirdTry53/runpod_try53_chain3.log
```

Safer Unicode-clean parsing:

```python
from pathlib import Path
p = Path('/workspace/TFGpractice/TFGFiftyThirdTry53/runpod_try53_chain3.log')
text = p.read_text(encoding='utf-8', errors='replace')
for line in text.splitlines():
    if 'Stage ' in line or 'train:' in line or 'val:' in line:
        print(''.join(ch if ord(ch) < 128 else '?' for ch in line))
```

### UPC

Queue:

```bash
squeue -u gmoreno
```

Accounting:

```bash
sacct -j <jobids> --format=JobID,State,Elapsed,ExitCode
```

## Operational Takeaways

1. RunPod is alive and genuinely training right now.
2. The active RunPod stage is `stage1 bootstrap`.
3. The actual intended RunPod batch size is `3`, even though one earlier log line still says `resolved_batch_size=1`.
4. RunPod `stage1` is currently around `~1.0 it/s` with `904` train iterations per epoch.
5. UPC is still in the first `stage1 bootstrap` of the currently queued `Try 53` chain.
6. `stage2` and `stage3` timings on RunPod still need to be measured once the chain reaches them.
