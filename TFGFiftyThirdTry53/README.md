# Try 53 (Cyclic Feedback)

`Try 53` is a cyclic training branch built around the stronger supervised
`Try 51` stage1 recipe and repeated feedback loops across the three stages.

Core cycle:

1. stage1 bootstrap from `Try 51` stage1 checkpoint
2. stage2 tail refiner (teacher=stage1)
3. stage3 NLoS global-context refiner (teacher=stage2)
4. stage1 feedback resume (stage1 is re-trained with regime weights tuned from latest stage2/stage3 validation)
5. stage2 again with updated stage1
6. stage3 again with updated stage2

The second cycle starts only after the first cycle finishes.

## What "cyclic" means in this Try

This implementation is **metric-guided cyclic feedback**.

- stage1 is resumed after stage2/stage3 have been trained.
- before stage1 feedback training, `cluster/prepare_try53_stage1_feedback_config.py` reads
  the latest stage2/stage3 validation JSON and adjusts stage1 regime weighting
  (mainly `nlos_weight` and `low_antenna_boost`).
- stage2 and stage3 then retrain on the updated stage1 teacher.

This is a stable intermediate step before full online co-training.

## New Try53 Configs

- `experiments/fiftythirdtry53_pmnet_prior_gan_cyclic/fiftythirdtry53_pmnet_prior_stage1_cycle0_bootstrap.yaml`
- `experiments/fiftythirdtry53_pmnet_prior_gan_cyclic/fiftythirdtry53_pmnet_prior_stage1_cycle_feedback_resume.yaml`
- `experiments/fiftythirdtry53_pmnet_tail_refiner_cyclic/fiftythirdtry53_pmnet_tail_refiner_stage2_cycle.yaml`
- `experiments/fiftythirdtry53_pmnet_tail_refiner_cyclic/fiftythirdtry53_pmnet_stage3_nlos_cycle.yaml`

## Cluster Entry Points

- `cluster/run_fiftythirdtry53_stage1_bootstrap_4gpu.slurm`
- `cluster/run_fiftythirdtry53_stage1_feedback_4gpu.slurm`
- `cluster/run_fiftythirdtry53_stage2_4gpu.slurm`
- `cluster/run_fiftythirdtry53_stage3_4gpu.slurm`
- `cluster/run_fiftythirdtry53_cleanup_sert2001_1gpu.slurm`
- `cluster/submit_try53_cyclic_chain.py`

## Output Paths

- stage1: `outputs/fiftythirdtry53_pmnet_prior_stage1_literature_cyclic_t53_stage1_literature_4gpu`
- stage2: `outputs/fiftythirdtry53_tail_refiner_stage2_teacher_literature_cyclic_4gpu`
- stage3: `outputs/fiftythirdtry53_stage3_nlos_global_context_cyclic_4gpu`

## How To Use

### 1) Run one cyclic chain (13 jobs)

```powershell
$env:SSH_PASSWORD = '***'
python C:/TFG/TFGpractice/TFGFiftyThirdTry53/cluster/submit_try53_cyclic_chain.py
```

### 2) Run multiple chained cycles (e.g. 39 jobs = 3 x 13)

```powershell
$env:SSH_PASSWORD = '***'
python C:/TFG/TFGpractice/TFGFiftyThirdTry53/cluster/submit_try53_cyclic_chain.py --chain-repeats 3
```

### 3) Cancel queue first, then submit

```powershell
$env:SSH_PASSWORD = '***'
python C:/TFG/TFGpractice/TFGFiftyThirdTry53/cluster/submit_try53_cyclic_chain.py --cancel-all-user-jobs --chain-repeats 3
```

### 4) Reuse already uploaded files

```powershell
$env:SSH_PASSWORD = '***'
python C:/TFG/TFGpractice/TFGFiftyThirdTry53/cluster/submit_try53_cyclic_chain.py --skip-upload --chain-repeats 3
```

## Operational Notes

- `Try53` bootstrap stage1 is initialized from the `Try 51` best stage1 checkpoint.
- `stage2` still learns from the `Try53` stage1 branch itself; only the `Try53` stage1 architecture/config was reverted to the stronger `Try 51` recipe.
- tail-refiner checkpoint resolution in this branch supports fallback to latest `epoch_*` if `best_*` is absent.
- stage3 supports automatic checkpoint adaptation when an older checkpoint exists with a smaller channel width.
- feedback tuning is intentionally bounded (`nlos_weight` is clamped) to avoid unstable oscillations. This maybe needs tuning, all error is in nLoS after all.
- this branch keeps the same data split policy (`city_holdout`) used in `Try52`.

## RunPod Runbook

### Connect

```powershell
ssh um4osf59dsejs6-64411e22@ssh.runpod.io -i C:\TFG\TFGpractice\runpod_ssh
```

Safer explicit form:

```powershell
ssh -i C:\TFG\TFGpractice\runpod_ssh -o IdentitiesOnly=yes um4osf59dsejs6-64411e22@ssh.runpod.io
```

After a successful login, you should see:

- the RunPod banner
- a `root@...:/#` prompt
- `/workspace/TFGpractice` inside the pod

If the shell does not open, check the private key path first. This runbook
expects `C:\TFG\TFGpractice\runpod_ssh`.

### RunPod: sí y no

#### Sí

- Use `ssh -i ...` from Windows PowerShell to enter the pod.
- Use `python C:\TFG\TFGpractice\cluster\upload_runpod_try53.py` to send the files.
- Keep `Datasets`, `TFGFiftyFirstTry51`, `TFGFiftyThirdTry53`, and `cluster` in sync before starting the cycle.
- Use `bash cluster/run_runpod_try53_sequential.sh` inside the pod.
- Verify the Try 51 checkpoint path before launching the chain.

#### No

- Do not use `sbatch`, `squeue`, or `scancel` on RunPod. Those are UPC cluster commands.
- Do not use `ssh -T` for manual login; you want an interactive shell.
- Do not start the cycle before uploading the repo tree and datasets.
- Do not run the UPC cluster launch scripts inside RunPod.
- Do not assume `/scratch/nas/3/gmoreno/TFGpractice` exists before the uploader creates the symlink.

### Upload everything needed to RunPod

```powershell
python C:\TFG\TFGpractice\cluster\upload_runpod_try53.py --ssh-key C:\TFG\TFGpractice\runpod_ssh --items datasets,try51_stage1_bootstrap,try53
```

This uploads:

- `Datasets` to `/workspace/TFGpractice/Datasets`
- `cluster_outputs/TFGFiftyFirstTry51/fiftyfirsttry51_pmnet_prior_stage1_literature_t51_stage1_w112_4gpu` to `/workspace/TFGpractice/TFGFiftyFirstTry51/outputs/fiftyfirsttry51_pmnet_prior_stage1_literature_t51_stage1_w112_4gpu`
- `TFGFiftyThirdTry53` to `/workspace/TFGpractice/TFGFiftyThirdTry53`

The uploader also creates this symlink inside the pod:

```bash
/scratch/nas/3/gmoreno/TFGpractice -> /workspace/TFGpractice
```

### Why this upload is faster now

- The uploader sends each selected folder as one raw `tar.gz` stream over the SSH PTY instead of base64-encoding every chunk.
- The remote side switches the PTY into full raw mode (`stty raw -echo -isig -ixon -ixoff -opost`), writes the payload directly with `cat > ...`, then extracts the tarball in a second SSH command.
- During test runs, the uploader now emits progress sooner and keeps the remote PTY alive with a short heartbeat, so failures show up faster.
- Progress is reported in Mbps and ETA, which is easier to read than the old byte-by-byte output.
- The big win is `Datasets`: uploading it as one tar stream is much faster than sending it file by file.
- Until the tar finishes and extraction runs, you will not see the new files appear in `/workspace/TFGpractice`.

### Works

- `python C:\TFG\TFGpractice\cluster\upload_runpod_try53.py --ssh-key C:\TFG\TFGpractice\runpod_ssh` works as the fast upload entry point.
- The default upload order for the current RunPod setup is `datasets`, `try51_stage1_bootstrap`, `try53`.
- `--items try53,try53_outputs,cluster` is useful when only code and local outputs changed.
- The `/scratch/nas/3/gmoreno/TFGpractice -> /workspace/TFGpractice` symlink creation works and is required for the old absolute paths.
- If you interrupt the upload with `Ctrl+C`, the script now exits cleanly instead of dumping a traceback.
- If the SSH connection drops during the tar stream, the script will stop cleanly and you must rerun it from the start.

### Doesn't

- `scp -O` does not work reliably in this RunPod setup.
- `ssh -T` is not the manual login path here; use the interactive `ssh -i ...` command.
- File-by-file dataset uploads are too slow and should be avoided.
- The old base64-over-PTY upload path still works as a fallback, but it is the slow option.
- Interrupted tar uploads are not resumable; rerun the same command from the start.
- A dropped SSH connection before extraction is treated the same way: restart from the beginning.

### Quick Check

After connecting, these checks confirm the expected files are there:

```bash
test -f /workspace/TFGpractice/Datasets/CKM_Dataset_270326.h5 && echo DATASET_OK
test -f /workspace/TFGpractice/TFGFiftyFirstTry51/outputs/fiftyfirsttry51_pmnet_prior_stage1_literature_t51_stage1_w112_4gpu/best_cgan.pt && echo TRY51_BOOTSTRAP_OK
test -d /workspace/TFGpractice/TFGFiftyThirdTry53 && echo TRY53_OK
```

### Run the full cyclic sequence sequentially on RunPod

```bash
cd /workspace/TFGpractice/TFGFiftyThirdTry53
bash cluster/run_runpod_try53_sequential.sh
```

### Important checkpoint path from Try 51

The stage1 bootstrap expects:

```bash
/scratch/nas/3/gmoreno/TFGpractice/TFGFiftyFirstTry51/outputs/fiftyfirsttry51_pmnet_prior_stage1_literature_t51_stage1_w112_4gpu/best_cgan.pt
```

### Status note

The Try 53 run is cyclic: stage3 validation feeds stage1 feedback, then stage2 and stage3 repeat with the updated teacher.

### RunPod-specific training choices

- RunPod is intended to use larger batches than UPC.
- Current launcher overrides are designed for:
  - `stage1 batch_size = 3`
  - `stage1 val_batch_size = 3`
  - `stage2 batch_size = 3`
  - `stage2 val_batch_size = 3`
  - `stage3 batch_size = 3`
  - `stage3 val_batch_size = 3`
- `stage1` also uses homogeneous `city_type` batches on RunPod to keep routed behavior consistent when batch size is greater than `1`.
- The line `resolved_batch_size=1` printed by `prepare_runtime_config.py` is not the final RunPod batch size; the RunPod launcher applies batch overrides afterwards when generating the temporary runtime YAML.
