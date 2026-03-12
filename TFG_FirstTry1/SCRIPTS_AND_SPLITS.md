# Scripts and Splits

This document explains what each main script does in `TFG_FirstTry1` and how dataset splitting works now.

## Current split policy

For the HDF5 configs, the project now uses an automatic **70/15/15** split:

- `train`: 70%
- `val`: 15%
- `test`: 15%

The split is created automatically from `CKM_Dataset.h5` using:

- `data.val_ratio: 0.15`
- `data.test_ratio: 0.15`
- `data.split_seed: 42`

Important details:

- The split is deterministic as long as `split_seed` stays the same.
- The split is currently **sample-level random**.
- That means `train`, `val`, and `test` do not share the same samples.
- It does **not** mean they are city-disjoint.
- So `test` is a proper held-out sample set, but not yet a held-out-city benchmark.

## What this means in practice

- `train` is used to optimize model weights.
- `val` is used during experimentation to compare runs, tune settings, and monitor overfitting.
- `test` is reserved for final evaluation.

Because the old HDF5 configs used `90/10` without a separate `test`, older checkpoints should not be treated as directly comparable to the new `70/15/15` protocol. If you want clean final numbers under the new protocol, retrain from scratch.

## Main scripts

### Training

- `train.py`
  - Trains the standard U-Net model.
  - Uses `train` for optimization and `val` for checkpoint selection.
  - Saves checkpoints like `best.pt` and `epoch_*.pt`.

- `train_cgan.py`
  - Trains the cGAN version: generator + discriminator.
  - Uses `train` for optimization and `val` for checkpoint selection.
  - Saves checkpoints like `best_cgan.pt` and `epoch_*_cgan.pt`.

- `cross_validate.py`
  - Runs **k-fold cross-validation** for the standard U-Net.
  - Builds folds from the current development pool, meaning `train + val`.
  - Keeps `test` outside the folds.
  - Saves one best checkpoint per fold plus an aggregated summary.

- `cross_validate_cgan.py`
  - Runs **k-fold cross-validation** for the cGAN model.
  - Builds folds from the current development pool, meaning `train + val`.
  - Keeps `test` outside the folds.
  - Saves one best generator checkpoint per fold plus an aggregated summary.

### Evaluation

- `evaluate.py`
  - Evaluates the standard U-Net on **test by default**.
  - This is the script for final held-out evaluation.
  - You can still override with `--split train`, `--split val`, `--split test`, or `--split both`.

- `evaluate_cgan.py`
  - Evaluates the cGAN generator on **test by default**.
  - This is the cGAN equivalent of final held-out evaluation.
  - Supports `--split train`, `--split val`, `--split test`, and `--split both`.

### Validation

- `validate.py`
  - Wrapper around `evaluate.py`.
  - Runs the same evaluation logic but on **validation** by default.
  - Intended for development-time comparison, not final reporting.

- `validate_cgan.py`
  - Wrapper around `evaluate_cgan.py`.
  - Runs the same evaluation logic but on **validation** by default.
  - Intended for development-time comparison, not final reporting.

### Inference

- `predict.py`
  - Runs one-sample inference with the standard U-Net.
  - Used to export predicted maps from a trained U-Net checkpoint.

- `predict_cgan.py`
  - Runs one-sample inference with the cGAN generator.
  - Exports predicted maps and the HDF5-specific postprocessed outputs.
  - Can also export derived physical maps from `path_loss`, depending on config.

## How train/val/test are used internally

The split logic is centralized in `data_utils.py`.

- Manifest mode:
  - `train_manifest` feeds `train`
  - `val_manifest` feeds `val`
  - `test_manifest` optionally feeds `test`

- HDF5 mode:
  - samples are listed from the HDF5 file
  - shuffled with `split_seed`
  - partitioned into `train`, `val`, and optional `test`

The evaluation scripts disable augmentation automatically, so `val` and `test` are always measured without random flips or rotations.

Because split_seed is set on the yaml as 42, the splits do not change in different iterations.

## Do I need to retrain to avoid cross contamination?

Short answer:

- **Sample overlap contamination is already avoided** by the current split logic.
- **But you should retrain from scratch** if you want clean results under the new `70/15/15` protocol and you previously trained under the old `90/10` setup.

What is already safe now:

- `train`, `val`, and `test` are disjoint at the sample level.
- The normal `train.py` and `train_cgan.py` workflow does **not** do cross-validation by itself.
- The new `cross_validate.py` and `cross_validate_cgan.py` scripts are separate tools and do not change the normal train/evaluate behavior unless you run them explicitly.

What can still contaminate a new protocol run:

- old checkpoints trained before the split change
- automatic resume from an old output directory

This matters because both training scripts support automatic resume:

- `train.py` looks for `epoch_*.pt` or `best.pt`
- `train_cgan.py` looks for `epoch_*_cgan.pt` or `best_cgan.pt`

So if you keep the same `output_dir` and it already contains checkpoints from the old `90/10` setup, the script may continue from weights that already saw data under the previous protocol. That would make the new `test` evaluation no longer clean as a final benchmark.

### Practical decision rule

You **should retrain from scratch** if any of these are true:

- the checkpoint was created before the `70/15/15` split was introduced
- the run resumed automatically from an old checkpoint in the same `output_dir`
- you want publishable or supervisor-facing final numbers under the new protocol

You **do not need to retrain again** if all of these are true:

- you started training only after the `70/15/15` change
- the run started from scratch, not from an old checkpoint
- the `output_dir` was fresh, or at least did not contain old resumable checkpoints

### Safe way to start a clean run

Use one of these approaches:

- change `runtime.output_dir` to a new empty directory
- or set `runtime.resume_checkpoint: null` and make sure the output directory does not contain old `best` or `epoch_*` checkpoints

Recommended interpretation:

- for exploratory work, reusing old runs may still be useful
- for final model selection and final `test` numbers, start from scratch under the new split

## Cross-validation policy

Cross-validation is now available, but it is **separate** from the normal train/validate/evaluate workflow.

- The fixed `test` split remains untouched.
- The k-fold procedure runs only on the development pool: `train + val`.
- For HDF5 configs with `70/15/15`, this means:
  - `test` stays fixed at 15%
  - the remaining 85% is used for cross-validation

This is the intended behavior:

- use `cross_validate.py` or `cross_validate_cgan.py` to estimate model stability and tuning quality
- use `train.py` or `train_cgan.py` for a normal single run
- use `evaluate.py` or `evaluate_cgan.py` only for final held-out test reporting

Current scope:

- cross-validation uses random folds over the development pool
- it is still sample-level, not city-level
- there is no city-held-out cross-validation yet

## Generalization and overfitting checks

Both evaluation scripts support:

- `--split train`
- `--split val`
- `--split test`
- `--split both`

When you use `--split both`, they report:

- metrics on `train`
- metrics on `val`
- `_generalization_gap`

This is useful for checking whether the model is learning or just fitting the training set too closely.

Interpretation:

- small gap between `train` and `val` usually means better generalization
- large gap means stronger overfitting risk
- final model reporting should still use `test`

## Recommended workflow

### Standard U-Net

Train:

```bash
python train.py --config configs/baseline_hdf5.yaml
```

Validation during development:

```bash
python validate.py --config configs/baseline_hdf5.yaml --checkpoint outputs/.../best.pt
```

Final test evaluation:

```bash
python evaluate.py --config configs/baseline_hdf5.yaml --checkpoint outputs/.../best.pt
```

### cGAN

Train:

```bash
python train_cgan.py --config configs/cgan_unet_hdf5_amd_midvram.yaml
```

Validation during development:

```bash
python validate_cgan.py --config configs/cgan_unet_hdf5_amd_midvram.yaml --checkpoint outputs/cgan_unet_hdf5_amd_midvram/best_cgan.pt
```

Final test evaluation:

```bash
python evaluate_cgan.py --config configs/cgan_unet_hdf5_amd_midvram.yaml --checkpoint outputs/cgan_unet_hdf5_amd_midvram/best_cgan.pt
```

Cross-validation over `train + val`, keeping `test` fixed:

```bash
python cross_validate_cgan.py --config configs/cgan_unet_hdf5_amd_midvram.yaml --folds 5
```

Optional test readout for each fold-best model:

```bash
python cross_validate_cgan.py --config configs/cgan_unet_hdf5_amd_midvram.yaml --folds 5 --evaluate-test
```

For the standard U-Net:

```bash
python cross_validate.py --config configs/baseline_hdf5.yaml --folds 5
```

## Cluster scripts

The main HDF5 cluster scripts are:

- `cluster/run_train_hdf5.slurm`
  - standard U-Net HDF5 training on SLURM

- `cluster/run_train_cgan_hdf5.slurm`
  - cGAN HDF5 training on SLURM

- `cluster/run_eval_hdf5.slurm`
  - evaluation on SLURM

- `cluster/run_multi_gpu_hdf5.slurm`
  - launches multiple independent training processes, one per GPU

- `cluster/prepare_runtime_config.py`
  - generates a runtime config before launch
  - used to adapt settings such as batch size or output directory at submission time

## Configs affected by the new split

All HDF5 configs in `configs/` were updated to the same policy:

- `baseline_hdf5.yaml`
- `baseline_hdf5_amd.yaml`
- `baseline_hdf5_cuda_max.yaml`
- `proposal_regression_only_hdf5.yaml`
- `cgan_unet_hdf5.yaml`
- `cgan_unet_hdf5_amd.yaml`
- `cgan_unet_hdf5_amd_lowvram.yaml`
- `cgan_unet_hdf5_amd_midvram.yaml`
- `cgan_unet_hdf5_amd_max.yaml`
- `cgan_unet_hdf5_cuda_max.yaml`

So from now on, `validate*.py` means validation and `evaluate*.py` means final test evaluation across the full HDF5 workflow.