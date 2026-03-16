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
  - Saves a lightweight validation JSON after each epoch and can run one final test pass at the end.
  - It no longer runs the heavy heuristic calibration inside training.
  - The expensive calibration step was moved out of the training loop to keep epoch transitions faster.

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
  - This is the cGAN equivalent of final held-out evaluation when used without overriding `--split`.
  - Supports `--split train`, `--split val`, `--split test`, and `--split both`.
  - It is the shared evaluation engine used by both final evaluation and validation wrappers.

### Validation

- `validate.py`
  - Wrapper around `evaluate.py`.
  - Runs the same evaluation logic but on **validation** by default.
  - Intended for development-time comparison, not final reporting.

- `validate_cgan.py`
  - Wrapper around `evaluate_cgan.py`.
  - Runs the same evaluation logic but on **validation** by default by forcing `--split val`.
  - Intended for development-time comparison, not final reporting.
  - Also saves heuristic calibration for later test-time reuse.
  - This is where the heavy heuristic calibration belongs now.

## Why `validate_cgan.py` calls `evaluate_cgan.py`

This is intentional.

- `evaluate_cgan.py` is not a test-only script internally.
- It is a generic evaluation backend that can run on `train`, `val`, `test`, or `both` depending on `--split`.
- `validate_cgan.py` exists only to provide the validation-specific default behavior:
  - force `--split val`
  - save heuristic calibration JSON

So the naming is:

- `evaluate_cgan.py`: generic evaluator, defaulting to `test`
- `validate_cgan.py`: validation wrapper over that same evaluator

This means the split logic is still correct even though one script calls the other.

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

## Heuristic calibration policy

For the cGAN HDF5 route, heuristic hyperparameters should be chosen on `validation`, not on `test`.

Current calibrated items:

- `path_loss` median-filter kernel

Workflow:

- `train_cgan.py` trains with lightweight validation only for checkpoint selection and per-epoch summaries
- `validate_cgan.py` evaluates on validation and saves the chosen heuristic parameters in `outputs/.../heuristic_calibration.json`
- `evaluate_cgan.py` evaluates on test and automatically loads that saved calibration
- `predict_cgan.py` also reuses the saved calibration by default when it exists

Important implementation detail:

- the heavy validation work is the heuristic calibration, especially `path_loss` median-kernel selection
- that heavy step is no longer executed inside `train_cgan.py`
- it is now reserved for the separate `validate_cgan.py` run

Important limitation:

- `augmented_los` does not exist as a ground-truth target in the HDF5 dataset
- so an `augmented_los` heuristic can only be derived if you have a separate LoS-related signal available at inference time, and it cannot be calibrated directly against a true HDF5 target the way `path_loss` can

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

Development-time validation and heuristic calibration:

```bash
python validate_cgan.py --config configs/cgan_unet_hdf5_amd_midvram.yaml --checkpoint outputs/.../best_cgan.pt
```

Final held-out test evaluation:

```bash
python evaluate_cgan.py --config configs/cgan_unet_hdf5_amd_midvram.yaml --checkpoint outputs/.../best_cgan.pt
```

Recommended sequence for cGAN HDF5 runs:

- train with `train_cgan.py`
- when you want proper validation metrics and updated heuristic calibration, run `validate_cgan.py`
- when you want final held-out numbers, run `evaluate_cgan.py`

With the current mid-VRAM config, this also:

- writes `validate_metrics_cgan_latest.json` after each epoch
- writes `validate_metrics_epoch_<N>_cgan.json` for each epoch
- updates `validate_metrics_cgan_best.json` and `heuristic_calibration.json` whenever a new best checkpoint appears
- runs one final `test` evaluation at the end using the best checkpoint and the saved calibration

Validation during development:

```bash
python validate_cgan.py --config configs/cgan_unet_hdf5_amd_midvram.yaml --checkpoint outputs/cgan_unet_hdf5_amd_midvram/best_cgan.pt
```

This validation step also saves heuristic calibration for later reuse in test and prediction.

Final test evaluation:

```bash
python evaluate_cgan.py --config configs/cgan_unet_hdf5_amd_midvram.yaml --checkpoint outputs/cgan_unet_hdf5_amd_midvram/best_cgan.pt
```

By default, test evaluation reuses the heuristic calibration saved from validation.

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