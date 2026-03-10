# TFG_FirstTry1

Proposal-aligned prototype for CKM prediction in UAV-enabled FR3 scenarios.

## What this prototype predicts
- **Delay spread**: `delay_spread`
- **Angular spread**: `angular_spread`
- **Channel power**: `channel_power`
- **Augmented LoS**: `augmented_los` (binary map, trained with BCE by default)

The outputs are fully configurable in `configs/baseline.yaml` through:
- `target_columns`
- `target_losses`
- `model.out_channels`

## Dataset format
This project does **not** depend on `TFGPractice`. Use your own dataset and manifests.

1. Copy `manifest_template.csv` and create your own train/val manifests.
2. Set paths in `configs/baseline.yaml`:
   - `data.root_dir`
   - `data.train_manifest`
   - `data.val_manifest`
3. Ensure each target column in `target_columns` exists in your manifests.

Optional:
- If your model input includes binary LoS map, set `data.los_input_column: binary_los`.
- If not, set `data.los_input_column: null`.

For scalar inputs:
- Put **varying** per-sample values in manifest columns listed in `data.scalar_feature_columns` (e.g., `antenna_height`).
- Put **hardcoded** global values in `data.constant_scalar_features` (e.g., `frequency_ghz: 7.125`, fixed `antenna_power`, fixed `bandwidth`).
- Typical reference values: macro BS around `46 dBm`, UE/small cell often around `23 dBm`.

If you want to **not predict augmented_los**, use `configs/proposal_regression_only.yaml` (3 outputs only).

### Toggle `augmented_los` ON/OFF
- **ON**: use [configs/baseline.yaml](configs/baseline.yaml) (includes `augmented_los` in `target_columns`).
- **OFF**: use [configs/proposal_regression_only.yaml](configs/proposal_regression_only.yaml) (excludes `augmented_los`).

You can switch any run by changing only the config file passed to `train.py`, `evaluate.py`, and `predict.py`.

## Install
```bash
python -m pip install -r requirements.txt
```

## Train
```bash
python train.py --config configs/baseline.yaml
python train.py --config configs/proposal_regression_only.yaml
```

## Evaluate
```bash
python evaluate.py --config configs/baseline.yaml --checkpoint outputs/baseline_run/best.pt
python evaluate.py --config configs/proposal_regression_only.yaml --checkpoint outputs/proposal_regression_only/best.pt
```

## Predict one sample
Without LoS input channel:
```bash
python predict.py --config configs/baseline.yaml --checkpoint outputs/baseline_run/best.pt --input path/to/input.png --scalar-values antenna_height=120
```

With LoS input channel enabled in config:
```bash
python predict.py --config configs/baseline.yaml --checkpoint outputs/baseline_run/best.pt --input path/to/input.png --los-input path/to/los_input.png --scalar-values antenna_height=120
```

## Cluster (UPC SLURM)
- Train: `cluster/run_train.slurm`
- Eval: `cluster/run_eval.slurm`

Submit with:
```bash
sbatch cluster/run_train.slurm
sbatch cluster/run_eval.slurm
```
