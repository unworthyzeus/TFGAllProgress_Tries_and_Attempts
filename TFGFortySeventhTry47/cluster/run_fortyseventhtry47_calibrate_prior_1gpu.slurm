#!/bin/bash
# Try 47 prior calibration on cluster GPU (CUDA)
#SBATCH --job-name=ckm-t47-calib
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs_calibrate_fortyseventhtry47_prior_1gpu_%j.out
#SBATCH --error=logs_calibrate_fortyseventhtry47_prior_1gpu_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/fortyseventhtry47_unet_prior_nlos_moe/fortyseventhtry47_unet_prior_nlos_moe.yaml}
PYTHON_BIN=${PYTHON_BIN:-python}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
DATASET_PATH=${DATASET_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_270326.h5}
SAMPLE_PROB=${SAMPLE_PROB:-0.01}
TRAIN_SAMPLE_FRAC=${TRAIN_SAMPLE_FRAC:-0.20}
VAL_SAMPLE_FRAC=${VAL_SAMPLE_FRAC:-0.35}
LOG_EVERY=${LOG_EVERY:-25}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4
"$PYTHON_BIN" fit_formula_prior_obstruction_calibration.py \
  --try-dir "$SLURM_SUBMIT_DIR" \
  --config "$CONFIG_PATH" \
  --dataset "$DATASET_PATH" \
  --base-calibration-json "$SLURM_SUBMIT_DIR/prior_calibration/regime_quadratic_train_only.json" \
  --out-json "$SLURM_SUBMIT_DIR/prior_calibration/regime_obstruction_train_only.json" \
  --out-md "$SLURM_SUBMIT_DIR/prior_calibration/regime_obstruction_train_only.md" \
  --device cuda \
  --sample-prob "$SAMPLE_PROB" \
  --train-sample-frac "$TRAIN_SAMPLE_FRAC" \
  --val-sample-frac "$VAL_SAMPLE_FRAC" \
  --log-every "$LOG_EVERY"
