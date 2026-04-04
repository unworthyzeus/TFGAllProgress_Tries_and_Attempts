#!/bin/bash
#SBATCH --job-name=t49-prior-calib-1gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs_t49_prior_calibration_1gpu_%j.out
#SBATCH --error=logs_t49_prior_calibration_1gpu_%j.err

set -euo pipefail

TRY_DIR=${TRY_DIR:-TFGFortyNinthTry49}
CONFIG_REL=${CONFIG_REL:-experiments/fortyninthtry49_pmnet_prior_gan_fastbatch/fortyninthtry49_pmnet_prior_stage1_widen112_initial.yaml}
DATASET=${DATASET:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_270326.h5}
BASE_CAL=${BASE_CAL:-../TFGFortyEighthTry48/prior_calibration/regime_obstruction_train_only_directml_quick.json}
OUT_JSON=${OUT_JSON:-prior_calibration/regime_obstruction_train_only_try49_cuda.json}
OUT_MD=${OUT_MD:-prior_calibration/regime_obstruction_train_only_try49_cuda.md}
SAMPLE_PROB=${SAMPLE_PROB:-0.01}
PYTHON_BIN=${PYTHON_BIN:-python}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

"$PYTHON_BIN" ../scripts/fit_formula_prior_obstruction_calibration.py \
  --try-dir "$TRY_DIR" \
  --config "$CONFIG_REL" \
  --dataset "$DATASET" \
  --base-calibration-json "$BASE_CAL" \
  --out-json "$OUT_JSON" \
  --out-md "$OUT_MD" \
  --sample-prob "$SAMPLE_PROB" \
  --device cuda
