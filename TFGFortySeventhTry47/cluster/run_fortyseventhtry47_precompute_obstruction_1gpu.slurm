#!/bin/bash
# Try 47 obstruction-feature precompute cache
#SBATCH --job-name=ckm-t47-precomp
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs_precompute_fortyseventhtry47_obstruction_1gpu_%j.out
#SBATCH --error=logs_precompute_fortyseventhtry47_obstruction_1gpu_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/fortyseventhtry47_unet_prior_nlos_moe/fortyseventhtry47_unet_prior_nlos_moe.yaml}
PYTHON_BIN=${PYTHON_BIN:-python}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_270326.h5}
OUT_HDF5=${OUT_HDF5:-precomputed/obstruction_features_u8.h5}
NUM_WORKERS=${NUM_WORKERS:-8}
MAX_INFLIGHT=${MAX_INFLIGHT:-16}
LOG_EVERY=${LOG_EVERY:-100}
COMPRESSION=${COMPRESSION:-gzip}
COMPRESSION_LEVEL=${COMPRESSION_LEVEL:-1}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
"$PYTHON_BIN" precompute_obstruction_features.py \
  --config "$CONFIG_PATH" \
  --dataset "$HDF5_PATH" \
  --out-hdf5 "$OUT_HDF5" \
  --num-workers "$NUM_WORKERS" \
  --max-inflight "$MAX_INFLIGHT" \
  --log-every "$LOG_EVERY" \
  --compression "$COMPRESSION" \
  --compression-level "$COMPRESSION_LEVEL"
