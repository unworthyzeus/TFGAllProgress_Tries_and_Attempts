#!/bin/bash
#SBATCH --job-name=ckm-hdf5-cgan
#SBATCH -p small_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=logs_train_cgan_hdf5_%j.out
#SBATCH --error=logs_train_cgan_hdf5_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-configs/cgan_unet_hdf5.yaml}
PYTHON_BIN=${PYTHON_BIN:-python}
VENV_PATH=${VENV_PATH:-/scratch/nas/g/gmoreno/tf_env/bin/activate}
RUNTIME_CONFIG=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_train_cgan_hdf5_XXXX.yaml")

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"
trap 'rm -f "$RUNTIME_CONFIG"' EXIT

echo "Running cGAN training with config: $CONFIG_PATH"
$PYTHON_BIN cluster/prepare_runtime_config.py --input-config "$CONFIG_PATH" --output-config "$RUNTIME_CONFIG"
$PYTHON_BIN train_cgan.py --config "$RUNTIME_CONFIG"
