#!/bin/bash
# Try 34 hybrid two-ray/COST231 prior-input path-loss variant - plain 1 GPU
#SBATCH --job-name=ckm-t34-plain-1gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs_train_thirtyfourthtry34_hybrid_two_ray_prior_input_1gpu_%j.out
#SBATCH --error=logs_train_thirtyfourthtry34_hybrid_two_ray_prior_input_1gpu_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/thirtyfourthtry34_hybrid_two_ray_prior_input/thirtyfourthtry34_hybrid_two_ray_prior_input.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train.py}
PYTHON_BIN=${PYTHON_BIN:-python}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-t34_hybrid_two_ray_prior_input_plain_1gpu}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_ddp_XXXX.yaml")
cleanup() { rm -f "$runtime_config"; }
trap cleanup EXIT

HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_180326_antenna_height.h5}

"$PYTHON_BIN" cluster/prepare_runtime_config.py \
    --input-config "$CONFIG_PATH" \
    --output-config "$runtime_config" \
    --output-suffix "$OUTPUT_SUFFIX" \
    --hdf5-path "$HDF5_PATH"

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} OMP_NUM_THREADS=4

"$PYTHON_BIN" "$TRAIN_SCRIPT" --config "$runtime_config"