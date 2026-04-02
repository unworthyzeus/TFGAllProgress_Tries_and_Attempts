#!/bin/bash
# Try 46 PMNet + explicit LoS/NLoS branching + NLoS-only MoE - 2 GPU
#SBATCH --job-name=ckm-t46-branch-2gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=medium_gpu
#SBATCH --gres=gpu:rtx2080:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs_train_fortysixthtry46_los_nlos_moe_prior_2gpu_%j.out
#SBATCH --error=logs_train_fortysixthtry46_los_nlos_moe_prior_2gpu_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/fortysixthtry46_los_nlos_moe_prior/fortysixthtry46_los_nlos_moe_prior.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_pmnet_residual.py}
PYTHON_BIN=${PYTHON_BIN:-python}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-t46_los_nlos_moe_prior_2gpu}
HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_270326.h5}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_pmnet_XXXX.yaml")
cleanup() { rm -f "$runtime_config"; }
trap cleanup EXIT

"$PYTHON_BIN" cluster/prepare_runtime_config.py \
  --input-config "$CONFIG_PATH" \
  --output-config "$runtime_config" \
  --output-suffix "$OUTPUT_SUFFIX" \
  --hdf5-path "$HDF5_PATH"

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} OMP_NUM_THREADS=4
torchrun --standalone --nproc_per_node=2 "$TRAIN_SCRIPT" --config "$runtime_config"
