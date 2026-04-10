#!/bin/bash
# Try 54 topology expert - 4 GPUs
# Usage via sbatch:
#   sbatch --export=ALL,CONFIG_PATH=experiments/.../expert.yaml,OUTPUT_SUFFIX=t54_dense_block_highrise \
#     cluster/run_fiftyfourthtry54_partitioned_expert_4gpu.slurm
#SBATCH --job-name=ckm-t54-expert-4gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=medium_gpu
#SBATCH --gres=gpu:rtx2080:4
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs_train_fiftyfourthtry54_%x_%j.out
#SBATCH --error=logs_train_fiftyfourthtry54_%x_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/fiftyfourthtry54_partitioned_stage1/fiftyfourthtry54_expert_open_sparse_lowrise.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_partitioned_pathloss_expert.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-}
MASTER_PORT=${MASTER_PORT:-29841}
HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_270326.h5}
PRECOMPUTE_PRIOR_CACHE=${PRECOMPUTE_PRIOR_CACHE:-0}
PRECOMPUTE_SPLITS=${PRECOMPUTE_SPLITS:-train val}
FORMULA_CACHE_DIR=${FORMULA_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/ckm_t54_prior_cache_${SLURM_JOB_ID}}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

mkdir -p "$FORMULA_CACHE_DIR"

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_t54_expert_XXXX.yaml")
cleanup() {
  rm -f "$runtime_config"
  rm -rf "$FORMULA_CACHE_DIR"
}
trap cleanup EXIT

"$PYTHON_BIN" cluster/prepare_runtime_config.py \
  --input-config "$CONFIG_PATH" \
  --output-config "$runtime_config" \
  --output-suffix "$OUTPUT_SUFFIX" \
  --hdf5-path "$HDF5_PATH" \
  --formula-cache-dir "$FORMULA_CACHE_DIR"

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

if [ "$PRECOMPUTE_PRIOR_CACHE" = "1" ]; then
  "$PYTHON_BIN" cluster/precompute_prior_cache.py --config "$runtime_config" --splits $PRECOMPUTE_SPLITS
fi

"$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node=4 --master-port "$MASTER_PORT" \
  "$TRAIN_SCRIPT" --config "$runtime_config"
