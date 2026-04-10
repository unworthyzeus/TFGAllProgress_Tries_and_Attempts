#!/bin/bash
# Try 55 topology expert - 1 GPU
# Usage via sbatch:
#   sbatch --export=ALL,CONFIG_PATH=experiments/.../expert.yaml \
#     cluster/run_fiftyfifthtry55_partitioned_expert_1gpu.slurm
#SBATCH --job-name=ckm-t55-expert-1gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs_train_fiftyfifthtry55_%x_%j.out
#SBATCH --error=logs_train_fiftyfifthtry55_%x_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/fiftyfifthtry55_partitioned_stage1/fiftyfifthtry55_expert_open_sparse_lowrise.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_partitioned_pathloss_expert.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
MASTER_PORT=${MASTER_PORT:-29901}
HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_270326.h5}
PRECOMPUTE_PRIOR_CACHE=${PRECOMPUTE_PRIOR_CACHE:-0}
PRECOMPUTE_SPLITS=${PRECOMPUTE_SPLITS:-train val}
FORMULA_CACHE_DIR=${FORMULA_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/ckm_t55_prior_cache_${SLURM_JOB_ID}}
FORMULA_PRECOMPUTED_HDF5=${FORMULA_PRECOMPUTED_HDF5:-}
CLEANUP_FORMULA_CACHE=${CLEANUP_FORMULA_CACHE:-1}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

mkdir -p "$FORMULA_CACHE_DIR"
if [ -z "$FORMULA_PRECOMPUTED_HDF5" ]; then
  FORMULA_PRECOMPUTED_HDF5="${FORMULA_CACHE_DIR}/local_formula_cache.h5"
fi

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_t55_expert_1gpu_XXXX.yaml")
cleanup() {
  rm -f "$runtime_config"
  if [ "$CLEANUP_FORMULA_CACHE" = "1" ]; then
    rm -rf "$FORMULA_CACHE_DIR"
  fi
}
trap cleanup EXIT

prepare_runtime_args=()
if [ -n "$RESUME_CHECKPOINT" ]; then
  prepare_runtime_args+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

"$PYTHON_BIN" cluster/prepare_runtime_config.py \
  --input-config "$CONFIG_PATH" \
  --output-config "$runtime_config" \
  --output-suffix "$OUTPUT_SUFFIX" \
  --hdf5-path "$HDF5_PATH" \
  --formula-cache-dir "$FORMULA_CACHE_DIR" \
  --formula-precomputed-hdf5 "$FORMULA_PRECOMPUTED_HDF5" \
  "${prepare_runtime_args[@]}"

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

if [ "$PRECOMPUTE_PRIOR_CACHE" = "1" ]; then
  "$PYTHON_BIN" cluster/precompute_prior_cache.py --config "$runtime_config" --splits $PRECOMPUTE_SPLITS --output-hdf5 "$FORMULA_PRECOMPUTED_HDF5"
fi

"$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node=1 --master-port "$MASTER_PORT" \
  "$TRAIN_SCRIPT" --config "$runtime_config"
