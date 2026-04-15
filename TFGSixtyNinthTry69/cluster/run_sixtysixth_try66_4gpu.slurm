#!/bin/bash
# Try 66 generic 4-GPU launcher
# Usage via sbatch:
#   sbatch --export=ALL,CONFIG_PATH=experiments/.../config.yaml,TRAIN_SCRIPT=train_partitioned_pathloss_expert.py \
#     cluster/run_sixtysixth_try66_4gpu.slurm
#SBATCH --job-name=ckm-t66-4gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=medium_gpu
#SBATCH --gres=gpu:rtx2080:4
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=0-04:00:00
#SBATCH --signal=B:TERM@90
#SBATCH --output=logs_train_try66_%x_%j.out
#SBATCH --error=logs_train_try66_%x_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/sixtysixth_try66_stage1/try66_expert_open_sparse_lowrise.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_partitioned_pathloss_expert.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
MASTER_PORT=${MASTER_PORT:-29976}
HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_270326.h5}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

runtime_config=""
child_pid=""
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ -n "${child_pid:-}" ] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
    sleep 5
    kill -KILL "$child_pid" 2>/dev/null || true
  fi
  if [ -n "${runtime_config:-}" ]; then
    rm -f "$runtime_config"
  fi
  exit $rc
}
trap cleanup EXIT INT TERM

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_t66_XXXX.yaml")

prepare_runtime_args=()
if [ -n "$RESUME_CHECKPOINT" ]; then
  prepare_runtime_args+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

"$PYTHON_BIN" cluster/prepare_runtime_config.py \
  --input-config "$CONFIG_PATH" \
  --output-config "$runtime_config" \
  --output-suffix "$OUTPUT_SUFFIX" \
  --hdf5-path "$HDF5_PATH" \
  "${prepare_runtime_args[@]}"

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export SLURM_STEP_KILL_TIMEOUT=${SLURM_STEP_KILL_TIMEOUT:-30}

srun --ntasks=1 --cpu-bind=none --kill-on-bad-exit=1 \
  "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" --master-port "$MASTER_PORT" \
  "$TRAIN_SCRIPT" --config "$runtime_config" &
child_pid=$!
wait "$child_pid"
