#!/bin/bash
# Try 68 — 1 GPU (38G RAM; YAML num_workers=0 for 3 parallel jobs on one node).
#SBATCH --job-name=ckm-t68-1gpu
#SBATCH -A gpu
#SBATCH --partition=gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:1
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=8
#SBATCH --mem=38G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@90
#SBATCH --output=logs_train_try68_1gpu_%x_%j.out
#SBATCH --error=logs_train_try68_1gpu_%x_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/sixtyeighth_try68_experts/try68_expert_open_sparse_lowrise.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_partitioned_pathloss_expert.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
MASTER_PORT=${MASTER_PORT:-29988}
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

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_t68_XXXX.yaml")

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
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
export SLURM_STEP_KILL_TIMEOUT=${SLURM_STEP_KILL_TIMEOUT:-30}

srun --ntasks=1 --cpu-bind=none --kill-on-bad-exit=1 \
  "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" --master-port "$MASTER_PORT" \
  "$TRAIN_SCRIPT" --config "$runtime_config" &
child_pid=$!
wait "$child_pid"
