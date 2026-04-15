#!/bin/bash
# Try 71 — 3×GPU, QoS medium_gpu (recommended for exactly 3×RTX2080 on SERT).
#
# Why NOT big_gpu for this script:
#   If squeue shows (QOSMinGRES), the big_gpu QoS requires a *minimum* GPU count (often 4).
#   A 3-GPU job is below that minimum → use medium_gpu here, or submit 4 GPUs with big_gpu.
#
# Other PD reason you may see with medium_gpu:
#   (MaxGRESPerAccount) — your Slurm account hit its cap on total concurrent GPUs across
#   all your jobs; cancel other GPU jobs or wait until usage drops.
#
# Dataloader: experiments/dataloader_batch_presets/presets/3gpu_medium.yml (same recipe as 4-GPU).
#
# Usage:
#   sbatch --export=ALL,CONFIG_PATH=experiments/seventyfirst_try71_experts/try71_expert_....yaml,TRAIN_SCRIPT=train_partitioned_pathloss_expert.py \
#     cluster/run_seventyfirst_try71_3gpu_medium_gpu.slurm
#
# Optional env (defaults shown):
#   NPROC_PER_NODE=3  MASTER_PORT=30003  VENV_PATH=/scratch/nas/3/gmoreno/tf_env/bin/activate
#SBATCH --job-name=ckm-t68-3gpu-med
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=medium_gpu
#SBATCH --gres=gpu:rtx2080:3
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=0-04:00:00
#SBATCH --signal=B:TERM@90
#SBATCH --output=logs_train_try71_3gpu_med_%x_%j.out
#SBATCH --error=logs_train_try71_3gpu_med_%x_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/seventyfirst_try71_experts/try71_expert_open_sparse_lowrise.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_partitioned_pathloss_expert.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
NPROC_PER_NODE=${NPROC_PER_NODE:-3}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
MASTER_PORT=${MASTER_PORT:-30003}
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

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_t68_3gpu_XXXX.yaml")

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
