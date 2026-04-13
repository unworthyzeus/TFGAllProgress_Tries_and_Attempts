#!/bin/bash
# Try 59 topology expert - 1 GPU
#SBATCH --job-name=ckm-t56-expert-1gpu
#SBATCH -A gpu
#SBATCH --partition=gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs_try59_expert_%x_%j.out
#SBATCH --error=logs_try59_expert_%x_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/nas/3/gmoreno/TFGpractice/TFGFiftyNinthTry59}"
CONFIG_PATH="${CONFIG_PATH:-experiments/fiftyninthtry59_topology_experts/fiftyninthtry59_expert_open_sparse_lowrise.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
MASTER_PORT="${MASTER_PORT:-30001}"
RUNTIME_CONFIG="$(mktemp "${SLURM_TMPDIR:-/tmp}/try59_runtime_1gpu_XXXX.yaml")"

cleanup() {
  rm -f "$RUNTIME_CONFIG"
}
trap cleanup EXIT

source /scratch/nas/3/gmoreno/tf_env/bin/activate
cd "$ROOT_DIR"

"$PYTHON_BIN" cluster/prepare_runtime_config.py \
  --input-config "$CONFIG_PATH" \
  --output-config "$RUNTIME_CONFIG" \
  --output-suffix "$OUTPUT_SUFFIX"

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

torchrun --standalone --nnodes=1 --nproc_per_node=1 --master-port "$MASTER_PORT" \
  train_topology_expert.py --config "$RUNTIME_CONFIG"
