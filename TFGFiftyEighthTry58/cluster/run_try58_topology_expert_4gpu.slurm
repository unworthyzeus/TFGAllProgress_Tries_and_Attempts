#!/bin/bash
#SBATCH --job-name=ckm-t56-expert
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx2080:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs_try58_expert_%j.out
#SBATCH --error=logs_try58_expert_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/nas/3/gmoreno/TFGpractice/TFGFiftyEighthTry58}"
CONFIG_PATH="${CONFIG_PATH:-experiments/fiftyeighthtry58_topology_experts/fiftyeighthtry58_expert_open_sparse_lowrise.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-cluster}"
RUNTIME_CONFIG="$(mktemp "${SLURM_TMPDIR:-/tmp}/try58_runtime_XXXX.yaml")"

cleanup() {
  rm -f "$RUNTIME_CONFIG"
}
trap cleanup EXIT

cd "$ROOT_DIR"

"$PYTHON_BIN" cluster/prepare_runtime_config.py \
  --input-config "$CONFIG_PATH" \
  --output-config "$RUNTIME_CONFIG" \
  --output-suffix "$OUTPUT_SUFFIX"

torchrun --standalone --nproc_per_node=4 train_topology_expert.py --config "$RUNTIME_CONFIG"
