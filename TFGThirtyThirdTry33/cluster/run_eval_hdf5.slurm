#!/bin/bash
#SBATCH --job-name=ckm-hdf5-eval
#SBATCH -p small_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs_eval_hdf5_%j.out
#SBATCH --error=logs_eval_hdf5_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-configs/baseline_hdf5.yaml}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-outputs/baseline_hdf5_run/best.pt}
PYTHON_BIN=${PYTHON_BIN:-python}
VENV_PATH=${VENV_PATH:-/scratch/nas/g/gmoreno/tf_env/bin/activate}
RUNTIME_CONFIG=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_eval_hdf5_XXXX.yaml")

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"
trap 'rm -f "$RUNTIME_CONFIG"' EXIT

echo "Running evaluation with config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
$PYTHON_BIN cluster/prepare_runtime_config.py --input-config "$CONFIG_PATH" --output-config "$RUNTIME_CONFIG"
$PYTHON_BIN evaluate.py --config "$RUNTIME_CONFIG" --checkpoint "$CHECKPOINT_PATH"