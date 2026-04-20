#!/bin/bash
# Try 77 1-GPU launcher (adapted from Try 76 — only SLURM scaffolding re-used).
# Usage via sbatch:
#   sbatch --export=ALL,CONFIG_PATH=experiments/seventyseventh_try77_experts/try77_expert_open_sparse_lowrise_delay_spread.yaml,TRAIN_SCRIPT=train_try77.py \
#     cluster/run_seventyseventh_try77_1gpu.slurm
#SBATCH --job-name=ckm-t77-1gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=10
#SBATCH --mem=36G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@90
#SBATCH --output=logs_train_try77_%x_%j.out
#SBATCH --error=logs_train_try77_%x_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/seventyseventh_try77_experts/try77_expert_open_sparse_lowrise_delay_spread.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_try77.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
WIPE_OUTPUTS=${WIPE_OUTPUTS:-0}
MASTER_PORT=${MASTER_PORT:-31788}
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

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_t77_XXXX.yaml")

if [ "$WIPE_OUTPUTS" = "1" ]; then
  echo "[wipe] WIPE_OUTPUTS=1 -> forcing a fresh run (resume disabled)"
  RESUME_CHECKPOINT=""
fi

"$PYTHON_BIN" cluster/prepare_runtime_config.py \
  --input-config "$CONFIG_PATH" \
  --output-config "$runtime_config" \
  --hdf5-path "$HDF5_PATH" \
  ${RESUME_CHECKPOINT:+--resume-checkpoint "$RESUME_CHECKPOINT"}

if [ "$WIPE_OUTPUTS" = "1" ]; then
  output_dir=$("$PYTHON_BIN" - <<'PY' "$runtime_config"
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(str(cfg.get("runtime", {}).get("output_dir", "")))
PY
)
  if [ -n "${output_dir:-}" ]; then
    echo "[wipe] Removing output dir: $output_dir"
    rm -rf -- "$output_dir"
  fi
fi

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export SLURM_STEP_KILL_TIMEOUT=${SLURM_STEP_KILL_TIMEOUT:-30}

srun --ntasks=1 --cpu-bind=none --kill-on-bad-exit=1 \
  "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" --master-port "$MASTER_PORT" \
  "$TRAIN_SCRIPT" --config "$runtime_config" &
child_pid=$!
wait "$child_pid"
