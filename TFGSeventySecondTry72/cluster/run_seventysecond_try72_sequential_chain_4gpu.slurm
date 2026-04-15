#!/bin/bash
# Run all 6 Try 72 topology experts sequentially on ONE 4-GPU allocation.
# Avoids MaxGRESPerAccount from submitting many pending 4-GPU + cleanup jobs at once.
#
#SBATCH --job-name=t72-chain-6exp
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=big_gpu
#SBATCH --gres=gpu:rtx2080:4
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=0-04:00:00
#SBATCH --signal=B:TERM@90
#SBATCH --output=logs_train_try72_chain_%j.out
#SBATCH --error=logs_train_try72_chain_%j.err

set -euo pipefail

TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_partitioned_pathloss_expert.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_270326.h5}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export SLURM_STEP_KILL_TIMEOUT=${SLURM_STEP_KILL_TIMEOUT:-30}

run_one_expert() {
  local CONFIG_PATH="$1"
  local MASTER_PORT="$2"
  local name child_pid rc=0
  name="$(basename "$CONFIG_PATH" .yaml)"
  echo "========== Try72 chain: ${name} (port ${MASTER_PORT}) =========="

  local runtime_config
  runtime_config="$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_t72_${name}_XXXX.yaml")"

  local prepare_runtime_args=()
  if [ -n "$RESUME_CHECKPOINT" ]; then
    prepare_runtime_args+=(--resume-checkpoint "$RESUME_CHECKPOINT")
  fi

  "$PYTHON_BIN" cluster/prepare_runtime_config.py \
    --input-config "$CONFIG_PATH" \
    --output-config "$runtime_config" \
    --output-suffix "$OUTPUT_SUFFIX" \
    --hdf5-path "$HDF5_PATH" \
    "${prepare_runtime_args[@]}"

  srun --ntasks=1 --cpu-bind=none --kill-on-bad-exit=1 \
    "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" --master-port "$MASTER_PORT" \
    "$TRAIN_SCRIPT" --config "$runtime_config" &
  child_pid=$!
  wait "$child_pid" || rc=$?
  rm -f "$runtime_config"
  if [ "$rc" -ne 0 ]; then
    echo "[chain] ERROR ${name} exited with ${rc}" >&2
    exit "$rc"
  fi
  echo "[chain] finished ${name}"
}

between_experts_cleanup() {
  echo "[chain] between-experts cleanup"
  nvidia-smi || true
  pkill -9 -u "${USER}" -f 'train_partitioned_pathloss_expert.py' || true
  pkill -9 -u "${USER}" -f 'torchrun.*TFGSeventySecondTry72' || true
  sleep 5
  nvidia-smi || true
}

# config.yaml:master_port (must match _relaunch_try72.py base_port order)
run_one_expert "experiments/seventysecond_try72_experts/try72_expert_open_sparse_lowrise.yaml" 30166
between_experts_cleanup
run_one_expert "experiments/seventysecond_try72_experts/try72_expert_open_sparse_vertical.yaml" 30167
between_experts_cleanup
run_one_expert "experiments/seventysecond_try72_experts/try72_expert_mixed_compact_lowrise.yaml" 30168
between_experts_cleanup
run_one_expert "experiments/seventysecond_try72_experts/try72_expert_mixed_compact_midrise.yaml" 30169
between_experts_cleanup
run_one_expert "experiments/seventysecond_try72_experts/try72_expert_dense_block_midrise.yaml" 30170
between_experts_cleanup
run_one_expert "experiments/seventysecond_try72_experts/try72_expert_dense_block_highrise.yaml" 30171

echo "========== Try72 chain: all 6 experts completed =========="
