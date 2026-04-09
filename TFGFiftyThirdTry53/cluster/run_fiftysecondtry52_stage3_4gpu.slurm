#!/bin/bash
# Try 52 stage3 NLoS-only global-context refiner - 4 GPUs
#SBATCH --job-name=ckm-t52-stage3-nlos-4gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=medium_gpu
#SBATCH --gres=gpu:rtx2080:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs_train_fiftysecondtry52_stage3_nlos_global_context_4gpu_%j.out
#SBATCH --error=logs_train_fiftysecondtry52_stage3_nlos_global_context_4gpu_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/fiftysecondtry52_pmnet_tail_refiner_fastbatch/fiftysecondtry52_pmnet_stage3_nlos_global_context.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_pmnet_tail_refiner.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
MASTER_PORT=${MASTER_PORT:-29651}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} OMP_NUM_THREADS=12

"$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node=4 --master-port "$MASTER_PORT" \
  "$TRAIN_SCRIPT" --config "$CONFIG_PATH"
