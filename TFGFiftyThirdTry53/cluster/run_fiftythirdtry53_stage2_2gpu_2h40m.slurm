#!/bin/bash
# Try 53 stage2 tail refiner - 2 GPUs, 2h40m
#SBATCH --job-name=ckm-t53-stage2-2gpu-160m
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=medium_gpu
#SBATCH --gres=gpu:rtx2080:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=0-02:40:00
#SBATCH --output=logs_train_fiftythirdtry53_tail_refiner_stage2_2gpu_160m_%j.out
#SBATCH --error=logs_train_fiftythirdtry53_tail_refiner_stage2_2gpu_160m_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/fiftythirdtry53_pmnet_tail_refiner_cyclic/fiftythirdtry53_pmnet_tail_refiner_stage2_cycle.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_pmnet_tail_refiner.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
MASTER_PORT=${MASTER_PORT:-29741}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} OMP_NUM_THREADS=8

"$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node=2 --master-port "$MASTER_PORT" \
  "$TRAIN_SCRIPT" --config "$CONFIG_PATH"
