#!/bin/bash
# Try 53 stage2 - 1 GPU, 2 days, intended to run in parallel with long stage1/stage3
#SBATCH --job-name=ckm-t53-stage2-1gpu-2d
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs_train_fiftythirdtry53_stage2_1gpu_2d_%j.out
#SBATCH --error=logs_train_fiftythirdtry53_stage2_1gpu_2d_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/fiftythirdtry53_pmnet_tail_refiner_cyclic/fiftythirdtry53_pmnet_tail_refiner_stage2_cycle.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_pmnet_tail_refiner.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
MASTER_PORT=${MASTER_PORT:-29941}
STAGE1_SHORT_DIR=${STAGE1_SHORT_DIR:-outputs/fiftythirdtry53_pmnet_prior_stage1_literature_cyclic_t53_stage1_literature_3gpu_short}
STAGE2_LONG_DIR=${STAGE2_LONG_DIR:-outputs/fiftythirdtry53_tail_refiner_stage2_teacher_literature_cyclic_1gpu_long}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_t53_stage2_1gpu_XXXX.yaml")
cleanup() { rm -f "$runtime_config"; }
trap cleanup EXIT

"$PYTHON_BIN" cluster/prepare_try53_refiner_runtime_config.py \
  --input-config "$CONFIG_PATH" \
  --output-config "$runtime_config" \
  --output-dir "$STAGE2_LONG_DIR" \
  --stage1-checkpoint "$STAGE1_SHORT_DIR/best_cgan.pt" \
  --val-batch-size 1

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} OMP_NUM_THREADS=6

"$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node=1 --master-port "$MASTER_PORT" \
  "$TRAIN_SCRIPT" --config "$runtime_config"
