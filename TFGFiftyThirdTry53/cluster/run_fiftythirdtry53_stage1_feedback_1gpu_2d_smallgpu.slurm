#!/bin/bash
# Try 53 stage1 feedback - 1 GPU, 2 days, intended to run in parallel with long stage2/stage3
#SBATCH --job-name=ckm-t53-s1-feedback-1gpu-2d
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs_train_fiftythirdtry53_stage1_feedback_1gpu_2d_%j.out
#SBATCH --error=logs_train_fiftythirdtry53_stage1_feedback_1gpu_2d_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/fiftythirdtry53_pmnet_prior_gan_cyclic/fiftythirdtry53_pmnet_prior_stage1_cycle_feedback_resume.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_pmnet_prior_gan.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-t53_stage1_literature_1gpu_long}
MASTER_PORT=${MASTER_PORT:-29931}
HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_270326.h5}
STAGE2_JSON=${STAGE2_JSON:-outputs/fiftythirdtry53_tail_refiner_stage2_teacher_literature_cyclic_3gpu_short/validate_metrics_tail_refiner_latest.json}
STAGE3_JSON=${STAGE3_JSON:-outputs/fiftythirdtry53_stage3_nlos_global_context_cyclic_3gpu_short/validate_metrics_tail_refiner_latest.json}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_t53_s1_feedback_runtime_1gpu_XXXX.yaml")
feedback_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_t53_s1_feedback_final_1gpu_XXXX.yaml")
cleanup() { rm -f "$runtime_config" "$feedback_config"; }
trap cleanup EXIT

"$PYTHON_BIN" cluster/prepare_runtime_config.py \
  --input-config "$CONFIG_PATH" \
  --output-config "$runtime_config" \
  --output-suffix "$OUTPUT_SUFFIX" \
  --hdf5-path "$HDF5_PATH"

"$PYTHON_BIN" cluster/prepare_try53_stage1_feedback_config.py \
  --input-config "$runtime_config" \
  --output-config "$feedback_config" \
  --stage2-json "$STAGE2_JSON" \
  --stage3-json "$STAGE3_JSON"

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} OMP_NUM_THREADS=6

"$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node=1 --master-port "$MASTER_PORT" \
  "$TRAIN_SCRIPT" --config "$feedback_config"
