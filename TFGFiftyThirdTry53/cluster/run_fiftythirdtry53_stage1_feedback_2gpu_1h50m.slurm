#!/bin/bash
# Try 53 Stage1 feedback pass - 2 GPUs, 1h50m
#SBATCH --job-name=ckm-t53-s1-feedback-2gpu-110m
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=medium_gpu
#SBATCH --gres=gpu:rtx2080:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=0-01:50:00
#SBATCH --output=logs_train_fiftythirdtry53_pmnet_prior_stage1_feedback_2gpu_110m_%j.out
#SBATCH --error=logs_train_fiftythirdtry53_pmnet_prior_stage1_feedback_2gpu_110m_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/fiftythirdtry53_pmnet_prior_gan_cyclic/fiftythirdtry53_pmnet_prior_stage1_cycle_feedback_resume.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_pmnet_prior_gan.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-t53_stage1_literature_2gpu}
MASTER_PORT=${MASTER_PORT:-29732}
HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_270326.h5}
PRECOMPUTE_PRIOR_CACHE=${PRECOMPUTE_PRIOR_CACHE:-0}
PRECOMPUTE_PRIOR_HDF5=${PRECOMPUTE_PRIOR_HDF5:-0}
PRECOMPUTE_SPLITS=${PRECOMPUTE_SPLITS:-train val}
STAGE2_JSON=${STAGE2_JSON:-outputs/fiftythirdtry53_tail_refiner_stage2_teacher_literature_cyclic_2gpu/validate_metrics_tail_refiner_latest.json}
STAGE3_JSON=${STAGE3_JSON:-outputs/fiftythirdtry53_stage3_nlos_global_context_cyclic_2gpu/validate_metrics_tail_refiner_latest.json}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_t53_s1_feedback_runtime_XXXX.yaml")
feedback_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_t53_s1_feedback_final_XXXX.yaml")
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
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} OMP_NUM_THREADS=8

if [ "$PRECOMPUTE_PRIOR_CACHE" = "1" ]; then
  "$PYTHON_BIN" cluster/precompute_prior_cache.py --config "$feedback_config" --splits $PRECOMPUTE_SPLITS
fi

if [ "$PRECOMPUTE_PRIOR_HDF5" = "1" ]; then
  "$PYTHON_BIN" cluster/precompute_prior_hdf5.py --config "$feedback_config" --splits $PRECOMPUTE_SPLITS --overwrite
fi

"$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node=2 --master-port "$MASTER_PORT" \
  "$TRAIN_SCRIPT" --config "$feedback_config"
