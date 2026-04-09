#!/bin/bash
# Try 51 Stage1 literature-aligned resume - 4 GPUs
#SBATCH --job-name=ckm-t51-s1-w112-resume-4gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=medium_gpu
#SBATCH --gres=gpu:rtx2080:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs_train_fiftyfirsttry51_pmnet_prior_stage1_w112_resume_4gpu_%j.out
#SBATCH --error=logs_train_fiftyfirsttry51_pmnet_prior_stage1_w112_resume_4gpu_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/fiftyfirsttry51_pmnet_prior_gan_fastbatch/fiftyfirsttry51_pmnet_prior_stage1_widen112_resume_literature.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_pmnet_prior_gan.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-t51_stage1_w112_4gpu}
MASTER_PORT=${MASTER_PORT:-29632}
HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_270326.h5}
PRECOMPUTE_PRIOR_CACHE=${PRECOMPUTE_PRIOR_CACHE:-0}
PRECOMPUTE_PRIOR_HDF5=${PRECOMPUTE_PRIOR_HDF5:-0}
PRECOMPUTE_SPLITS=${PRECOMPUTE_SPLITS:-train val}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_t51_s1_w112_resume_XXXX.yaml")
cleanup() { rm -f "$runtime_config"; }
trap cleanup EXIT

"$PYTHON_BIN" cluster/prepare_runtime_config.py \
  --input-config "$CONFIG_PATH" \
  --output-config "$runtime_config" \
  --output-suffix "$OUTPUT_SUFFIX" \
  --hdf5-path "$HDF5_PATH"

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} OMP_NUM_THREADS=12

if [ "$PRECOMPUTE_PRIOR_CACHE" = "1" ]; then
  "$PYTHON_BIN" cluster/precompute_prior_cache.py --config "$runtime_config" --splits $PRECOMPUTE_SPLITS
fi

if [ "$PRECOMPUTE_PRIOR_HDF5" = "1" ]; then
  "$PYTHON_BIN" cluster/precompute_prior_hdf5.py --config "$runtime_config" --splits $PRECOMPUTE_SPLITS --overwrite
fi

"$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node=4 --master-port "$MASTER_PORT" \
  "$TRAIN_SCRIPT" --config "$runtime_config"
