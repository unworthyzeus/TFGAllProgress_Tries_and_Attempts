#!/bin/bash
# Try 44 PMNet-v3-style direct path-loss model without physical prior - 1 GPU
#SBATCH --job-name=ckm-t44-pmnetv3-1gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs_train_fortyfourthtry44_pmnet_v3_no_prior_1gpu_%j.out
#SBATCH --error=logs_train_fortyfourthtry44_pmnet_v3_no_prior_1gpu_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/fortyfourthtry44_pmnet_v3_no_prior/fortyfourthtry44_pmnet_v3_no_prior.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_pmnet_direct.py}
PYTHON_BIN=${PYTHON_BIN:-python}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-t44_pmnet_v3_no_prior_1gpu}
HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_270326.h5}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_pmnet_XXXX.yaml")
cleanup() { rm -f "$runtime_config"; }
trap cleanup EXIT

"$PYTHON_BIN" cluster/prepare_runtime_config.py \
  --input-config "$CONFIG_PATH" \
  --output-config "$runtime_config" \
  --output-suffix "$OUTPUT_SUFFIX" \
  --hdf5-path "$HDF5_PATH"

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} OMP_NUM_THREADS=4
"$PYTHON_BIN" "$TRAIN_SCRIPT" --config "$runtime_config"
