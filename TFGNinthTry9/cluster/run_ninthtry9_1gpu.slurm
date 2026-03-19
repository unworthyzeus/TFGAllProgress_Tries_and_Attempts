#!/bin/bash
# NinthTry9: 1 GPU (torchrun --nproc-per-node=1).
#SBATCH --job-name=ckm-ninthtry9-1gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=medium_gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-16:00:00
#SBATCH --output=logs_train_ninthtry9_ddp_1gpu_%j.out
#SBATCH --error=logs_train_ninthtry9_ddp_1gpu_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_ninthtry9.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_cgan.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-ddp1_ninthtry9}
MASTER_PORT=${MASTER_PORT:-29518}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_ddp_XXXX.yaml")
cleanup() {
    rm -f "$runtime_config"
}
trap cleanup EXIT

GPU_COUNT=1
HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_180326.h5}
SCALAR_CSV_PATH=${SCALAR_CSV_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_180326_antenna_height.csv}

prepare_args=(
    --input-config "$CONFIG_PATH"
    --output-config "$runtime_config"
    --output-suffix "$OUTPUT_SUFFIX"
    --hdf5-path "$HDF5_PATH"
    --scalar-csv-path "$SCALAR_CSV_PATH"
)

"$PYTHON_BIN" cluster/prepare_runtime_config.py "${prepare_args[@]}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=4

"$TORCHRUN_BIN" \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="$GPU_COUNT" \
    --master-port "$MASTER_PORT" \
    "$TRAIN_SCRIPT" \
    --config "$runtime_config"
