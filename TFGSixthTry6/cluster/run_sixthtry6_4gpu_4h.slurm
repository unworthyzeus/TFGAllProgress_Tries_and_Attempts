#!/bin/bash
# SixthTry6: 4 GPUs DDP (gpu/big_gpu), 4h walltime
#SBATCH --job-name=ckm-sixthtry6-4gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=big_gpu
#SBATCH --gres=gpu:rtx2080:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs_train_sixthtry6_4gpu_%j.out
#SBATCH --error=logs_train_sixthtry6_4gpu_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_sixthtry6.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_cgan.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-ddp4_sixthtry6}
MASTER_PORT=${MASTER_PORT:-29504}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_ddp_XXXX.yaml")
cleanup() {
    rm -f "$runtime_config"
}
trap cleanup EXIT

DDP_GPU_COUNT_OVERRIDE=4
GPU_COUNT=4

# New shared dataset location on cluster
HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_180326.h5}

prepare_args=(
    --input-config "$CONFIG_PATH"
    --output-config "$runtime_config"
    --output-suffix "$OUTPUT_SUFFIX"
    --hdf5-path "$HDF5_PATH"
)

"$PYTHON_BIN" cluster/prepare_runtime_config.py "${prepare_args[@]}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=4

echo "SixthTry6 DDP (4 GPUs, 4h max, direct MSE pred_db vs target_db in dB + ReduceLROnPlateau)"
echo "Base config: $CONFIG_PATH"
echo "Runtime config: $runtime_config"
echo "World size: $GPU_COUNT"

"$TORCHRUN_BIN" \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="$GPU_COUNT" \
    --master-port "$MASTER_PORT" \
    "$TRAIN_SCRIPT" \
    --config "$runtime_config"

