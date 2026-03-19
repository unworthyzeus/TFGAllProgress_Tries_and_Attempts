#!/bin/bash
# FourthTry4: fused-map-in-dB loss, 2 GPUs DDP (medium_gpu)
#SBATCH --job-name=ckm-fourthtry4-2gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=medium_gpu
#SBATCH --gres=gpu:rtx2080:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-16:00:00
#SBATCH --output=logs_train_fourthtry4_ddp_%j.out
#SBATCH --error=logs_train_fourthtry4_ddp_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_fourthtry4.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_cgan.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-ddp2_fourthtry4}
MASTER_PORT=${MASTER_PORT:-29501}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_ddp_XXXX.yaml")
cleanup() {
    rm -f "$runtime_config"
}
trap cleanup EXIT

DDP_GPU_COUNT_OVERRIDE=2
GPU_COUNT=2

prepare_args=(
    --input-config "$CONFIG_PATH"
    --output-config "$runtime_config"
    --output-suffix "$OUTPUT_SUFFIX"
)

"$PYTHON_BIN" cluster/prepare_runtime_config.py "${prepare_args[@]}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=4

echo "FourthTry4 DDP (fused-map-in-dB loss)"
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
