#!/bin/bash
#SBATCH --job-name=ckm-hdf5-cgan-ddp
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=big_gpu
#SBATCH --gres=gpu:rtx2080:5
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs_train_cgan_hdf5_ddp_%j.out
#SBATCH --error=logs_train_cgan_hdf5_ddp_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max256_blend.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_cgan.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-}
MASTER_PORT=${MASTER_PORT:-29500}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_ddp_XXXX.yaml")
cleanup() {
    rm -f "$runtime_config"
}
trap cleanup EXIT

if [[ -n "${DDP_GPU_COUNT_OVERRIDE:-}" ]]; then
    GPU_COUNT="$DDP_GPU_COUNT_OVERRIDE"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    GPU_COUNT="$SLURM_GPUS_ON_NODE"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    GPU_COUNT="${#GPU_IDS[@]}"
else
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
fi

if [[ -z "$GPU_COUNT" || "$GPU_COUNT" -lt 1 ]]; then
    echo "No visible GPUs found for this allocation."
    exit 1
fi

prepare_args=(
    --input-config "$CONFIG_PATH"
    --output-config "$runtime_config"
)
if [[ -n "$OUTPUT_SUFFIX" ]]; then
    prepare_args+=(--output-suffix "$OUTPUT_SUFFIX")
fi

"$PYTHON_BIN" cluster/prepare_runtime_config.py "${prepare_args[@]}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
    CPUS_PER_RANK=$(( SLURM_CPUS_PER_TASK / GPU_COUNT ))
    if [[ "$CPUS_PER_RANK" -lt 1 ]]; then
        CPUS_PER_RANK=1
    fi
    export OMP_NUM_THREADS="$CPUS_PER_RANK"
fi

echo "Launching single DDP run"
echo "Base config: $CONFIG_PATH"
echo "Runtime config: $runtime_config"
echo "Train script: $TRAIN_SCRIPT"
echo "World size: $GPU_COUNT"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"

"$TORCHRUN_BIN" \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="$GPU_COUNT" \
    --master-port "$MASTER_PORT" \
    "$TRAIN_SCRIPT" \
    --config "$runtime_config"
