#!/bin/bash
#SBATCH --job-name=ckm-hdf5-multigpu
#SBATCH -p medium_gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs_multigpu_hdf5_%j.out
#SBATCH --error=logs_multigpu_hdf5_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-configs/cgan_unet_hdf5_cuda_max.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_cgan.py}
PYTHON_BIN=${PYTHON_BIN:-python}
VENV_PATH=${VENV_PATH:-/scratch/nas/g/gmoreno/tf_env/bin/activate}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

declare -a GPU_IDS
declare -a RUNTIME_CONFIGS
declare -a PIDS

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi

if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    echo "No visible GPUs found for this allocation."
    exit 1
fi

cleanup() {
    for runtime_config in "${RUNTIME_CONFIGS[@]:-}"; do
        rm -f "$runtime_config"
    done
}
trap cleanup EXIT

echo "Launching ${#GPU_IDS[@]} parallel single-GPU runs"
echo "Base config: $CONFIG_PATH"
echo "Train script: $TRAIN_SCRIPT"

for worker_idx in "${!GPU_IDS[@]}"; do
    gpu_id="${GPU_IDS[$worker_idx]}"
    runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_multigpu_${worker_idx}_XXXX.yaml")
    RUNTIME_CONFIGS+=("$runtime_config")

    "$PYTHON_BIN" cluster/prepare_runtime_config.py \
        --input-config "$CONFIG_PATH" \
        --output-config "$runtime_config" \
        --output-suffix "gpu${worker_idx}" \
        --seed-offset "$worker_idx"

    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        export PYTHONUNBUFFERED=1
        echo "[worker ${worker_idx}] physical_gpu=${gpu_id} config=${runtime_config}"
        "$PYTHON_BIN" "$TRAIN_SCRIPT" --config "$runtime_config"
    ) > "logs_multigpu_${SLURM_JOB_ID}_${worker_idx}.out" 2> "logs_multigpu_${SLURM_JOB_ID}_${worker_idx}.err" &

    PIDS+=("$!")
done

status=0
for worker_idx in "${!PIDS[@]}"; do
    pid="${PIDS[$worker_idx]}"
    if ! wait "$pid"; then
        echo "worker ${worker_idx} failed"
        status=1
    fi
done

exit "$status"