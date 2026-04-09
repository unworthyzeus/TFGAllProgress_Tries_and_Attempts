#!/bin/bash
# Try 49 stage2 checkpoint shrink 96->84 channels
#SBATCH --job-name=ckm-t49-tail-refiner-shrink84
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs_shrink_fortyninthtry49_tail_refiner_stage2_84_%j.out
#SBATCH --error=logs_shrink_fortyninthtry49_tail_refiner_stage2_84_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/fortyninthtry49_pmnet_tail_refiner_fastbatch/fortyninthtry49_pmnet_tail_refiner_stage2.yaml}
PYTHON_BIN=${PYTHON_BIN:-python}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/fortyninthtry49_tail_refiner_stage2_4gpu}
SRC_CKPT=${SRC_CKPT:-$OUTPUT_DIR/best_tail_refiner.pt}
BACKUP_CKPT=${BACKUP_CKPT:-$OUTPUT_DIR/best_tail_refiner_96ch_backup.pt}
TMP_CKPT=${TMP_CKPT:-$OUTPUT_DIR/best_tail_refiner_84_tmp.pt}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

if [ ! -f "$SRC_CKPT" ]; then
  echo "Missing source checkpoint: $SRC_CKPT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$BACKUP_CKPT" ]; then
  cp "$SRC_CKPT" "$BACKUP_CKPT"
fi

"$PYTHON_BIN" scripts/widen_pmnet_checkpoint.py \
  --dst-config "$CONFIG_PATH" \
  --src-ckpt "$SRC_CKPT" \
  --out-ckpt "$TMP_CKPT" \
  --model-kind tail_refiner

mv "$TMP_CKPT" "$SRC_CKPT"
echo "[ok] replaced $SRC_CKPT with 84-channel adapted checkpoint"
