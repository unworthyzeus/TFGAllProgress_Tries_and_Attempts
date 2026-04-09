#!/bin/bash
# Try 49 Stage1 export outputs HDF5 - 4 GPU
#SBATCH --job-name=ckm-t49-stage1-export-hdf5-4gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=medium_gpu
#SBATCH --gres=gpu:rtx2080:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs_train_fortyninthtry49_stage1_export_hdf5_4gpu_%j.out
#SBATCH --error=logs_train_fortyninthtry49_stage1_export_hdf5_4gpu_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/fortyninthtry49_pmnet_prior_gan_fastbatch/fortyninthtry49_pmnet_prior_stage1_widen112_resume.yaml}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/scratch/nas/3/gmoreno/TFGpractice/TFGFortyNinthTry49/outputs/fortyninthtry49_pmnet_prior_stage1_t49_stage1_w112_4gpu/best_cgan.pt}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/fortyninthtry49_stage1_outputs_hdf5}
EXPORT_SPLITS=${EXPORT_SPLITS:-train val}
BATCH_SIZE=${BATCH_SIZE:-2}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
MASTER_PORT=${MASTER_PORT:-29649}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} OMP_NUM_THREADS=8
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

"$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node=4 --master-port "$MASTER_PORT" \
  scripts/export_stage1_outputs_hdf5.py \
  --config "$CONFIG_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --splits $EXPORT_SPLITS \
  --batch-size "$BATCH_SIZE" \
  --overwrite
