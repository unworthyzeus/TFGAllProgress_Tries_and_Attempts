#!/bin/bash
# Try 53 Stage1 bootstrap from Try51 literature checkpoint - 1 GPU probe
#SBATCH --job-name=ckm-t53-s1-bootstrap-1gpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs_train_fiftythirdtry53_pmnet_prior_stage1_bootstrap_1gpu_%j.out
#SBATCH --error=logs_train_fiftythirdtry53_pmnet_prior_stage1_bootstrap_1gpu_%j.err

set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-experiments/fiftythirdtry53_pmnet_prior_gan_cyclic/fiftythirdtry53_pmnet_prior_stage1_cycle0_bootstrap.yaml}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_pmnet_prior_gan.py}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-t53_stage1_literature_1gpu_probe}
MASTER_PORT=${MASTER_PORT:-29961}
HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_270326.h5}
SOURCE_CKPT=${SOURCE_CKPT:-/scratch/nas/3/gmoreno/TFGpractice/TFGFiftyFirstTry51/outputs/fiftyfirsttry51_pmnet_prior_stage1_literature_t51_stage1_w112_4gpu/best_cgan.pt}
REDUCED_BASE_CKPT=${REDUCED_BASE_CKPT:-outputs/fiftythirdtry53_pmnet_prior_stage1_literature_cyclic_t53_stage1_literature_1gpu_probe/reduced_try51_best_to_literature112.pt}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

mkdir -p "$(dirname "$REDUCED_BASE_CKPT")"

"$PYTHON_BIN" scripts/widen_pmnet_checkpoint.py \
  --dst-config "$CONFIG_PATH" \
  --src-ckpt "$SOURCE_CKPT" \
  --out-ckpt "$REDUCED_BASE_CKPT"

runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_t53_s1_bootstrap_1gpu_XXXX.yaml")
cleanup() { rm -f "$runtime_config"; }
trap cleanup EXIT

"$PYTHON_BIN" cluster/prepare_runtime_config.py \
  --input-config "$CONFIG_PATH" \
  --output-config "$runtime_config" \
  --output-suffix "$OUTPUT_SUFFIX" \
  --hdf5-path "$HDF5_PATH"

export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} OMP_NUM_THREADS=6

"$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node=1 --master-port "$MASTER_PORT" \
  "$TRAIN_SCRIPT" --config "$runtime_config"
