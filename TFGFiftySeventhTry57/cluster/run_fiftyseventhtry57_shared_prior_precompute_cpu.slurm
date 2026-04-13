#!/bin/bash
# Try 57 shared prior cache precompute - CPU-only, low-resource
# Preserves any existing shared HDF5 and appends missing samples only.
#SBATCH --job-name=ckm-t55-prior-precompute-cpu
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs_train_fiftyseventhtry57_%x_%j.out
#SBATCH --error=logs_train_fiftyseventhtry57_%x_%j.err

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_270326.h5}
SHARED_FORMULA_CACHE_DIR=${SHARED_FORMULA_CACHE_DIR:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/try57_shared_formula_cache}
SHARED_FORMULA_PRECOMPUTED_HDF5=${SHARED_FORMULA_PRECOMPUTED_HDF5:-${SHARED_FORMULA_CACHE_DIR}/try57_shared_formula_cache.h5}
PRECOMPUTE_SPLITS=${PRECOMPUTE_SPLITS:-train val test}
RESET_SHARED_FORMULA_PRECOMPUTED_HDF5=${RESET_SHARED_FORMULA_PRECOMPUTED_HDF5:-0}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

mkdir -p "$SHARED_FORMULA_CACHE_DIR"
if [ "$RESET_SHARED_FORMULA_PRECOMPUTED_HDF5" = "1" ]; then
  rm -f "$SHARED_FORMULA_PRECOMPUTED_HDF5"
fi

shopt -s nullglob
expert_configs=(experiments/fiftyseventhtry57_partitioned_stage1/fiftyseventhtry57_expert_*.yaml)
shopt -u nullglob

for config in "${expert_configs[@]}"; do
  case "$config" in
    *fiftyseventhtry57_expert_registry.yaml)
      continue
      ;;
  esac
  runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_t55_precompute_XXXX.yaml")
  trap 'rm -f "$runtime_config"' EXIT
  "$PYTHON_BIN" cluster/prepare_runtime_config.py \
    --input-config "$config" \
    --output-config "$runtime_config" \
    --hdf5-path "$HDF5_PATH"
  "$PYTHON_BIN" cluster/precompute_prior_cache.py \
    --config "$runtime_config" \
    --splits $PRECOMPUTE_SPLITS \
    --output-hdf5 "$SHARED_FORMULA_PRECOMPUTED_HDF5"
  rm -f "$runtime_config"
  trap - EXIT
done
