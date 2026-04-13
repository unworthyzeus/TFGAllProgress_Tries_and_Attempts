#!/bin/bash
# Try 57 shared prior cache precompute - parallel 4 GPU reservation
# Usage via sbatch:
#   sbatch --export=ALL,SHARED_FORMULA_CACHE_DIR=/scratch/.../prior_cache/try57_shared cluster/run_fiftyseventhtry57_shared_prior_precompute_4gpu.slurm
# Notes:
#   - The prior itself is still computed in Python/HDF5 code, not by a neural net.
#   - We reserve 4 GPUs so Slurm gives us the same lane as the expert jobs, then run
#     4 parallel workers to precompute different expert partitions concurrently.
#   - Each worker writes a partial HDF5; the job merges them into one shared HDF5 at the end.
#SBATCH --job-name=ckm-t55-prior-precompute
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=medium_gpu
#SBATCH --gres=gpu:rtx2080:4
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs_train_fiftyseventhtry57_%x_%j.out
#SBATCH --error=logs_train_fiftyseventhtry57_%x_%j.err

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
VENV_PATH=${VENV_PATH:-/scratch/nas/3/gmoreno/tf_env/bin/activate}
HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_270326.h5}
SHARED_FORMULA_CACHE_DIR=${SHARED_FORMULA_CACHE_DIR:-/scratch/nas/3/gmoreno/TFGpractice/TFGFiftySeventhTry57/prior_cache/try57_shared_formula_cache}
SHARED_FORMULA_PRECOMPUTED_HDF5=${SHARED_FORMULA_PRECOMPUTED_HDF5:-${SHARED_FORMULA_CACHE_DIR}/try57_shared_formula_cache.h5}
PRECOMPUTE_SPLITS=${PRECOMPUTE_SPLITS:-train val}
WORKERS=${WORKERS:-4}

source "$VENV_PATH"
cd "$SLURM_SUBMIT_DIR"

mkdir -p "$SHARED_FORMULA_CACHE_DIR"
rm -f "$SHARED_FORMULA_PRECOMPUTED_HDF5"

shopt -s nullglob
CONFIGS=(experiments/fiftyseventhtry57_partitioned_stage1/fiftyseventhtry57_expert_*.yaml)
shopt -u nullglob

run_worker() {
  local worker_id="$1"
  local partial_hdf5="${SHARED_FORMULA_CACHE_DIR}/try57_shared_formula_cache.part${worker_id}.h5"
  rm -f "$partial_hdf5"
  local idx=0
  for config in "${CONFIGS[@]}"; do
    case "$config" in
      *fiftyseventhtry57_expert_registry.yaml)
        continue
        ;;
    esac
    if (( idx % WORKERS != worker_id )); then
      idx=$((idx + 1))
      continue
    fi
    local runtime_config
    runtime_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/ckm_t55_precompute_${worker_id}_XXXX.yaml")
    "$PYTHON_BIN" cluster/prepare_runtime_config.py \
      --input-config "$config" \
      --output-config "$runtime_config" \
      --hdf5-path "$HDF5_PATH"
    "$PYTHON_BIN" cluster/precompute_prior_cache.py \
      --config "$runtime_config" \
      --splits $PRECOMPUTE_SPLITS \
      --output-hdf5 "$partial_hdf5"
    rm -f "$runtime_config"
    idx=$((idx + 1))
  done
}

for worker_id in $(seq 0 $((WORKERS - 1))); do
  CUDA_VISIBLE_DEVICES="$worker_id" run_worker "$worker_id" &
done
wait

"$PYTHON_BIN" - <<PY
from __future__ import annotations
import h5py
from pathlib import Path

cache_dir = Path("${SHARED_FORMULA_CACHE_DIR}")
output_path = Path("${SHARED_FORMULA_PRECOMPUTED_HDF5}")
part_paths = sorted(cache_dir.glob("try57_shared_formula_cache.part*.h5"))

with h5py.File(output_path, "w") as out:
    out.attrs["format"] = "try57_formula_prior_cache"
    out.attrs["cache_dtype"] = "uint8"
    out.attrs["compression"] = "gzip"
    out.attrs["compression_level"] = 4
    for part_path in part_paths:
        with h5py.File(part_path, "r") as src:
            for key, value in src.attrs.items():
                if key not in out.attrs:
                    out.attrs[key] = value
            for city in src.keys():
                if city not in out:
                    src.copy(city, out, name=city)
                else:
                    dst_city = out[city]
                    src_city = src[city]
                    for sample in src_city.keys():
                        if sample not in dst_city:
                            src_city.copy(sample, dst_city, name=sample)

for part_path in part_paths:
    part_path.unlink(missing_ok=True)

print({"merged_prior_hdf5": str(output_path), "parts": len(part_paths)})
PY
