#!/usr/bin/env bash
set -euo pipefail

TRY_ROOT=${TRY_ROOT:-/workspace/TFGpractice/TFGFiftyThirdTry53}
REPO_ROOT=${REPO_ROOT:-/workspace/TFGpractice}
PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
DATASET_PATH=${DATASET_PATH:-/workspace/TFGpractice/Datasets/CKM_Dataset_270326.h5}
CHAIN_REPEATS=${CHAIN_REPEATS:-1}
START_FROM_SCRATCH=${START_FROM_SCRATCH:-1}
RUNPOD_STAGE1_BATCH_SIZE=${RUNPOD_STAGE1_BATCH_SIZE:-3}
RUNPOD_STAGE1_VAL_BATCH_SIZE=${RUNPOD_STAGE1_VAL_BATCH_SIZE:-3}
RUNPOD_STAGE23_BATCH_SIZE=${RUNPOD_STAGE23_BATCH_SIZE:-3}
RUNPOD_STAGE23_VAL_BATCH_SIZE=${RUNPOD_STAGE23_VAL_BATCH_SIZE:-3}
MASTER_PORT_STAGE1_BOOTSTRAP=${MASTER_PORT_STAGE1_BOOTSTRAP:-29731}
MASTER_PORT_STAGE2=${MASTER_PORT_STAGE2:-29741}
MASTER_PORT_STAGE3=${MASTER_PORT_STAGE3:-29751}
MASTER_PORT_STAGE1_FEEDBACK=${MASTER_PORT_STAGE1_FEEDBACK:-29732}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-t53_stage1_literature_4gpu}

STAGE1_BOOTSTRAP_CFG=${STAGE1_BOOTSTRAP_CFG:-experiments/fiftythirdtry53_pmnet_prior_gan_cyclic/fiftythirdtry53_pmnet_prior_stage1_cycle0_bootstrap.yaml}
STAGE1_FEEDBACK_CFG=${STAGE1_FEEDBACK_CFG:-experiments/fiftythirdtry53_pmnet_prior_gan_cyclic/fiftythirdtry53_pmnet_prior_stage1_cycle_feedback_resume.yaml}
STAGE2_CFG=${STAGE2_CFG:-experiments/fiftythirdtry53_pmnet_tail_refiner_cyclic/fiftythirdtry53_pmnet_tail_refiner_stage2_cycle.yaml}
STAGE3_CFG=${STAGE3_CFG:-experiments/fiftythirdtry53_pmnet_tail_refiner_cyclic/fiftythirdtry53_pmnet_stage3_nlos_cycle.yaml}
STAGE2_JSON=${STAGE2_JSON:-outputs/fiftythirdtry53_tail_refiner_stage2_teacher_literature_cyclic_4gpu/validate_metrics_tail_refiner_latest.json}
STAGE3_JSON=${STAGE3_JSON:-outputs/fiftythirdtry53_stage3_nlos_global_context_cyclic_4gpu/validate_metrics_tail_refiner_latest.json}
SOURCE_CKPT=${SOURCE_CKPT:-/workspace/TFGpractice/TFGFiftyFirstTry51/outputs/fiftyfirsttry51_pmnet_prior_stage1_literature_t51_stage1_w112_4gpu/best_cgan.pt}
REDUCED_BASE_CKPT=${REDUCED_BASE_CKPT:-outputs/fiftythirdtry53_pmnet_prior_stage1_literature_cyclic_t53_stage1_literature_4gpu/reduced_try51_best_to_literature112.pt}

cd "$TRY_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12}

stage1_bootstrap_runtime=$(mktemp "${TMPDIR:-/tmp}/try53_stage1_bootstrap_XXXX.yaml")
stage1_feedback_runtime=$(mktemp "${TMPDIR:-/tmp}/try53_stage1_feedback_runtime_XXXX.yaml")
stage1_feedback_final=$(mktemp "${TMPDIR:-/tmp}/try53_stage1_feedback_final_XXXX.yaml")
stage2_runtime=$(mktemp "${TMPDIR:-/tmp}/try53_stage2_runtime_XXXX.yaml")
stage3_runtime=$(mktemp "${TMPDIR:-/tmp}/try53_stage3_runtime_XXXX.yaml")
cleanup() {
  rm -f \
    "$stage1_bootstrap_runtime" \
    "$stage1_feedback_runtime" \
    "$stage1_feedback_final" \
    "$stage2_runtime" \
    "$stage3_runtime"
}
trap cleanup EXIT

ensure_python_deps() {
  "$PYTHON_BIN" - <<'PY'
missing = []
for module, package in [
    ("tqdm", "tqdm"),
    ("yaml", "pyyaml"),
    ("h5py", "h5py"),
    ("pandas", "pandas"),
    ("PIL", "pillow"),
    ("torchvision", "torchvision"),
]:
    try:
        __import__(module)
    except Exception:
        missing.append(package)

if missing:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
PY
}

summarize_runtime_config() {
  local config_path=$1
  local label=$2
  "$PYTHON_BIN" - "$config_path" "$label" <<'PY'
import sys
import yaml

config_path, label = sys.argv[1:]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

training = cfg.get("training", {}) or {}
data = cfg.get("data", {}) or {}
tail_refiner = cfg.get("tail_refiner", {}) or {}
runtime = cfg.get("runtime", {}) or {}

print(f"[runpod-config] {label}")
print(f"  output_dir={runtime.get('output_dir')}")
print(f"  training.batch_size={training.get('batch_size')}")
print(f"  training.homogeneous_city_type_batches={training.get('homogeneous_city_type_batches')}")
print(f"  data.val_batch_size={data.get('val_batch_size')}")
print(f"  tail_refiner.teacher_kind={tail_refiner.get('teacher_kind')}")
print(f"  tail_refiner.refiner_base_channels={tail_refiner.get('refiner_base_channels')}")
print(f"  tail_refiner.nlos_only={tail_refiner.get('nlos_only')}")
print(f"  runtime.resume_checkpoint={runtime.get('resume_checkpoint')}")
PY
}

apply_runpod_overrides() {
  local input_config=$1
  local output_config=$2
  local batch_size=$3
  local val_batch_size=$4
  local clear_resume=${5:-0}
  local homogeneous_city_type_batches=${6:-0}
  "$PYTHON_BIN" - "$input_config" "$output_config" "$batch_size" "$val_batch_size" "$clear_resume" "$homogeneous_city_type_batches" <<'PY'
import sys
from pathlib import Path
import yaml

inp, outp, batch_size, val_batch_size, clear_resume, homogeneous_city_type_batches = sys.argv[1:]
batch_size = int(batch_size)
val_batch_size = int(val_batch_size)
clear_resume = clear_resume == "1"
homogeneous_city_type_batches = homogeneous_city_type_batches == "1"

with open(inp, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("training", {})
cfg["training"]["batch_size"] = batch_size
cfg["training"]["homogeneous_city_type_batches"] = homogeneous_city_type_batches
cfg.setdefault("data", {})
cfg["data"]["val_batch_size"] = val_batch_size
cfg["data"]["num_workers"] = int(cfg["data"].get("num_workers", 2))
cfg["data"]["val_num_workers"] = int(cfg["data"].get("val_num_workers", 1))
if clear_resume:
    cfg.setdefault("runtime", {})
    cfg["runtime"]["resume_checkpoint"] = ""

with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

run_stage1_bootstrap() {
  "$PYTHON_BIN" cluster/prepare_runtime_config.py \
    --input-config "$STAGE1_BOOTSTRAP_CFG" \
    --output-config "$stage1_bootstrap_runtime" \
    --output-suffix "$OUTPUT_SUFFIX" \
    --hdf5-path "$DATASET_PATH"

  if [ "$START_FROM_SCRATCH" = "1" ]; then
    apply_runpod_overrides "$stage1_bootstrap_runtime" "$stage1_bootstrap_runtime" "$RUNPOD_STAGE1_BATCH_SIZE" "$RUNPOD_STAGE1_VAL_BATCH_SIZE" 1 1
  else
    mkdir -p "$(dirname "$REDUCED_BASE_CKPT")"
    "$PYTHON_BIN" scripts/widen_pmnet_checkpoint.py \
      --dst-config "$STAGE1_BOOTSTRAP_CFG" \
      --src-ckpt "$SOURCE_CKPT" \
      --out-ckpt "$REDUCED_BASE_CKPT"
    apply_runpod_overrides "$stage1_bootstrap_runtime" "$stage1_bootstrap_runtime" "$RUNPOD_STAGE1_BATCH_SIZE" "$RUNPOD_STAGE1_VAL_BATCH_SIZE" 0 1
  fi
  summarize_runtime_config "$stage1_bootstrap_runtime" "stage1_bootstrap"

  "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node=4 --master-port "$MASTER_PORT_STAGE1_BOOTSTRAP" \
    train_pmnet_prior_gan.py --config "$stage1_bootstrap_runtime"
}

run_stage2() {
  apply_runpod_overrides "$STAGE2_CFG" "$stage2_runtime" "$RUNPOD_STAGE23_BATCH_SIZE" "$RUNPOD_STAGE23_VAL_BATCH_SIZE" 0 0
  summarize_runtime_config "$stage2_runtime" "stage2"
  "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node=4 --master-port "$MASTER_PORT_STAGE2" \
    train_pmnet_tail_refiner.py --config "$stage2_runtime"
}

run_stage3() {
  apply_runpod_overrides "$STAGE3_CFG" "$stage3_runtime" "$RUNPOD_STAGE23_BATCH_SIZE" "$RUNPOD_STAGE23_VAL_BATCH_SIZE" 0 0
  summarize_runtime_config "$stage3_runtime" "stage3"
  "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node=4 --master-port "$MASTER_PORT_STAGE3" \
    train_pmnet_tail_refiner.py --config "$stage3_runtime"
}

run_stage1_feedback() {
  "$PYTHON_BIN" cluster/prepare_runtime_config.py \
    --input-config "$STAGE1_FEEDBACK_CFG" \
    --output-config "$stage1_feedback_runtime" \
    --output-suffix "$OUTPUT_SUFFIX" \
    --hdf5-path "$DATASET_PATH"

  "$PYTHON_BIN" cluster/prepare_try53_stage1_feedback_config.py \
    --input-config "$stage1_feedback_runtime" \
    --output-config "$stage1_feedback_final" \
    --stage2-json "$STAGE2_JSON" \
    --stage3-json "$STAGE3_JSON"

  apply_runpod_overrides "$stage1_feedback_final" "$stage1_feedback_final" "$RUNPOD_STAGE1_BATCH_SIZE" "$RUNPOD_STAGE1_VAL_BATCH_SIZE" 0 1
  summarize_runtime_config "$stage1_feedback_final" "stage1_feedback"

  "$TORCHRUN_BIN" --standalone --nnodes=1 --nproc-per-node=4 --master-port "$MASTER_PORT_STAGE1_FEEDBACK" \
    train_pmnet_prior_gan.py --config "$stage1_feedback_final"
}

echo "Try53 sequential RunPod launch"
echo "TRY_ROOT=$TRY_ROOT"
echo "DATASET_PATH=$DATASET_PATH"
echo "CHAIN_REPEATS=$CHAIN_REPEATS"
echo "START_FROM_SCRATCH=$START_FROM_SCRATCH"
echo "RUNPOD_STAGE1_BATCH_SIZE=$RUNPOD_STAGE1_BATCH_SIZE"
echo "RUNPOD_STAGE1_VAL_BATCH_SIZE=$RUNPOD_STAGE1_VAL_BATCH_SIZE"
echo "RUNPOD_STAGE23_BATCH_SIZE=$RUNPOD_STAGE23_BATCH_SIZE"
echo "RUNPOD_STAGE23_VAL_BATCH_SIZE=$RUNPOD_STAGE23_VAL_BATCH_SIZE"

ensure_python_deps

for ((cycle=1; cycle<=CHAIN_REPEATS; cycle++)); do
  echo "=== Try53 sequential cycle ${cycle}/${CHAIN_REPEATS} ==="

  echo "Stage 1 bootstrap"
  run_stage1_bootstrap

  echo "Stage 2"
  run_stage2

  echo "Stage 3"
  run_stage3

  echo "Stage 1 feedback"
  run_stage1_feedback

  echo "Stage 2 (feedback teacher)"
  run_stage2

  echo "Stage 3 (feedback teacher)"
  run_stage3
done

echo "Try53 sequential cycles complete"
