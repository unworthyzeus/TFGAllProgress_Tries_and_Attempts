#!/bin/bash
# Try 76 cleanup on sert-2001: 1 GPU only (avoids MaxGRESPerAccount from chained pending jobs).
#SBATCH --job-name=ckm-t76-cleanup-sert2001
#SBATCH -A gpu
#SBATCH --partition=gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:1
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0-00:05:00
#SBATCH --output=logs_cleanup_try76_%j.out
#SBATCH --error=logs_cleanup_try76_%j.err

set -euo pipefail

echo "[cleanup] host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi || true

pkill -9 -u gmoreno -f 'train_try76.py' || true
pkill -9 -u gmoreno -f 'torchrun.*TFGSeventySixthTry76' || true

sleep 5

echo "[cleanup] user processes after kill"
ps -u gmoreno -o pid,ppid,comm,args | egrep 'train_try76|torchrun.*TFGSeventySixthTry76' || true

echo "[cleanup] nvidia-smi after"
nvidia-smi || true
echo "[cleanup] end=$(date --iso-8601=seconds)"
