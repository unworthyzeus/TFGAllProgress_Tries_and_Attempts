#!/bin/bash
# Try 70 cleanup on sert-2001 (1 GPU, small QOS).
#SBATCH --job-name=ckm-t70-cleanup-sert2001
#SBATCH -A gpu
#SBATCH --partition=gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:1
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0-00:05:00
#SBATCH --output=logs_cleanup_try70_%j.out
#SBATCH --error=logs_cleanup_try70_%j.err

set -euo pipefail

echo "[cleanup t70] host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi || true

pkill -9 -u gmoreno -f 'train_partitioned_pathloss_expert.py' || true
pkill -9 -u gmoreno -f 'torchrun.*TFGSeventiethTry70' || true

sleep 5

ps -u gmoreno -o pid,ppid,comm,args | egrep 'train_partitioned_pathloss_expert|torchrun.*TFGSeventiethTry70' || true
nvidia-smi || true
echo "[cleanup t70] end=$(date --iso-8601=seconds)"
