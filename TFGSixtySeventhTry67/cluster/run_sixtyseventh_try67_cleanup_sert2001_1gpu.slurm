#!/bin/bash
# Try 67 cleanup on sert-2001: 1 GPU only (avoids MaxGRESPerAccount from 13x 4-GPU pending jobs).
#SBATCH --job-name=ckm-t67-cleanup-sert2001
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-00:05:00
#SBATCH --output=logs_cleanup_try67_%j.out
#SBATCH --error=logs_cleanup_try67_%j.err

set -euo pipefail

echo "[cleanup] host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi || true

pkill -9 -u gmoreno -f 'train_partitioned_pathloss_expert.py' || true
pkill -9 -u gmoreno -f 'torchrun.*TFGSixtySeventhTry67' || true

sleep 5

echo "[cleanup] user processes after kill"
ps -u gmoreno -o pid,ppid,comm,args | egrep 'train_partitioned_pathloss_expert|torchrun.*TFGSixtySeventhTry67' || true

echo "[cleanup] nvidia-smi after"
nvidia-smi || true
echo "[cleanup] end=$(date --iso-8601=seconds)"
