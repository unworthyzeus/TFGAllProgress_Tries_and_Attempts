#!/bin/bash
# Try 49 cleanup job on sert-2001 to clear stale CUDA processes between jobs
#SBATCH --job-name=ckm-t49-cleanup-sert2001
#SBATCH -A gpu
#SBATCH -p gpu
#SBATCH --qos=small_gpu
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -w sert-2001
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-00:05:00
#SBATCH --output=logs_cleanup_fortyninthtry49_%j.out
#SBATCH --error=logs_cleanup_fortyninthtry49_%j.err

set -euo pipefail

echo "[cleanup] host=$(hostname) start=$(date --iso-8601=seconds)"
echo "[cleanup] nvidia-smi before"
nvidia-smi || true

# Kill stale training processes from Try49/Try50 chains if the previous job
# timed out and left CUDA contexts behind on sert-2001.
pkill -9 -u gmoreno -f 'train_pmnet_tail_refiner.py' || true
pkill -9 -u gmoreno -f 'train_pmnet_prior_gan.py' || true
pkill -9 -u gmoreno -f 'torchrun.*TFGFortyNinthTry49' || true
pkill -9 -u gmoreno -f 'torchrun.*TFGFiftiethTry50' || true

sleep 5

echo "[cleanup] user processes after kill"
ps -u gmoreno -o pid,ppid,comm,args | egrep 'train_pmnet_(tail_refiner|prior_gan)|torchrun' || true

echo "[cleanup] nvidia-smi after"
nvidia-smi || true
echo "[cleanup] end=$(date --iso-8601=seconds)"
