#!/bin/bash
#SBATCH --job-name=sparc_r2
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%A_%a.out
# usage: sbatch --array=1-N run2.slurm <jobsfile>   (jobsfile relative to ws/hprc, default jobs2.txt)
set -eo pipefail
source ws/hprc/env.sh; source $P/venv/bin/activate
JF=${1:-jobs2.txt}
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ws/hprc/$JF)
TAG=$(echo "$ARGS" | sed 's/--//g; s/ /_/g')
cd ws; mkdir -p results2
if [ -f ../logs/done2_${TAG} ]; then echo "already done"; exit 0; fi
echo "ARGS: $ARGS"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python hest1k_run2.py $ARGS
echo DONE > ../logs/done2_${TAG}
