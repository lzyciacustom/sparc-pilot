#!/bin/bash
#SBATCH --job-name=sparc_hest1k
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%A_%a.out
set -eo pipefail
source ws/hprc/env.sh; source $P/venv/bin/activate
read T M H S X < <(sed -n "${SLURM_ARRAY_TASK_ID}p" ws/hprc/jobs.txt)
TAG=${T}__${M}__${H}__s${S}${X:+__crconly}
cd ws
if [ -f ../logs/done_${TAG} ]; then echo "already done"; exit 0; fi
python hest1k_run.py --target $T --model $M --head $H --seed $S $X
echo DONE > ../logs/done_${TAG}
