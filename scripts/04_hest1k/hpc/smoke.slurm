#!/bin/bash
#SBATCH --job-name=sparc_smoke
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
source ws/hprc/env.sh; source $P/venv/bin/activate
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; print('cuda available', torch.cuda.is_available())"
cd ws
python hest1k_run.py --target ulm --model gpk16 --head gauss --seed 0 --folds 0 --smoke
python hest1k_run.py --target ulm --model base --head det --seed 0 --folds 0
echo SMOKE_OK
