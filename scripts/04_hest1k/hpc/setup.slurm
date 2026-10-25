#!/bin/bash
#SBATCH --job-name=sparc_setup
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
source ws/hprc/env.sh
mkdir -p $PIP_CACHE_DIR $XDG_CACHE_HOME
if module load WebProxy 2>/dev/null; then echo "WebProxy loaded: http_proxy=${http_proxy:-unset}"; else echo 'WARNING: WebProxy module not available'; fi
[ -d $P/venv ] || python -m venv $P/venv
source $P/venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scipy scikit-learn h5py pyarrow
python -c "import torch, sklearn, pyarrow, h5py, scipy; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
echo SETUP_OK
