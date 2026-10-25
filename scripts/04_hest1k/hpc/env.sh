# sourced by every job; everything lives under $SCRATCH (nothing written to $HOME)
export P=${HPC_SCRATCH:-$SCRATCH}/sparc_pilot   # cluster-specific: edit module names below for your site
export PIP_CACHE_DIR=$P/cache/pip XDG_CACHE_HOME=$P/cache/xdg TORCH_HOME=$P/cache/torch MPLCONFIGDIR=$P/cache/mpl PYTHONNOUSERSITE=1
module purge >/dev/null 2>&1; module load Anaconda3/2025.12-2
