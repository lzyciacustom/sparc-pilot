# cluster-specific: edit the module line for your site
export P=${HPC_SCRATCH:-$SCRATCH}/sparc_pilot
export PIP_CACHE_DIR=$P/cache/pip XDG_CACHE_HOME=$P/cache/xdg TORCH_HOME=$P/cache/torch MPLCONFIGDIR=$P/cache/mpl PYTHONNOUSERSITE=1
module purge >/dev/null 2>&1; module load Anaconda3/2025.12-2
export SPARC_ENCODER=${SPARC_ENCODER:-h1}   # h1 = H-optimus-1 features (default), h0 = H-optimus-0
