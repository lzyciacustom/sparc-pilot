#!/bin/bash
# H-optimus-1 training runs reported in docs/CORE_FINDINGS.md (base SPARC-Map, deterministic head, 6 folds).
#   bash h1_train.sh crconly   # CRC training folds only (~23 min/run on one RTX 3090)
#   bash h1_train.sh hest1k    # HEST-1k pool + CRC training folds (~1 h/run)
# Targets (targets_crc/*.parquet) come from compute_targets_crc.py, compute_targets_pool.py and score_random_*.py.
set -euo pipefail
MODE=${1:?crconly|hest1k}; FD=features/hoptimus1_crc; EXTRA=""; SUF=""
[ "$MODE" = crconly ] && { EXTRA="--crc_only"; SUF="__crconly"; }
JOBS=("ssgsea 0" "ulm 0" "rand0ssgsea 0" "rand0ulm 0" "ssgsea_depthres 0" "mlm 0" "rand1ssgsea 0" "rand2ssgsea 0" "ssgsea 1" "ulm 1")
[ "$MODE" = hest1k ] && JOBS+=("ssgsea 2" "ulm 2")
for job in "${JOBS[@]}"; do
  set -- $job
  [ -f "results/${1}__base__det__s${2}${SUF}.csv" ] && continue
  python hest1k_run.py --target "$1" --model base --head det --seed "$2" $EXTRA --feat_dir $FD
done
echo "H1 $MODE runs done"
