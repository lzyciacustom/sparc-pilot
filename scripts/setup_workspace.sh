#!/bin/bash
# Build the flat run directory the scripts expect (scripts are symlinked).
# usage: bash scripts/setup_workspace.sh /path/to/workdir
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; W="${1:?usage: setup_workspace.sh WORKDIR}"
mkdir -p "$W/trackB" "$W/data" "$W/models"
for f in "$REPO"/src/sparcpilot/*.py "$REPO"/scripts/*/*.py "$REPO"/scripts/*/*/*.py "$REPO"/scripts/*/*/*.R "$REPO"/scripts/*/*.sh "$REPO"/scripts/*/*/*.sh "$REPO"/metadata/*; do
  case "$f" in */scripts/09_composition/*) continue;; esac
  [ -e "$f" ] && ln -sf "$f" "$W/$(basename "$f")"; done
# composition / MDACC scripts resolve the workspace as their parent directory, so they live in $W/comp
mkdir -p "$W/comp"; for f in "$REPO"/scripts/09_composition/*.py; do ln -sf "$f" "$W/comp/$(basename "$f")"; done
ln -sf "$REPO/src/sparcpilot/spatial_models.py" "$W/trackB/spatial_models.py"
mkdir -p "$W/hprc"; for f in "$REPO"/scripts/04_hest1k/hpc/*; do ln -sf "$f" "$W/hprc/$(basename "$f")"; done
if [ ! -d "$W/sparc_repo" ]; then git clone https://github.com/aziz-ayed/SPARC.git "$W/sparc_repo"; git -C "$W/sparc_repo" checkout a046572fda0e97085bc159dbcda3d21c9ec6cd73; fi
echo "workspace ready: $W  (next: cd $W && python download_hest.py meta)"
