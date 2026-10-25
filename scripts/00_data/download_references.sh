#!/bin/bash
# External references used by the decoupling and mouse analyses.
set -euo pipefail
mkdir -p data/ref/GSE132465 data/ref/mgi
# Lee et al. 2020 Nat Genet, SMC CRC scRNA-seq (63,689 cells) - composition reference for deconvolution
B=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132465/suppl
for f in GSE132465_GEO_processed_CRC_10X_raw_UMI_count_matrix.txt.gz GSE132465_GEO_processed_CRC_10X_cell_annotation.txt.gz; do
  [ -f data/ref/GSE132465/$f ] || curl -L -o data/ref/GSE132465/$f $B/$f; done
# MGI human-mouse homology (strict one-to-one pairs are derived in compute_targets_mouse.py)
[ -f data/ref/mgi/HOM_MouseHumanSequence.rpt ] || curl -L -o data/ref/mgi/HOM_MouseHumanSequence.rpt https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt
echo done
