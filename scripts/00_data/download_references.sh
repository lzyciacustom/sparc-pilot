#!/bin/bash
set -euo pipefail
mkdir -p data/ref/GSE132465 data/ref/mgi
# Lee et al. 2020, CRC scRNA-seq reference
B=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE132nnn/GSE132465/suppl
for f in GSE132465_GEO_processed_CRC_10X_raw_UMI_count_matrix.txt.gz GSE132465_GEO_processed_CRC_10X_cell_annotation.txt.gz; do
  [ -f data/ref/GSE132465/$f ] || curl -L -o data/ref/GSE132465/$f $B/$f; done
# MGI human-mouse homology
[ -f data/ref/mgi/HOM_MouseHumanSequence.rpt ] || curl -L -o data/ref/mgi/HOM_MouseHumanSequence.rpt https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt
echo done
