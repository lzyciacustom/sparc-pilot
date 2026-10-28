"""Cache the GSE132465 raw UMI matrix as a sparse cell x gene matrix with 12-type labels, and link the
deconvolution outputs the composition scripts read from comp/.
Run from <workspace>/comp after scripts/00_data/download_references.sh, build_reference.py and run_all.py.
Outputs: gse132465_counts.npz, gse132465_cells.csv, gse132465_genes.csv; links ref_profiles.parquet, out_parquet/."""
import os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, WS)
import decon_resid as dr
C = f'{WS}/comp'; D = f'{WS}/data/ref/GSE132465/GSE132465_GEO_processed_CRC_10X_'
for name in ['ref_profiles.parquet', 'out_parquet']:
    if os.path.exists(f'{WS}/{name}') and not os.path.exists(f'{C}/{name}'): os.symlink(f'{WS}/{name}', f'{C}/{name}')
if not os.path.exists(f'{C}/gse132465_counts.npz'):
    ann = pd.read_csv(D + 'cell_annotation.txt.gz', sep='\t'); t = time.time(); blocks, genes = [], []
    for ch in pd.read_csv(D + 'raw_UMI_count_matrix.txt.gz', sep='\t', index_col=0, chunksize=3000):
        genes += ch.index.tolist(); blocks.append(sp.csr_matrix(ch.values.astype(np.float32))); cells = ch.columns
    G = sp.vstack(blocks).T.tocsr(); assert (cells == ann.Index.values).all()
    lab = np.array([dr.map_type(c, s, k) or 'NA' for c, s, k in zip(ann.Cell_type, ann.Cell_subtype, ann.Class)])
    sp.save_npz(f'{C}/gse132465_counts.npz', G)
    pd.DataFrame({'cell': cells, 'type': lab, 'patient': ann.Patient.values, 'cls': ann.Class.values}).to_csv(f'{C}/gse132465_cells.csv', index=False)
    pd.Series(genes).to_csv(f'{C}/gse132465_genes.csv', index=False)
    print(G.shape, pd.Series(lab).value_counts().to_dict(), f'{time.time() - t:.0f}s')
