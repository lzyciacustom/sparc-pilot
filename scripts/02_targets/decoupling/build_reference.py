"""Build cell-type reference profiles (ref_profiles.parquet) from GSE132465 for NNLS deconvolution.
Reconstructed from the pilot's interactive session (same calls, same order): parse the raw UMI matrix in 3,000-gene chunks,
map the 23-patient SMC annotation to 12 types (decon_resid.map_type), average per type (decon_resid.type_profiles).
Inputs: data/ref/GSE132465/*.txt.gz (scripts/00_data/download_references.sh). Run from the flat workspace."""
import sys, numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, '.'); import decon_resid as dr
D = 'data/ref/GSE132465/GSE132465_GEO_processed_CRC_10X_'
ann = pd.read_csv(D + 'cell_annotation.txt.gz', sep='\t')
blocks, genes = [], []
for ch in pd.read_csv(D + 'raw_UMI_count_matrix.txt.gz', sep='\t', index_col=0, chunksize=3000):
    genes += ch.index.tolist(); blocks.append(sp.csr_matrix(ch.values.astype(np.float32))); cells = ch.columns
G = sp.vstack(blocks).T.tocsr()                                   # cells x genes (63,689 x 33,694 in the pilot)
assert (cells == ann.Index.values).all(), 'cell order mismatch between matrix and annotation'
lab = np.array([dr.map_type(c, s, k) for c, s, k in zip(ann.Cell_type, ann.Cell_subtype, ann.Class)], dtype=object)
P, nc = dr.type_profiles(G, lab, pd.Index(genes)); P.to_parquet('ref_profiles.parquet')
ann.assign(mapped=lab).groupby(['Cell_type', 'Cell_subtype', 'Class', 'mapped'], dropna=False).size().rename('n_cells').reset_index().to_csv('celltype_mapping.csv', index=False)
print(G.shape, pd.Series(lab).value_counts(dropna=False).to_dict())
