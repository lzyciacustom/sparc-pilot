"""Combined colon (GSE132465, 12 types) + liver (GSE115469, MacParland 2018; 5 parenchymal/sinusoidal types) reference.
GSE115469_Data.csv stores log2(count / size factor + 1); counts are recovered exactly as round((2^x - 1) / min_nonzero(2^x - 1)) per cell.
Outputs: comp/ref17_counts.npz (cells x genes, raw UMI), ref17_cells.csv, ref17_genes.csv, ref17_profiles.parquet (mean per-cell fractions)."""
import os, numpy as np, pandas as pd, scipy.sparse as sp
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = f'{WS}/comp'
EXT = os.environ.get('SPARC_EXT', os.path.join(WS, 'ext'))  # external cohorts (scripts/00_data/download_external.py)
E = f'{EXT}/liver'
TYPES17 = ['Epi_tumor', 'Epi_normal', 'Fibroblast', 'Endothelial', 'Pericyte_SMC', 'Glial', 'T_NK', 'B', 'Plasma', 'Mono_Mac', 'DC', 'Mast',
           'Hepatocyte', 'Cholangiocyte', 'LSEC', 'Kupffer', 'Stellate']
if __name__ == '__main__':
    L = pd.read_csv(f'{E}/GSE115469_Data.csv.gz', index_col=0)
    ann = pd.read_csv(f'{E}/GSE115469_CellClusterType.txt.gz', sep='\t').set_index('CellName').reindex(L.columns)
    mp = {**{f'Hepatocyte_{i}': 'Hepatocyte' for i in range(1, 7)}, 'Cholangiocytes': 'Cholangiocyte', 'Periportal_LSECs': 'LSEC',
          'Central_venous_LSECs': 'LSEC', 'Non-inflammatory_Macrophage': 'Kupffer', 'Hepatic_Stellate_Cells': 'Stellate'}
    lab = ann.CellType.map(mp); keep = lab.notna().values
    e = np.power(2.0, L.values[:, keep]) - 1; mins = np.array([col[col > 0].min() for col in e.T])
    cnt = np.round(e / mins); assert np.mean(np.abs(e / mins - cnt)[e > 0] < 0.02) > 0.999
    Lc = sp.csr_matrix(cnt.T.astype(np.float32)); llab = lab.values[keep]; lg = np.array(L.index)
    G = sp.load_npz(f'{R}/gse132465_counts.npz').tocsr(); cg = pd.read_csv(f'{R}/gse132465_genes.csv').iloc[:, 0].values
    cells = pd.read_csv(f'{R}/gse132465_cells.csv')
    common = pd.Index(cg).intersection(pd.Index(lg))
    M = sp.vstack([G[:, pd.Index(cg).get_indexer(common)], Lc[:, pd.Index(lg).get_indexer(common)]]).tocsr()
    labs = np.concatenate([cells.type.values, llab]); src = np.array(['colon'] * G.shape[0] + ['liver'] * Lc.shape[0])
    sp.save_npz(f'{R}/ref17_counts.npz', M); pd.DataFrame({'type': labs, 'source': src}).to_csv(f'{R}/ref17_cells.csv', index=False)
    pd.Series(common).to_csv(f'{R}/ref17_genes.csv', index=False)
    lib = np.asarray(M.sum(1)).ravel(); Mn = sp.diags(1 / np.maximum(lib, 1)) @ M
    P = pd.DataFrame({t: np.asarray(Mn[np.where(labs == t)[0]].mean(0)).ravel() for t in TYPES17}, index=common).T
    P.to_parquet(f'{R}/ref17_profiles.parquet')
    print('cells per type', pd.Series(labs).value_counts().to_dict(), '| genes', len(common), '| liver UMI median', np.median(cnt.sum(0)))
