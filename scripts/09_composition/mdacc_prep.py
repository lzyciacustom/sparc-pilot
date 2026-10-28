"""Preprocess MD Anderson CRC metastasis Visium (GSE294385; Liu et al., Cancer Cell 2026): 32 liver + primary-colon sections.
Lung sections are excluded (no lung parenchyma in the reference).
Per section: in-tissue spots with a QC'd pathology annotation (authors' visium_meta_after_qc.rds, exported to CSV);
ULM program scores (curated 36 + rand0 36, vectorised decoupleR-equivalent, genes detected in the section);
log1p total UMI; NNLS composition on the 17-type colon+liver reference (top-50 markers per type, as in decon_resid.py),
with the full marker panel and with a gene-disjoint panel (all program and rand0 genes removed).
Outputs: ext/mdacc/proc/<sample>.parquet ; ext/mdacc/universe.txt (genes detected in >=90% of sections)
"""
import os, sys, json, glob, tarfile, numpy as np, pandas as pd, scipy.sparse as sp, scipy.io
from joblib import Parallel, delayed
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, WS); sys.path.insert(0, f'{WS}/comp')
EXT = os.environ.get('SPARC_EXT', os.path.join(WS, 'ext'))  # external cohorts (scripts/00_data/download_external.py)
from noise_ceiling import ulm, lognorm
from decon_resid import select_markers, nnls_deconv
from build_ref17 import TYPES17
import sparcmap_lib as L
X_ = f'{EXT}/mdacc'; OUT = f'{X_}/proc'; os.makedirs(OUT, exist_ok=True)
SETS = json.load(open(f'{WS}/sparc_programs_msigdb.json'))['sets']; RS = json.load(open(f'{WS}/random_gene_sets.json'))['sets']['rand0']
EXCL = {g for v in SETS.values() for g in v} | {g for v in RS.values() for g in v}
P_all = pd.read_parquet(f'{WS}/comp/ref17_profiles.parquet'); P_all = P_all.loc[TYPES17, P_all.sum(0) > 0]
meta = pd.read_csv(f'{X_}/visium_meta_after_qc.csv')
meta['sample'] = meta.Barcode.str.split('~').str[0]; meta['barcode'] = meta.Barcode.str.split('~').str[1]


def load(sample):
    d = f'{X_}/unpacked/{sample}/filtered_feature_bc_matrix'
    if not os.path.isdir(d):
        tf = glob.glob(f'{X_}/raw/*_{sample}.tar.gz')[0]; tarfile.open(tf).extractall(f'{X_}/unpacked')
    M = scipy.io.mmread(f'{d}/matrix.mtx.gz').T.tocsr().astype(np.float64)
    bc = pd.read_csv(f'{d}/barcodes.tsv.gz', header=None)[0].values; ft = pd.read_csv(f'{d}/features.tsv.gz', header=None, sep='\t')
    genes = ft[1].values; first = ~pd.Index(genes).duplicated(); M = M[:, np.where(first)[0]]; genes = genes[first]
    return M, bc, genes


def run(sample):
    M, bc, genes = load(sample)
    m = meta[meta['sample'] == sample].set_index('barcode'); keep = pd.Index(bc).isin(m.index)
    M, bc = M[keep], bc[keep]; det = np.asarray((M > 0).sum(0)).ravel() > 0
    Mk, gk = M[:, det], genes[det]
    df = pd.DataFrame({'barcode': bc, 'total_counts': np.asarray(M.sum(1)).ravel()})
    for c in ['Layer1', 'Layer2', 'Layer3']: df[c] = m.loc[bc, c].values
    Xl = lognorm(Mk)
    for tag, sets in [('ulm', SETS), ('rand0ulm', RS)]:
        T = ulm(Xl, gk, sets)
        for p in L.PROGRAMS: df[f'{tag}__{p}'] = T[p].values
    common = pd.Index(genes).intersection(P_all.columns); Xc = M[:, pd.Index(genes).get_indexer(common)]
    P = P_all[common]; P = P.div(P.sum(1), axis=0) * 1e6
    lib = np.asarray(Xc.sum(1)).ravel()
    with np.errstate(invalid='ignore', divide='ignore'):
        cpm = sp.diags(np.where(lib > 0, 1e6 / lib, 0)) @ Xc
    for label, avail in [('comp', common), ('compdj', pd.Index([g for g in common if g not in EXCL]))]:
        mk = select_markers(P / 1e6, avail, n=50); mg = pd.Index(sorted({g for v in mk.values() for g in v}))
        pi = nnls_deconv(cpm[:, common.get_indexer(mg)].toarray(), P[mg].values); pi[lib == 0] = np.nan
        for j, t in enumerate(TYPES17): df[f'{label}__{t}'] = pi[:, j]
    df.to_parquet(f'{OUT}/{sample}.parquet', index=False)
    return sample, len(df), set(gk)


if __name__ == '__main__':
    samples = sorted(meta.loc[meta.Layer1 != 'Lung', 'sample'].unique())
    res = Parallel(n_jobs=8)(delayed(run)(s) for s in samples)
    cnt = {}
    for _, _, g in res:
        for x in g: cnt[x] = cnt.get(x, 0) + 1
    U = sorted(x for x, c in cnt.items() if c >= 0.9 * len(res)); open(f'{X_}/universe.txt', 'w').write('\n'.join(U))
    print('sections', len(res), 'spots', sum(r[1] for r in res), 'universe', len(U))
