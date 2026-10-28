"""NNLS deconvolution as in scripts/02_targets/decoupling/run_all.py, but with every gene belonging to any of the
36 programs (curated) or the rand0 random sets removed from the marker panel, so composition estimates and
program scores share no genes. Output: comp/out_disjoint/<sid>.parquet (barcode, comp__<type>)."""
import os, sys, json, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from joblib import Parallel, delayed
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, WS)
from decon_resid import TYPES, select_markers, nnls_deconv
P_all = pd.read_parquet(f'{WS}/comp/ref_profiles.parquet'); P_all = P_all.loc[TYPES, P_all.sum(0) > 0]
EXCL = {g for v in json.load(open(f'{WS}/sparc_programs_msigdb.json'))['sets'].values() for g in v}
EXCL |= {g for v in json.load(open(f'{WS}/random_gene_sets.json'))['sets']['rand0'].values() for g in v}
os.makedirs(f'{WS}/comp/out_disjoint', exist_ok=True)

def run(sid):
    a = ad.read_h5ad(f'{WS}/data/hest1k_crc/st/{sid}.h5ad')
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    vn = pd.Index(a.var_names); first = ~vn.duplicated(); X = X[:, np.where(first)[0]]; vn = vn[first]
    common = vn.intersection(P_all.columns); Xc = X[:, vn.get_indexer(common)]
    P = P_all[common]; P = P.div(P.sum(1), axis=0) * 1e6
    avail = pd.Index([g for g in common if g not in EXCL])
    mk = select_markers(P / 1e6, avail, n=50)
    mg = pd.Index(sorted({g for v in mk.values() for g in v}))
    lib = np.asarray(Xc.sum(1)).ravel()
    with np.errstate(invalid='ignore', divide='ignore'):
        cpm = sp.diags(np.where(lib > 0, 1e6 / lib, 0)) @ Xc
    comp = nnls_deconv(cpm[:, common.get_indexer(mg)].toarray(), P[mg].values); comp[lib == 0] = np.nan
    df = pd.DataFrame({'barcode': a.obs_names.astype(str)})
    for j, t in enumerate(TYPES): df['comp__' + t] = comp[:, j]
    df.to_parquet(f'{WS}/comp/out_disjoint/{sid}.parquet', index=False)
    return sid, len(mg), len(set(mg) & EXCL)

ids = sorted(pd.read_csv(f'{WS}/crc_folds.csv').id)
res = Parallel(n_jobs=12)(delayed(run)(s) for s in ids)
print(pd.DataFrame(res, columns=['sid', 'n_markers', 'overlap']).describe().round(1).to_string())
