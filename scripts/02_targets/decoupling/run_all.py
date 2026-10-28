import os, sys, json, time, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from scipy.stats import spearmanr
from joblib import Parallel, delayed
sys.path.insert(0, '.')
from decon_resid import TYPES, DROP_TYPE, select_markers, nnls_deconv, residualise, SANITY

ST = os.environ.get('SPARC_ST', 'data/hest1k_crc/st')  # usage: python run_all.py <programs.json> <ids.csv> <n_jobs>
P_all = pd.read_parquet('ref_profiles.parquet')
P_all = P_all.loc[TYPES, P_all.sum(0) > 0]
PROGS = list(json.load(open(sys.argv[1]))['sets'])
ids = pd.read_csv(sys.argv[2])['id'].tolist()
os.makedirs('out_parquet', exist_ok=True)

def run(sid):
    t0 = time.time()
    a = ad.read_h5ad(f'{ST}/{sid}.h5ad')
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    total = np.asarray(X.sum(1)).ravel()
    vn = pd.Index(a.var_names)
    first = ~vn.duplicated()
    X = X[:, np.where(first)[0]]; vn = vn[first]
    common = vn.intersection(P_all.columns)
    gi = vn.get_indexer(common)
    Xc = X[:, gi]
    P = P_all[common]; P = P.div(P.sum(1), axis=0) * 1e6
    mk = select_markers(P / 1e6, common, n=50)
    mgenes = pd.Index(sorted(set(g for v in mk.values() for g in v)))
    libc = np.asarray(Xc.sum(1)).ravel()
    with np.errstate(invalid='ignore', divide='ignore'):
        cpm = sp.diags(np.where(libc > 0, 1e6 / libc, 0)) @ Xc
    Y = cpm[:, common.get_indexer(mgenes)].toarray()
    comp = nnls_deconv(Y, P[mgenes].values)
    comp[libc == 0] = np.nan
    df = pd.DataFrame({'barcode': a.obs_names.astype(str), 'total_counts': total})
    for j, t in enumerate(TYPES): df['comp__' + t] = comp[:, j]
    san = {'slide': sid}
    cpm_full = cpm.tocsc()
    for k, (tt, gg) in SANITY.items():
        gg = [g for g in gg if g in common]
        if not gg: san[k] = np.nan; continue
        e = np.log1p(np.asarray(cpm_full[:, common.get_indexer(gg)].sum(1)).ravel())
        p = np.nansum(np.column_stack([df['comp__' + t] for t in tt]), 1)
        ok = libc > 0
        san[k] = spearmanr(p[ok], e[ok]).correlation
        san[k + '_markers_in_nnls'] = int(sum(g in mgenes for g in gg))
    tg = pd.read_parquet(f'targets_crc/{sid}.parquet', columns=['barcode'] + ['ulm__' + p for p in PROGS])
    tg = tg.set_index('barcode').reindex(df.barcode)
    T = tg.values.astype(float)
    C = df[['comp__' + t for t in TYPES if t != DROP_TYPE]].values
    ok = np.isfinite(T).all(1) & np.isfinite(C).all(1)
    Rs = np.full_like(T, np.nan)
    R, r2 = residualise(T[ok], C[ok])
    Rs[ok] = R
    for j, p in enumerate(PROGS): df['ulmres__' + p] = Rs[:, j]
    df.to_parquet(f'out_parquet/{sid}.parquet', index=False)
    info = dict(slide=sid, n_spots=len(df), n_target_spots=int(np.isfinite(T).all(1).sum()), n_regressed=int(ok.sum()),
                n_genes_slide=int(len(vn)), n_common=int(len(common)), n_marker_genes=int(len(mgenes)),
                zero_lib_spots=int((libc == 0).sum()), sec=round(time.time() - t0, 1))
    info.update({'meanprop__' + t: float(np.nanmean(comp[:, j])) for j, t in enumerate(TYPES)})
    return info, san, dict(zip(PROGS, r2)), mk

res = Parallel(n_jobs=int(sys.argv[3]), verbose=0)(delayed(run)(s) for s in ids)
pd.DataFrame([r[0] for r in res]).to_csv('slide_info.csv', index=False)
pd.DataFrame([r[1] for r in res]).to_csv('sanity_spearman.csv', index=False)
pd.DataFrame({r[0]['slide']: r[2] for r in res}).T.to_csv('r2_by_slide.csv')
json.dump({r[0]['slide']: r[3] for r in res}, open('markers_by_slide.json', 'w'))
print('done', len(res))
