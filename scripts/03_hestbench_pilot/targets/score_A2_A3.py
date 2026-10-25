import os, sys, json; sys.path.append(os.getcwd())
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch
import sparcmap_lib as L, score_lib as SL
sets = json.load(open('sparc_programs_msigdb.json'))['sets']; P = L.PROGRAMS
folds = L.wt_folds(); wt = folds['sample'].tolist()
# ---------- gene universe for random sets: detected in >=90% of the 57 WT slides
if not os.path.exists('results/random_gene_sets.json'):
    cnt = {}
    for s in wt:
        a = ad.read_h5ad(f"data/hest_bench/{SL.INV.loc[s,'task']}/adata/{s}.h5ad", backed='r')
        X = a.X[:]; X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
        for g in np.array(a.var_names)[np.asarray((X > 0).sum(0)).ravel() > 0]: cnt[g] = cnt.get(g, 0) + 1
        a.file.close()
    U = sorted(g for g, c in cnt.items() if c >= 0.9 * len(wt))
    rng = np.random.default_rng(2026); Us = set(U); R = {}
    for d in range(3):
        for p in P:
            k = len(set(sets[p]) & Us); R[f'rand{d}'] = R.get(f'rand{d}', {}); R[f'rand{d}'][p] = list(rng.choice(U, k, replace=False))
    json.dump(dict(universe_size=len(U), sizes={p: len(set(sets[p]) & Us) for p in P}, sets=R), open('results/random_gene_sets.json', 'w'))
RS = json.load(open('results/random_gene_sets.json'))['sets']
# ---------- split halves: 10 repeats per program (nominal gene list, split before slide intersection)
rng = np.random.default_rng(7); HALF = []
for rep in range(10):
    A, B = {}, {}
    for p in P:
        g = np.array(sorted(set(sets[p]))); rng.shuffle(g); h = len(g) // 2
        A[p], B[p] = list(g[:h]), list(g[h:])
    HALF.append((A, B))
def corr_cols(A, B):
    A = (A - A.mean(0)) / (A.std(0) + 1e-12); B = (B - B.mean(0)) / (B.std(0) + 1e-12); return A.T @ B / len(A)
def centred(Y):
    Z = (Y - Y.mean(0)) / (Y.std(0) + 1e-12); return Z - Z.mean(1, keepdims=True)
for sid in wt:
    if os.path.exists(f'targets_rand/{sid}.parquet'): continue
    C = SL.load_counts(sid); Xn = SL.lognorm(C); good = np.asarray(C['X'].sum(1)).ravel() > 0
    Ms = [SL.membership(C['genes'], Sd)[0] for Sd in RS.values()]
    ES = SL.ssgsea(C, Ms); cols = {'barcode': C['barcode']}
    for (d, Sd), M, es in zip(RS.items(), Ms, ES):
        assert SL.membership(C['genes'], Sd)[1] == P
        um = SL.ulm_mlm(C, M, Xn=Xn)
        for m, Y in [('ssgsea', es), ('ulm', um['ulm']), ('mlm', um['mlm'])]:
            Y = Y.astype(np.float32); Y[~good] = np.nan
            for j, p in enumerate(P): cols[f'{d}_{m}__{p}'] = Y[:, j]
    pd.DataFrame(cols).to_parquet(f'targets_rand/{sid}.parquet'); print('A3', sid, flush=True)
for sid in wt:
    of = f'results/A2_perslide/{sid}.csv'
    if os.path.exists(of): continue
    C = SL.load_counts(sid); Xn = SL.lognorm(C); depth = C['depth']
    good = np.asarray(C['X'].sum(1)).ravel() > 0
    rows = []; dg = depth[good]
    MM = [(SL.membership(C['genes'], A), SL.membership(C['genes'], B)) for A, B in HALF]
    ES = SL.ssgsea(C, [x for (MA, _), (MB, _) in MM for x in (MA, MB)])
    for rep, ((MA, nA), (MB, nB)) in enumerate(MM):
        ok = [p for p in P if p in nA and p in nB]; ia = [nA.index(p) for p in ok]; ib = [nB.index(p) for p in ok]
        sc = {}
        sc['ssgsea'] = (ES[2 * rep][good][:, ia], ES[2 * rep + 1][good][:, ib])
        ua, ub = SL.ulm_mlm(C, MA, Xn=Xn), SL.ulm_mlm(C, MB, Xn=Xn)
        for m in ['ulm', 'mlm']: sc[m] = (ua[m][good][:, ia], ub[m][good][:, ib])
        sc['ssgsea_centred'] = (centred(sc['ssgsea'][0]), centred(sc['ssgsea'][1]))
        for m, (YA, YB) in sc.items():
            RA, RB = L.residualize(YA.astype(np.float64), dg[:, None]), L.residualize(YB.astype(np.float64), dg[:, None])
            Craw, Cres = corr_cols(YA, YB), corr_cols(RA, RB)
            for j, p in enumerate(ok):
                off = lambda Cm: (np.delete(Cm[j], j).mean() + np.delete(Cm[:, j], j).mean()) / 2
                rows.append(dict(sample=sid, method=m, rep=rep, program=p, nA=int(MA[:, ia[j]].sum()), nB=int(MB[:, ib[j]].sum()),
                                 r_raw=Craw[j, j], r_depthres=Cres[j, j], offdiag_raw=off(Craw), offdiag_depthres=off(Cres)))
    pd.DataFrame(rows).to_csv(of, index=False); print(sid, len(rows), flush=True)
