"""Reliability (noise ceiling) of measured program targets on the 58 CRC test sections.

Two split-half estimates per section x program, scored exactly as the training targets:
  count split : each UMI assigned to half A or B (binomial thinning, p=0.5); score both halves.
                Captures technical (counting) noise only -> optimistic ceiling.
  gene split  : each program's genes split at random into two halves; score both halves (ULM only).
                Captures counting noise plus gene-set heterogeneity -> internal consistency.
Agreement between halves is Pearson r after residualising both on log1p(total UMI) of the section
(the same covariate used in evaluation); Spearman-Brown gives full-length reliability rel = 2r/(1+r).
Maximum attainable depth-adjusted correlation of any predictor with the measured target = sqrt(rel).
Output: results/noise_ceiling_long.csv
"""
import os, sys, json, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
WS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, WS)
from ssgsea import ssgsea_gpu
import sparcmap_lib as L

SETS = json.load(open(os.path.join(WS, 'sparc_programs_msigdb.json')))['sets']
RS = json.load(open(os.path.join(WS, 'random_gene_sets.json')))['sets']['rand0']
P = L.PROGRAMS
ST = os.path.join(WS, 'data/hest1k_crc/st'); TGT = os.path.join(WS, 'targets_crc')


def lognorm(X):
    X = X.astype(np.float64); s = np.asarray(X.sum(1)).ravel(); s[s == 0] = 1
    return sp.csr_matrix(sp.diags(1e4 / s) @ X).log1p()


def ulm(Xl, genes, sets, tmin=5):
    """Vectorised decoupler ULM: per spot, t-statistic of the slope of expression on set membership
    across all genes in the data."""
    gi = {g: i for i, g in enumerate(genes)}; G = len(genes)
    names, W = [], []
    for p, gs in sets.items():
        ii = sorted({gi[g] for g in gs if g in gi})
        if len(ii) >= tmin:
            w = np.zeros(G); w[ii] = 1; names.append(p); W.append(w)
    W = np.array(W).T
    mx = np.asarray(Xl.mean(1)).ravel(); sxx = np.asarray(Xl.multiply(Xl).sum(1)).ravel() - G * mx ** 2
    mw = W.mean(0); sww = (W ** 2).sum(0) - G * mw ** 2
    sxw = np.asarray(Xl @ W) - G * mx[:, None] * mw[None, :]
    r = sxw / np.sqrt(np.outer(sxx, sww) + 1e-30); r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt((G - 2) / (1 - r ** 2))
    return pd.DataFrame(t, columns=names).reindex(columns=list(sets))


def depth_r(A, B, d):
    Ar, Br = L.residualize(A, d[:, None]), L.residualize(B, d[:, None])
    return L._pearson_cols(Ar, Br), L._pearson_cols(A, B)


def run(sid, rng, n_count=2, n_gene=5):
    t = pd.read_parquet(f'{TGT}/{sid}.parquet')
    a = ad.read_h5ad(f'{ST}/{sid}.h5ad'); a.var_names_make_unique()
    a = a[pd.Index(a.obs_names).get_indexer(t.barcode)]
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    keep = np.asarray((X > 0).sum(0)).ravel() > 0; X = X[:, keep].astype(np.float64); genes = np.array(a.var_names[keep])
    d = np.log1p(t.total_counts.values.astype(float))
    rows = []
    # validation against stored targets
    full = ulm(lognorm(X), genes, SETS)
    chk = np.nanmean([np.corrcoef(full[p], t[f'ulm__{p}'])[0, 1] for p in P if t[f'ulm__{p}'].notna().all()])
    add = lambda kind, meth, rd, rr, rep: rows.extend(
        dict(sample=sid, program=p, split=kind, target=meth, rep=rep, r_half_depth=rd[j], r_half_raw=rr[j]) for j, p in enumerate(P))
    for rep in range(n_count):
        Xa = X.copy(); Xa.data = rng.binomial(X.data.astype(np.int64), 0.5).astype(np.float64); Xb = X - Xa
        for meth, sets in [('ulm', SETS), ('rand0ulm', RS)]:
            A = ulm(lognorm(Xa), genes, sets).values; B = ulm(lognorm(Xb), genes, sets).values
            add('count', meth, *depth_r(A, B, d), rep)
        ea, na = ssgsea_gpu(Xa, genes, SETS); eb, nb = ssgsea_gpu(Xb, genes, SETS)
        A = pd.DataFrame(ea, columns=na).reindex(columns=P).values; B = pd.DataFrame(eb, columns=nb).reindex(columns=P).values
        add('count', 'ssgsea', *depth_r(A, B, d), rep)
    Xl = lognorm(X); gset = set(genes)
    for rep in range(n_gene):
        for meth, sets in [('ulm', SETS), ('rand0ulm', RS)]:
            SA, SB = {}, {}
            for p in P:
                g = np.array(sorted(set(sets[p]) & gset)); rng.shuffle(g); h = len(g) // 2
                SA[p], SB[p] = list(g[:h]), list(g[h:])
            A = ulm(Xl, genes, SA).values; B = ulm(Xl, genes, SB).values
            add('gene', meth, *depth_r(A, B, d), rep)
    return rows, chk


def _one(args):
    sid, seed = args
    return sid, *run(sid, np.random.default_rng(seed))


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    ids = sorted(pd.read_csv(os.path.join(WS, 'crc_folds.csv')).id); out = []; chks = {}
    with ProcessPoolExecutor(8) as ex:
        for sid, r, c in ex.map(_one, [(s, 2026 + i) for i, s in enumerate(ids)]):
            out += r; chks[sid] = c; print(sid, f'ulm_check_r={c:.4f}', flush=True)
    pd.DataFrame(out).to_csv(os.path.join(WS, 'results/noise_ceiling_long.csv'), index=False)
    pd.Series(chks).to_csv(os.path.join(WS, 'results/noise_ceiling_ulm_check.csv'))
