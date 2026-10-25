"""Gene-set scoring for Track A (split-half, random-set null).
ssGSEA: same closed form as compute_targets.ssgsea_gpu (rank of raw counts over detected genes, w=0.25).
ULM / MLM: re-implemented in torch (t-values of OLS slopes on log1p(CP10k) over all detected genes,
membership weight 1, intercept), validated against decoupler-2.2 values stored in targets/."""
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch
from scipy.stats import rankdata
import sparcmap_lib as L
INV = L.INV.set_index('sample'); import os; dev = os.environ.get('SCORE_DEV', 'cuda'); torch.set_num_threads(int(os.environ.get('SCORE_THREADS', '8')))

def load_counts(sid):
    t = pd.read_parquet(f'{L.TGT_DIR}/{sid}.parquet')
    a = ad.read_h5ad(f"data/hest_bench/{INV.loc[sid,'task']}/adata/{sid}.h5ad"); a.var_names_make_unique()
    a = a[pd.Index(a.obs_names).get_indexer(t.barcode)]
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    keep = np.asarray((X > 0).sum(0)).ravel() > 0
    return dict(X=X[:, keep].astype(np.float32), genes=np.array(a.var_names[keep]), barcode=t.barcode.values,
                depth=np.log1p(t.total_counts.values.astype(np.float64)))

def membership(genes, sets, min_size=5):
    gi = {g: i for i, g in enumerate(genes)}; names, cols = [], []
    for p, g in sets.items():
        ii = sorted({gi[x] for x in g if x in gi})
        if len(ii) >= min_size: names.append(p); cols.append(ii)
    M = np.zeros((len(genes), len(names)), np.float32)
    for j, ii in enumerate(cols): M[ii, j] = 1
    return M, names

@torch.no_grad()
def ssgsea(C, M, w=0.25, bs=512):
    """M: one membership matrix or a list of them (ranks computed once). Returns array or list of arrays."""
    Ms = M if isinstance(M, list) else [M]
    N = C['X'].shape[1]; Mt = torch.tensor(np.concatenate(Ms, 1), device=dev); nS = Mt.sum(0); out = []
    for s in range(0, C['X'].shape[0], bs):
        xb = C['X'][s:s + bs].toarray()
        r = torch.tensor(rankdata(xb, axis=1), device=dev, dtype=torch.float32); wts = r ** w
        out.append((((r * wts) @ Mt) / (wts @ Mt) - (N * (N + 1) / 2 - r @ Mt) / (N - nS)).cpu().numpy())
    E = np.vstack(out)
    if not isinstance(M, list): return E
    cut = np.cumsum([0] + [m.shape[1] for m in Ms]); return [E[:, cut[i]:cut[i + 1]] for i in range(len(Ms))]

def lognorm(C):
    X = C['X']; tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    return sp.diags(1e4 / tot) @ X

@torch.no_grad()
def _ols_t(Yb, D):
    """Yb: genes x spots, D: genes x p (with intercept col). returns t-values (spots x p-1)."""
    n, p = D.shape; G = torch.linalg.inv(D.T @ D); B = G @ D.T @ Yb
    rss = ((Yb - D @ B) ** 2).sum(0); s2 = rss / (n - p)
    se = torch.sqrt(torch.diag(G)[:, None] * s2[None, :])
    return (B / se)[1:].T

@torch.no_grad()
def ulm_mlm(C, M, which=('ulm', 'mlm'), bs=512, Xn=None):
    Xn = lognorm(C) if Xn is None else Xn
    Mt = torch.tensor(M, device=dev, dtype=torch.float64); n = Mt.shape[0]; one = torch.ones(n, 1, device=dev, dtype=torch.float64)
    out = {m: [] for m in which}
    for s in range(0, Xn.shape[0], bs):
        Yb = torch.tensor(np.log1p(Xn[s:s + bs].toarray()).T, device=dev, dtype=torch.float64)
        if 'mlm' in which: out['mlm'].append(_ols_t(Yb, torch.cat([one, Mt], 1)).cpu().numpy())
        if 'ulm' in which:
            ym = Yb.mean(0, keepdim=True); yc = Yb - ym
            xm = Mt.mean(0, keepdim=True); xc = Mt - xm
            sxx = (xc ** 2).sum(0)[:, None]; sxy = xc.T @ yc
            b = sxy / sxx; rss = (yc ** 2).sum(0)[None, :] - b * sxy
            t = b / torch.sqrt(rss / (n - 2) / sxx)
            out['ulm'].append(t.T.cpu().numpy())
    return {m: np.vstack(v) for m, v in out.items()}
