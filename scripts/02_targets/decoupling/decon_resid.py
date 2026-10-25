"""Composition vs state decoupling targets for 58 HEST-1k CRC slides.
Reference: GSE132465 (Lee et al. 2020 Nat Genet, SMC cohort) raw UMI + annotation.
Deconvolution: per-slide NNLS of CPM-normalised spot counts on reference cell-type mean
profiles over the union of top-50 marker genes per type (genes restricted to slide ∩ reference).
Residualisation: per slide, OLS of each ulm__<program> on proportions (intercept, drop one type).
"""
import warnings, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.optimize import nnls
warnings.filterwarnings('ignore')

TYPES = ['Epi_tumor', 'Epi_normal', 'Fibroblast', 'Endothelial', 'Pericyte_SMC', 'Glial',
         'T_NK', 'B', 'Plasma', 'Mono_Mac', 'DC', 'Mast']
DROP_TYPE = 'Epi_tumor'  # reference level dropped (sum-to-one collinearity)

def map_type(ct, sub, cls):
    if ct == 'Epithelial cells':
        return 'Epi_tumor' if cls == 'Tumor' else 'Epi_normal'
    if ct == 'Stromal cells':
        if sub in ('Myofibroblasts', 'Stromal 1', 'Stromal 2', 'Stromal 3'): return 'Fibroblast'
        if sub in ('Pericytes', 'Smooth muscle cells'): return 'Pericyte_SMC'
        if sub == 'Enteric glial cells': return 'Glial'
        if 'ECs' in sub: return 'Endothelial'
        return None
    if ct == 'T cells': return 'T_NK'
    if ct == 'B cells': return 'Plasma' if 'Plasma' in sub else 'B'
    if ct == 'Myeloids': return 'DC' if sub == 'cDC' else 'Mono_Mac'
    if ct == 'Mast cells': return 'Mast'
    return None

def type_profiles(G, labels, genes):
    """G: cells x genes raw counts (csr). Returns DataFrame types x genes of mean per-cell fractions,
    plus mean of log1p(CPM) for marker ranking."""
    lib = np.asarray(G.sum(1)).ravel()
    Gn = sp.diags(1.0 / lib) @ G
    prof, nc = {}, {}
    for t in TYPES:
        idx = np.where(labels == t)[0]
        prof[t] = np.asarray(Gn[idx].mean(0)).ravel(); nc[t] = len(idx)
    return pd.DataFrame(prof, index=genes).T, nc

def select_markers(P, genes_avail, n=50, min_frac=2e-5):
    """Top-n genes per type by log2 FC of mean fraction vs. the max over other types,
    requiring mean fraction >= min_frac (20 CPM) in the type. P: types x genes (already restricted)."""
    P = P.loc[:, genes_avail]
    eps = 1e-6
    out = {}
    for t in P.index:
        other = P.drop(index=t).max(0)
        lfc = np.log2((P.loc[t] + eps) / (other + eps))
        lfc = lfc[(P.loc[t] >= min_frac)]
        out[t] = lfc.sort_values(ascending=False).index[:n].tolist()
    return out

def nnls_deconv(Y, M):
    """Y: spots x genes (CPM over slide∩ref genes), M: types x genes (CPM). Genes scaled by
    sqrt(mean over types) (Poisson-type variance weighting)."""
    s = np.sqrt(M.mean(0)); s[s == 0] = 1  # sqrt (Poisson-type) weights
    A = (M / s).T
    Ys = Y / s
    W = np.zeros((Y.shape[0], M.shape[0]))
    for i in range(Y.shape[0]):
        W[i], _ = nnls(A, Ys[i])
    tot = W.sum(1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        Pm = W / tot
    Pm[tot.ravel() == 0] = np.nan
    return Pm

def residualise(T, C):
    """T: n x p program scores, C: n x k proportions (rows complete). OLS with intercept,
    dropping DROP_TYPE. Returns residuals and per-program R^2."""
    X = np.column_stack([np.ones(len(C)), C])
    beta, *_ = np.linalg.lstsq(X, T, rcond=None)
    R = T - X @ beta
    sst = ((T - T.mean(0)) ** 2).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        r2 = 1 - (R ** 2).sum(0) / sst
    return R, r2

SANITY = {'Epithelial': (['Epi_tumor', 'Epi_normal'], ['EPCAM', 'KRT20']),
          'Fibroblast': (['Fibroblast'], ['COL1A1']),
          'T_NK': (['T_NK'], ['CD3E']),
          'Plasma': (['Plasma'], ['JCHAIN', 'MZB1']),
          'B': (['B'], ['MS4A1']),
          'Mono_Mac': (['Mono_Mac'], ['CD68', 'C1QA']),
          'Endothelial': (['Endothelial'], ['PECAM1', 'VWF'])}
