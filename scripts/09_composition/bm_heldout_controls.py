"""Held-out-donor controls for the reference composition baseline b_M (addresses reviewer comments on the residual).
Pipeline as build_bM.py (pseudospots of 1 + Poisson(9) GSE132465 cells, transcript-fraction labels, log-uniform depth thinning,
library-size normalisation, log1p, ULM over a fixed gene universe, MLP (12 types + log depth) -> 36 programs), except:
  * b_M is trained on pseudospots from TRAINING donors only; all tests use cells from 8 HELD-OUT donors;
  * universe = 13,000 genes most often detected in training-donor cells (the original universe came from HEST CRC sections);
  * no affine calibration to Visium (tests are within the reference), i.e. b = b_M.
Tests (residual delta = y - b_M(pi, depth), in SD units of y over held-out null pseudospots):
  A  generalisation: held-out R^2 within training donors vs on held-out donors
  B  composition-only contrasts (null): per held-out donor, region B - region A residual difference (50 spots each), Wilcoxon across
     the 8 donors -> false-positive rate per program; pi known (transcript fraction), estimated (NNLS), or cell-count fraction
  C  planted state shifts at fixed composition: program genes scaled by f in one cell type in region B -> recovery rate and
     recovered / planted effect ratio; pi known or estimated
Run from <workdir>/comp after cache_gse132465.py (reads comp/gse132465_*, ../sparc_programs_msigdb.json, ../decon_resid.py).
Outputs: comp/results/bm_heldout_test{A,B,C}*.csv; summary printed.
Follow-up (three donor splits, RNA-content conversion): bm_heldout_splits.py."""
import os, json, sys, time, numpy as np, pandas as pd, scipy.sparse as sp, torch
from scipy.stats import wilcoxon
from torch import nn
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, WS)
import decon_resid as dr
C = f'{WS}/comp'; OUT = f'{C}/results'
SETS = json.load(open(f'{WS}/sparc_programs_msigdb.json'))['sets']
P = list(SETS); TYPES = dr.TYPES; K = len(TYPES); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026); torch.manual_seed(0)

def lognorm(X):
    X = X.astype(np.float64); s = np.asarray(X.sum(1)).ravel(); s[s == 0] = 1
    return sp.csr_matrix(sp.diags(1e4 / s) @ X).log1p()
def ulm(Xl, genes, sets, tmin=5):
    gi = {g: i for i, g in enumerate(genes)}; G = len(genes); names, W = [], []
    for p, gs in sets.items():
        ii = sorted({gi[g] for g in gs if g in gi})
        if len(ii) >= tmin: w = np.zeros(G); w[ii] = 1; names.append(p); W.append(w)
    W = np.array(W).T
    mx = np.asarray(Xl.mean(1)).ravel(); sxx = np.asarray(Xl.multiply(Xl).sum(1)).ravel() - G * mx ** 2
    mw = W.mean(0); sww = (W ** 2).sum(0) - G * mw ** 2
    sxw = np.asarray(Xl @ W) - G * mx[:, None] * mw[None, :]
    r = np.clip(sxw / np.sqrt(np.outer(sxx, sww) + 1e-30), -0.999999, 0.999999)
    return pd.DataFrame(r * np.sqrt((G - 2) / (1 - r ** 2)), columns=names).reindex(columns=list(sets)).values.astype(np.float32)

# ---------------- data, donor split, universe
G = sp.load_npz(f'{C}/gse132465_counts.npz').tocsr(); cells = pd.read_csv(f'{C}/gse132465_cells.csv')
genes = pd.read_csv(f'{C}/gse132465_genes.csv').iloc[:, 0].values
keep = cells.type.isin(TYPES).values; G = G[keep]; cells = cells[keep].reset_index(drop=True)
pats = sorted(cells.patient.unique())
cnt = pd.crosstab(cells.patient, cells.type).reindex(columns=TYPES, fill_value=0)
# held-out donors: 4 from the donors with near-complete type coverage (>=5 cells in >=10 types; mostly tumour+normal pairs)
# and 4 from the rest (>=5 cells in >=8 types), so training keeps full-coverage donors for the rare types
full = [p for p in pats if (cnt.loc[p] >= 5).sum() >= 10]; part = [p for p in pats if p not in full and (cnt.loc[p] >= 5).sum() >= 8]
HO = sorted(list(rng.choice(full, 4, replace=False)) + list(rng.choice(part, 4, replace=False))); TR = [p for p in pats if p not in HO]
tr = cells.patient.isin(TR).values
det = np.asarray((G[tr] > 0).mean(0)).ravel(); ui = np.sort(np.argsort(-det)[:13000]); ug = genes[ui]
G = G[:, ui].tocsr(); lib_full = np.asarray(G.sum(1)).ravel()
lab = cells.type.values; tix = np.array([TYPES.index(t) for t in lab])
print('donors', len(pats), 'train', len(TR), 'held-out', HO, flush=True)
print(cnt.loc[HO].to_string(), flush=True)

def simulate(cell_pool, pi, ncell, D, Gm=None):
    """cell_pool: dict type -> array of cell indices; pi: n x K; returns counts (csr), transcript frac, cell frac, log depth"""
    Gm = G if Gm is None else Gm
    rows, cols = [], []
    for i in range(len(pi)):
        k = rng.multinomial(ncell[i], pi[i])
        for t in range(K):
            if k[t]: rows += [i] * k[t]; cols += list(rng.choice(cell_pool[t], k[t]))
    n = len(pi); A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, Gm.shape[0]))
    X = (A @ Gm).tocsr().astype(np.float64)
    T = sp.csr_matrix((lib_full[cols], (rows, tix[cols])), shape=(n, K)).toarray(); frac = T / T.sum(1, keepdims=True)
    C = sp.csr_matrix((np.ones(len(cols)), (rows, tix[cols])), shape=(n, K)).toarray(); cfrac = C / C.sum(1, keepdims=True)
    tot = np.asarray(X.sum(1)).ravel(); p = np.minimum(1, D / tot)
    X.data = rng.binomial(X.data.astype(np.int64), np.repeat(p, np.diff(X.indptr))).astype(np.float64)
    return X, frac, cfrac, np.log1p(np.asarray(X.sum(1)).ravel())

def pi_prior(N):
    mode = rng.integers(0, 3, N); pi = np.empty((N, K))
    pi[mode == 0] = rng.dirichlet(np.ones(K), (mode == 0).sum()); pi[mode == 1] = rng.dirichlet(0.2 * np.ones(K), (mode == 1).sum())
    m2 = mode == 2; dom = rng.integers(0, K, m2.sum()); base = rng.dirichlet(0.5 * np.ones(K), m2.sum()) * 0.3
    base[np.arange(m2.sum()), dom] += 0.7; pi[m2] = base; return pi

# ---------------- train b_M on training donors
t0 = time.time(); pool_tr = {t: np.where(tr & (tix == t))[0] for t in range(K)}
N = 80000; piN = pi_prior(N); X, frac, cfrac, dep = simulate(pool_tr, piN, 1 + rng.poisson(9, N), np.exp(rng.uniform(np.log(1e2), np.log(1e5), N)))
Y = ulm(lognorm(X), ug, SETS); ok = np.isfinite(Y).all(1); X, frac, dep, Y = X[ok], frac[ok], dep[ok], Y[ok]; ntr = int(0.9 * len(Y))
dmu, dsd = dep[:ntr].mean(), dep[:ntr].std()
Xt = torch.tensor(np.column_stack([frac, (dep - dmu) / dsd]), dtype=torch.float32); Yt = torch.tensor(Y); mu, sd = Yt[:ntr].mean(0), Yt[:ntr].std(0)
net = nn.Sequential(nn.Linear(K + 1, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, len(P)))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
for ep in range(60):
    perm = torch.randperm(ntr)
    for b in range(0, ntr, 1024):
        j = perm[b:b + 1024]; loss = ((net(Xt[j]) - (Yt[j] - mu) / sd) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
net.eval()
def bM(pi, d):
    with torch.no_grad(): return (net(torch.tensor(np.column_stack([pi, (d - dmu) / dsd]), dtype=torch.float32)) * sd + mu).numpy()
def r2(yh, y): return 1 - ((yh - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)
r2_within = r2(bM(frac[ntr:], dep[ntr:]), Y[ntr:])
print(f'b_M trained {time.time()-t0:.0f}s; within-training-donor held-out R2 median {np.median(r2_within):.3f}', flush=True)

# reference profiles from TRAINING donors for NNLS (as decon_resid / run_all.py)
Pr, _ = dr.type_profiles(G[tr], lab[tr], pd.Index(ug)); Pr = Pr.loc[TYPES]
mk = dr.select_markers(Pr, pd.Index(ug), n=50); mg = pd.Index(sorted({g for v in mk.values() for g in v})); mgi = pd.Index(ug).get_indexer(mg)
Mcpm = (Pr[mg] / Pr.sum(1).values[:, None] * 1e6).values
def nnls_pi(X):
    lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
    cpm = (sp.diags(1e6 / lib) @ X)[:, mgi].toarray(); return np.nan_to_num(dr.nnls_deconv(cpm, Mcpm), nan=1.0 / K)

# ---------------- test A: held-out donors, prior compositions
ho = cells.patient.isin(HO).values; pool_ho = {t: np.where(ho & (tix == t))[0] for t in range(K)}
NA = 20000; piA = pi_prior(NA); XA, fA, cA, dA = simulate(pool_ho, piA, 1 + rng.poisson(9, NA), np.exp(rng.uniform(np.log(1e2), np.log(1e5), NA)))
YA = ulm(lognorm(XA), ug, SETS); okA = np.isfinite(YA).all(1); XA, fA, cA, dA, YA = XA[okA], fA[okA], cA[okA], dA[okA], YA[okA]
SDY = YA.std(0); piA_hat = nnls_pi(XA)
dfA = pd.DataFrame({'program': P, 'r2_within_train_donors': r2_within, 'r2_heldout_known_pi': r2(bM(fA, dA), YA),
                    'r2_heldout_nnls_pi': r2(bM(piA_hat, dA), YA), 'r2_heldout_cell_fraction': r2(bM(cA, dA), YA),
                    'mean_resid_sd_known_pi': ((YA - bM(fA, dA)).mean(0)) / SDY, 'mean_resid_sd_nnls_pi': ((YA - bM(piA_hat, dA)).mean(0)) / SDY})
dfA.to_csv(f'{OUT}/bm_heldout_testA_generalisation.csv', index=False)
print(dfA.drop(columns='program').median().round(3).to_string(), flush=True)
print('NNLS composition r (held-out pseudospots), median over types:', np.median([np.corrcoef(piA_hat[:, k], fA[:, k])[0, 1] for k in range(K)]).round(3), flush=True)

# ---------------- contrast machinery (per held-out donor, 2 regions x NS spots, Visium-like depth)
NS = 50
def region(donor, centre, Gm=None, conc=30.0):
    pool = {t: np.where((cells.patient.values == donor) & (tix == t))[0] for t in range(K)}
    avail = np.array([len(pool[t]) > 0 for t in range(K)]); c = centre * avail; c = c / c.sum()
    pi = np.array([rng.dirichlet(conc * c[avail] + 1e-3) for _ in range(NS)]); full = np.zeros((NS, K)); full[:, avail] = pi
    for t in range(K):
        if not avail[t]: pool[t] = pool[np.where(avail)[0][0]]
    X, f, cf, d = simulate(pool, full, 1 + rng.poisson(9, NS), np.exp(rng.uniform(np.log(1e3), np.log(2e4), NS)), Gm)
    y = ulm(lognorm(X), ug, SETS); return y, f, cf, d, nnls_pi(X)
def contrast(cA, cB, GmB=None):
    """returns dict kind -> donors x programs array of (mean_B - mean_A)/SDY"""
    out = {k: [] for k in ['total', 'resid_known', 'resid_nnls', 'resid_cellfrac', 'comp_known']}
    for dn in HO:
        ya, fa, ca, da, pa = region(dn, cA); yb, fb, cb, db, pb = region(dn, cB, GmB)
        m = lambda a, b: (np.nanmean(b, 0) - np.nanmean(a, 0)) / SDY
        out['total'].append(m(ya, yb)); out['comp_known'].append(m(bM(fa, da), bM(fb, db)))
        out['resid_known'].append(m(ya - bM(fa, da), yb - bM(fb, db)))
        out['resid_nnls'].append(m(ya - bM(pa, da), yb - bM(pb, db)))
        out['resid_cellfrac'].append(m(ya - bM(ca, da), yb - bM(cb, db)))
    return {k: np.array(v) for k, v in out.items()}
def wp(x):
    x = x[np.isfinite(x)]
    return wilcoxon(x).pvalue if len(x) >= 6 and np.any(x != 0) else np.nan

# ---------------- test B: composition-only contrasts
rows = []
for rep in range(40):
    cA, cB = rng.dirichlet(np.ones(K)), rng.dirichlet(np.ones(K))
    o = contrast(cA, cB)
    for j, p in enumerate(P):
        rows.append(dict(rep=rep, program=p, total=o['total'][:, j].mean(), comp=o['comp_known'][:, j].mean(),
                         p_total=wp(o['total'][:, j]), **{f'resid_{k}': o[f'resid_{k}'][:, j].mean() for k in ['known', 'nnls', 'cellfrac']},
                         **{f'p_{k}': wp(o[f'resid_{k}'][:, j]) for k in ['known', 'nnls', 'cellfrac']}))
    if rep % 10 == 9: print('test B rep', rep + 1, flush=True)
B = pd.DataFrame(rows); B.to_csv(f'{OUT}/bm_heldout_testB_null_contrasts.csv', index=False)
for k in ['total', 'known', 'nnls', 'cellfrac']:
    col = 'p_total' if k == 'total' else f'p_{k}'
    print(f'B FPR(p<0.05) {k:9s} {np.nanmean(B[col] < 0.05):.3f}   median |effect| {np.nanmedian(np.abs(B["total" if k=="total" else "resid_"+k])):.3f} SD', flush=True)

# ---------------- test C: planted state shifts at fixed composition
PLANT = [('GSE26495_PD1HIGH_VS_PD1LOW_CD8_TCELL_UP', 'T_NK'), ('REACTOME_COLLAGEN_FORMATION', 'Fibroblast'),
         ('HALLMARK_E2F_TARGETS', 'Epi_tumor'), ('HALLMARK_INTERFERON_GAMMA_RESPONSE', 'Mono_Mac')]
gidx = {g: i for i, g in enumerate(ug)}; rows = []
for prog, ttype in PLANT:
    pg = np.array(sorted({gidx[g] for g in SETS[prog] if g in gidx})); tcells = np.where(lab == ttype)[0]
    for f in [1.25, 1.5, 2.0]:
        Gs = G.copy(); rr = np.repeat(np.arange(G.shape[0]), np.diff(G.indptr))
        msk = np.isin(rr, tcells) & np.isin(G.indices, pg)          # non-zero counts of program genes in the target cell type
        Gs.data = Gs.data.copy(); Gs.data[msk] = rng.poisson(Gs.data[msk] * f).astype(Gs.data.dtype)
        for rep in range(10):
            c = rng.dirichlet(np.ones(K)); c[TYPES.index(ttype)] += 0.3; c /= c.sum()
            o = contrast(c, c, GmB=Gs); j = P.index(prog)
            others = [i for i in range(len(P)) if i != j]
            rows.append(dict(program=prog, cell_type=ttype, fold=f, rep=rep, total=o['total'][:, j].mean(), comp=o['comp_known'][:, j].mean(),
                             resid_known=o['resid_known'][:, j].mean(), resid_nnls=o['resid_nnls'][:, j].mean(),
                             p_known=wp(o['resid_known'][:, j]), p_nnls=wp(o['resid_nnls'][:, j]),
                             offtarget_fpr_known=np.nanmean([wp(o['resid_known'][:, i]) < 0.05 for i in others])))
        print('test C', prog[:30], f, flush=True)
C = pd.DataFrame(rows); C.to_csv(f'{OUT}/bm_heldout_testC_planted_shifts.csv', index=False)
s = C.groupby(['program', 'fold']).agg(total=('total', 'mean'), resid_known=('resid_known', 'mean'), resid_nnls=('resid_nnls', 'mean'),
        recov_known=('p_known', lambda x: np.mean(x < 0.05)), recov_nnls=('p_nnls', lambda x: np.mean(x < 0.05)),
        offtarget_fpr=('offtarget_fpr_known', 'mean')).round(3)
s.to_csv(f'{OUT}/bm_heldout_testC_summary.csv'); print(s.to_string(), flush=True)
print(f'done {time.time()-t0:.0f}s')
