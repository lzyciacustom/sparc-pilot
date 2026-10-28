"""Niche-level decomposition of measured program enrichment into composition, depth and residual parts.

Niches: unsupervised, H&E only. H-optimus-1 spot features averaged over each spot and its 6 nearest neighbours,
centred per section (removes section-level stain/scanner offsets), PCA (30) on a pooled subsample, k-means (k=8).
Per spot (fold-specific calibration fitted on training sections, comp/comp_model.py):
    y = b(pi, d_ref)            composition part     (reference expectation at a fixed depth)
      + [b(pi, d) - b(pi, d_ref)]  depth part
      + delta                    residual (reference-relative, composition and depth removed)
Enrichment of niche n in section s for program p: mean over niche spots minus section mean, in units of the section SD of y.
The decomposition is exact: E_y = E_comp + E_depth + E_resid. Group level = mean over sections of a patient group.
Tests: residual enrichment across the 16 independent patient groups (Wilcoxon, BH-FDR over niche x program), with a random-gene-set
null (rand0 ULM with its own b_M) and a gene-disjoint deconvolution control. H&E-predicted decomposition from the composition-only
and residual (two-stage) models is compared with the measured one.
Outputs: comp/results/niche_*.csv
"""
import os, sys, glob, json, numpy as np, pandas as pd, h5py
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, WS); sys.path.insert(0, f'{WS}/comp')
import sparcmap_lib as L
from decon_resid import TYPES
from analyze_comp import BM, run_preds
LEAK = {'COLONMAP_8270_AS', 'TENX_immune_P1'}
crc = pd.read_csv(f'{WS}/crc_folds.csv').set_index('id'); grp = crc.group
OUT = f'{WS}/comp/results'; KN = 8; MIN_SPOTS = 30
rng = np.random.default_rng(2026)

# ---- niches from H&E features
F = {}
for sid in crc.index:
    with h5py.File(f'{WS}/features/hoptimus1_crc/{sid}.h5') as f:
        X = f['features'][:].astype(np.float32); xy = f['coords'][:].astype(np.float32)
        bc = np.array([b[0].decode() if isinstance(b[0], bytes) else str(b[0]) for b in f['barcode'][:]])
    _, nn_ = cKDTree(xy).query(xy, k=7); Xs = X[nn_].mean(1); Xs -= Xs.mean(0)
    F[sid] = (bc, Xs)
sub = np.concatenate([v[1][rng.choice(len(v[1]), min(2000, len(v[1])), replace=False)] for v in F.values()])
pca = PCA(30, random_state=0).fit(sub); km = KMeans(KN, n_init=10, random_state=0).fit(pca.transform(sub))
NICHE = {sid: pd.Series(km.predict(pca.transform(Xs)), index=bc) for sid, (bc, Xs) in F.items()}


def decompose(target, decon, cal_tag):
    cal = run_preds(cal_tag)[1]
    rows, spot_pi = [], []
    for sid in crc.index:
        k = int(crc.loc[sid, 'cv_fold']); a, c = cal[k]['ab'][:, 0], cal[k]['ab'][:, 1]; dref = float(cal[k]['d_ref'])
        t = pd.read_parquet(f'{WS}/targets_crc/{sid}.parquet').set_index('barcode')
        cp = pd.read_parquet(f'{WS}/comp/{decon}/{sid}.parquet').set_index('barcode')
        bc = NICHE[sid].index.intersection(t.index); t = t.loc[bc]; cp = cp.reindex(bc); nic = NICHE[sid].loc[bc].values
        Y = t[[f'{target}__{p}' for p in L.PROGRAMS]].values.astype(float); d = np.log1p(t.total_counts.values.astype(float))
        pi = cp[[f'comp__{x}' for x in TYPES]].values.astype(float); ok = np.isfinite(pi).all(1) & np.isfinite(Y).all(1)
        Y, d, pi, nic = Y[ok], d[ok], pi[ok], nic[ok]
        bref = a + c * BM[target](pi, dref); bfull = a + c * BM[target](pi, d)
        parts = dict(y=Y, comp=bref, depth=bfull - bref, resid=Y - bfull)
        sd = Y.std(0) + 1e-9
        for n in range(KN):
            m = nic == n
            if m.sum() < MIN_SPOTS: continue
            r = dict(sample=sid, group=grp[sid], niche=n, n_spots=int(m.sum()), frac=float(m.mean()))
            for key, A in parts.items():
                e = (A[m].mean(0) - A.mean(0)) / sd
                for j, p in enumerate(L.PROGRAMS): r[f'{key}__{p}'] = e[j]
            rows.append(r)
            spot_pi.append(dict(sample=sid, group=grp[sid], niche=n, **{x: pi[m, q].mean() for q, x in enumerate(TYPES)}))
    return pd.DataFrame(rows), pd.DataFrame(spot_pi)


def to_long(E):
    E = E[~E.group.isin(LEAK)]
    G = E.drop(columns=['sample', 'n_spots', 'frac']).groupby(['group', 'niche']).mean()
    L_ = []
    for key in ['y', 'comp', 'depth', 'resid']:
        s = G[[f'{key}__{p}' for p in L.PROGRAMS]]; s.columns = L.PROGRAMS
        L_.append(s.stack().rename(key))
    return pd.concat(L_, axis=1).reset_index().rename(columns={'level_2': 'program'})


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty_like(p); n = len(p)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; return np.minimum(q, 1)


def tests(Lg, label):
    out = []
    for (n, p), g in Lg.groupby(['niche', 'program']):
        if len(g) < 8: continue
        row = dict(set=label, niche=n, program=p, n_groups=len(g))
        for key in ['y', 'comp', 'depth', 'resid']:
            v = g[key].values; row[f'{key}_mean'] = v.mean()
            row[f'{key}_pos'] = int((v > 0).sum())
            row[f'{key}_p'] = wilcoxon(v).pvalue if np.any(v != 0) else 1.0
        out.append(row)
    T = pd.DataFrame(out)
    for key in ['y', 'comp', 'depth', 'resid']: T[f'{key}_q'] = bh(T[f'{key}_p'])
    ss = lambda c: (T[c] ** 2)
    T['share_comp'] = ss('comp_mean') / (ss('comp_mean') + ss('depth_mean') + ss('resid_mean'))
    return T


res = {}
for label, target, decon, tag in [('curated', 'ulm', 'out_parquet', 'delta__ulm__out_parquet__s0'),
                                  ('random', 'rand0ulm', 'out_parquet', 'delta__rand0ulm__out_parquet__s0'),
                                  ('curated_disjoint', 'ulm', 'out_disjoint', 'delta__ulm__out_disjoint__s0')]:
    E, Pn = decompose(target, decon, tag); Lg = to_long(E); T = tests(Lg, label); res[label] = (E, Lg, T)
    T.to_csv(f'{OUT}/niche_tests_{label}.csv', index=False); Lg.to_csv(f'{OUT}/niche_group_long_{label}.csv', index=False)
    if label == 'curated':
        comp_n = Pn[~Pn.group.isin(LEAK)].groupby(['group', 'niche'])[TYPES].mean().groupby('niche').mean()
        comp_n['n_groups'] = Pn[~Pn.group.isin(LEAK)].groupby('niche').group.nunique()
        comp_n['spot_frac'] = E.groupby('niche').frac.mean()
        comp_n.to_csv(f'{OUT}/niche_composition.csv'); print(comp_n.round(2).to_string())
    sig_y = (T.y_q < 0.1).sum(); sig_r = (T.resid_q < 0.1).sum()
    print(f'[{label}] niche x program cells tested {len(T)} | measured enrichment FDR<0.1: {sig_y} | residual FDR<0.1: {sig_r} '
          f'| median composition share {T.share_comp.median():.2f}', flush=True)

# composition share among significant measured enrichments; overlap of residual hits with the gene-disjoint control
T = res['curated'][2]; Td = res['curated_disjoint'][2].set_index(['niche', 'program'])
hits = T[T.resid_q < 0.1].set_index(['niche', 'program'])
rep = Td.reindex(hits.index)
print('curated residual hits replicated with gene-disjoint deconvolution (same sign, q<0.1):',
      int(((np.sign(rep.resid_mean) == np.sign(hits.resid_mean)) & (rep.resid_q < 0.1)).sum()), '/', len(hits))
sigy = T[T.y_q < 0.1]
print('among significant measured enrichments: median composition share', round(sigy.share_comp.median(), 2),
      '| composition-dominated (share>0.5):', int((sigy.share_comp > 0.5).sum()), '| residual-dominated (resid^2 > comp^2+depth^2):',
      int((sigy.resid_mean ** 2 > sigy.comp_mean ** 2 + sigy.depth_mean ** 2).sum()), '/', len(sigy))

# H&E-predicted decomposition (two-stage, seed 0) vs measured, niche x program, per group; curated and random gene sets
pc = run_preds('comp__ulm__out_parquet__s0')
for tgt, lab in [('ulm', 'curated'), ('rand0ulm', 'random')]:
    pdl = run_preds(f'delta__{tgt}__out_parquet__s0'); rows = []
    for sid, z in pc[0].items():
        if grp[sid] in LEAK or sid not in pdl[0]: continue
        k = int(z['fold']); cal = pdl[1][k]; a, c = cal['ab'][:, 0], cal['ab'][:, 1]; dref = float(cal['d_ref'])
        t = pd.read_parquet(f'{WS}/targets_crc/{sid}.parquet').set_index('barcode').reindex(z['barcode'])
        Y = t[[f'{tgt}__{p}' for p in L.PROGRAMS]].values.astype(float); sd = np.nanstd(Y, 0) + 1e-9
        nic = NICHE[sid].reindex(z['barcode']).values
        bh_ = a + c * BM[tgt](z['pi'], dref); dh = pdl[0][sid]['d'][pd.Index(pdl[0][sid]['barcode']).get_indexer(z['barcode'])]
        for n in range(KN):
            m = nic == n
            if m.sum() < MIN_SPOTS: continue
            r = dict(group=grp[sid], niche=n)
            for key, A in [('comp_hat', bh_), ('resid_hat', dh)]:
                e = (A[m].mean(0) - A.mean(0)) / sd
                for j, p in enumerate(L.PROGRAMS): r[f'{key}__{p}'] = e[j]
            rows.append(r)
    Ph = pd.DataFrame(rows).groupby(['group', 'niche']).mean()
    Lp = pd.concat([Ph[[f'{key}__{p}' for p in L.PROGRAMS]].set_axis(L.PROGRAMS, axis=1).stack().rename(key) for key in ['comp_hat', 'resid_hat']], axis=1)
    Lp = Lp.reset_index().rename(columns={'level_2': 'program'})
    M = res[lab][1].merge(Lp, on=['group', 'niche', 'program'])
    per = M.groupby('group').apply(lambda g: pd.Series(dict(r_comp=np.corrcoef(g.comp, g.comp_hat)[0, 1], r_resid=np.corrcoef(g.resid, g.resid_hat)[0, 1])))
    print(lab, 'H&E-predicted vs measured niche enrichment (per group r over niche x gene set): composition', per.r_comp.mean().round(3),
          '| residual', per.r_resid.mean().round(3), 'n groups', len(per))
    per.to_csv(f'{OUT}/niche_pred_vs_measured_by_group_{lab}.csv')
