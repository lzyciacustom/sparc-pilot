"""Null-calibrated niche enrichment: is a program's niche enrichment larger than that of generic gene sets?

For each patient group g and niche n, the 36 size-matched random gene sets (rand0, scored and decomposed with their own b_M)
give a reference distribution of enrichments. Curated program p:  z_gnp = (E_gnp - mean_rand_gn) / sd_rand_gn,
computed separately for the total measured enrichment (y) and each component (comp, depth, resid).
Test: Wilcoxon of z across patient groups, BH-FDR over niche x program. Calibration: each random set tested the same way
against the other 35 (leave-one-out) gives the empirical false-positive rate.
Also: H&E-predicted vs measured residual enrichment, curated vs random (two-stage models).
Inputs: comp/results/niche_group_long_{curated,random,curated_disjoint}.csv ; outputs comp/results/niche_null_*.csv
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = f'{WS}/comp/results'


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty_like(p); n = len(p)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; return np.minimum(q, 1)

KEYS = ['y', 'comp', 'depth', 'resid']


def zscores(cur, rnd, loo=False):
    out = []
    for (g, n), rr in rnd.groupby(['group', 'niche']):
        cc = cur[(cur.group == g) & (cur.niche == n)]
        for _, row in cc.iterrows():
            ref = rr[rr.program != row.program] if loo else rr
            out.append(dict(group=g, niche=n, program=row.program,
                            **{f'z_{k}': (row[k] - ref[k].mean()) / (ref[k].std() + 1e-9) for k in KEYS}))
    return pd.DataFrame(out)


def test(Z, label):
    rows = []
    for (n, p), g in Z.groupby(['niche', 'program']):
        if len(g) < 8: continue
        r = dict(set=label, niche=n, program=p, n_groups=len(g))
        for k in KEYS:
            v = g[f'z_{k}'].values; r[f'z_{k}_mean'] = v.mean(); r[f'{k}_pos'] = int((v > 0).sum()); r[f'{k}_p'] = wilcoxon(v).pvalue
        rows.append(r)
    T = pd.DataFrame(rows)
    for k in KEYS: T[f'{k}_q'] = bh(T[f'{k}_p'])
    return T


if __name__ == '__main__':
    cur = pd.read_csv(f'{R}/niche_group_long_curated.csv'); rnd = pd.read_csv(f'{R}/niche_group_long_random.csv')
    dis = pd.read_csv(f'{R}/niche_group_long_curated_disjoint.csv')
    Tc = test(zscores(cur, rnd), 'curated'); Tr = test(zscores(rnd, rnd, loo=True), 'random_loo')
    Td = test(zscores(dis, rnd), 'curated_disjoint')
    for T in (Tc, Tr, Td):
        print(f"[{T.set.iloc[0]}] cells {len(T)} | q<0.1 beyond random: total {(T.y_q < 0.1).sum()}  composition {(T.comp_q < 0.1).sum()}  "
              f"depth {(T.depth_q < 0.1).sum()}  residual {(T.resid_q < 0.1).sum()}")
    hits = Tc[Tc.resid_q < 0.1].set_index(['niche', 'program']); D = Td.set_index(['niche', 'program']).reindex(hits.index)
    print('curated residual hits (beyond random) replicated with gene-disjoint deconvolution:',
          int(((np.sign(D.z_resid_mean) == np.sign(hits.z_resid_mean)) & (D.resid_q < 0.1)).sum()), '/', len(hits))
    ty = Tc[Tc.y_q < 0.1]
    dom = np.where(ty.z_resid_mean.abs() > ty.z_comp_mean.abs(), 'residual', 'composition')
    print('program enrichments beyond random (total, q<0.1):', len(ty), '| larger component: composition', int((dom == 'composition').sum()),
          ', residual', int((dom == 'residual').sum()))
    Tc.to_csv(f'{R}/niche_null_curated.csv', index=False); Tr.to_csv(f'{R}/niche_null_random_loo.csv', index=False); Td.to_csv(f'{R}/niche_null_curated_disjoint.csv', index=False)
    show = Tc[(Tc.y_q < 0.1) | (Tc.resid_q < 0.1)].sort_values(['niche', 'z_resid_mean'])
    print(show[['niche', 'program', 'n_groups', 'z_y_mean', 'z_comp_mean', 'z_depth_mean', 'z_resid_mean', 'resid_pos', 'resid_q']].round(3).to_string())
