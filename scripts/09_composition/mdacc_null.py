"""Random-set-calibrated MDACC contrasts: per section, a curated program's paired region difference is expressed as a z-score
against the 36 random gene sets' differences in the same section and component; Wilcoxon of z across sections, BH over programs.
Calibration of the calibration: each random set against the other 35 (leave-one-out)."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, f'{WS}/comp'); sys.path.insert(0, WS)
from mdacc_decomp import region_means, bh, CONTRASTS
import sparcmap_lib as L
OUT = f'{WS}/comp/results'; KEYS = ['y', 'comp', 'depth', 'resid']
Rc, _, _ = region_means('ulm', 'comp'); Rr, _, _ = region_means('rand0ulm', 'comp'); Rd, _, _ = region_means('ulm', 'compdj')

def diffs(R, a, b):
    A = R[R.region == a].set_index('sample'); B = R[R.region == b].set_index('sample'); s = A.index.intersection(B.index)
    return {k: (A.loc[s, [f'{k}__{p}' for p in L.PROGRAMS]].values - B.loc[s, [f'{k}__{p}' for p in L.PROGRAMS]].values) for k in KEYS}, list(s)

rows = []
for name, a, b, paired in CONTRASTS:
    if not paired: continue
    for lab, R in [('curated', Rc), ('curated_disjoint', Rd), ('random_loo', Rr)]:
        Dc, s = diffs(R, a, b); Dr, s2 = diffs(Rr, a, b); assert s == s2
        for k in KEYS:
            for j, p in enumerate(L.PROGRAMS):
                ref = np.delete(Dr[k], j, axis=1) if lab == 'random_loo' else Dr[k]
                z = (Dc[k][:, j] - ref.mean(1)) / (ref.std(1) + 1e-9)
                pv = wilcoxon(z).pvalue if len(z) >= 5 else np.nan
                rows.append(dict(contrast=name, set=lab, component=k, program=p, n=len(z), z_mean=z.mean(), pos=int((z > 0).sum()), p=pv, raw_diff=Dc[k][:, j].mean()))
T = pd.DataFrame(rows); T['q'] = T.groupby(['contrast', 'set', 'component']).p.transform(lambda s: bh(s.fillna(1)))
T.to_csv(f'{OUT}/mdacc_null_calibrated.csv', index=False)
print(T.groupby(['contrast', 'set', 'component']).q.apply(lambda s: int((s < 0.1).sum())).unstack('component').to_string())
H = T[(T.set == 'curated') & (T.q < 0.1) & T.component.isin(['y', 'resid'])].sort_values(['contrast', 'component', 'p'])
print(H[['contrast', 'component', 'program', 'n', 'raw_diff', 'z_mean', 'pos', 'q']].round(3).to_string())
D = T[T.set == 'curated_disjoint'].set_index(['contrast', 'component', 'program'])
hh = H.set_index(['contrast', 'component', 'program'])
print('replicated with gene-disjoint deconvolution (q<0.1, same sign):', int(((D.loc[hh.index].q < 0.1) & (np.sign(D.loc[hh.index].z_mean) == np.sign(hh.z_mean))).sum()), '/', len(hh))
