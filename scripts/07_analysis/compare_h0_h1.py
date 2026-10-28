"""Paired H-optimus-0 vs H-optimus-1 comparison over 16 patient groups (the two groups with linked training slides excluded).
Unit: per-slide mean over 36 programs -> mean over seeds -> mean per patient group; paired Wilcoxon over groups.
usage (repo root): python scripts/07_analysis/compare_h0_h1.py   -> results/tables/h1/h0_vs_h1_{crconly,hest1k}_16groups.csv"""
import os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
LEAK = {'COLONMAP_8270_AS', 'TENX_immune_P1'}; M = ['pearson', 'pearson_depthres']; OUT = 'results/tables/h1'; os.makedirs(OUT, exist_ok=True)
grp = pd.read_csv('metadata/crc_folds.csv').set_index('id')['group']; grp = grp[~grp.isin(LEAK)]
T = ['ssgsea', 'ulm', 'mlm', 'ssgsea_depthres', 'rand0ssgsea', 'rand1ssgsea', 'rand2ssgsea', 'rand0ulm']
H0all = pd.read_parquet('results/raw/hest1k_wave1_long.parquet'); H1all = pd.read_parquet('results/raw/hest1k_h1_long.parquet')


def per_group(df):
    s = df.groupby(['seed', 'sample'])[M].mean().reset_index().groupby('sample')[M].mean()
    return s.groupby(s.index.map(grp)).mean()


def compare(train):
    sel = lambda D: D[(D.train == train) & (D.model == 'base') & (D['head'] == 'det')]
    G = {}
    for enc, D in [('H0', sel(H0all)), ('H1', sel(H1all))]:
        for t in T:
            x = D[D.target == t]
            if len(x) and x.fold.nunique() == 6: G[(enc, t)] = per_group(x)
    rows = []
    for t in T:
        if ('H0', t) in G and ('H1', t) in G:
            for m in M:
                a, b = G[('H1', t)][m], G[('H0', t)][m]; d = (a - b).dropna()
                rows.append(dict(contrast=f'H1 - H0 [{t}]', metric=m, H0=b.mean(), H1=a.mean(), diff=d.mean(), pos=f'{(d > 0).sum()}/{len(d)}', p=wilcoxon(d).pvalue))

    def rnd(enc, kind):
        ks = [k for k in ([f'rand{i}ssgsea' for i in range(3)] if kind == 'ssgsea' else ['rand0ulm']) if (enc, k) in G]
        return sum(G[(enc, k)] for k in ks) / len(ks) if ks else None
    for enc in ['H0', 'H1']:
        for a, b, lab in [('ulm', 'ssgsea', 'ULM - ssGSEA'), ('ssgsea_depthres', 'ssgsea', 'depth-regressed - ssGSEA'), ('mlm', 'ssgsea', 'MLM - ssGSEA')]:
            if (enc, a) in G and (enc, b) in G:
                d = (G[(enc, a)].pearson_depthres - G[(enc, b)].pearson_depthres).dropna()
                rows.append(dict(contrast=f'{enc}: {lab}', metric='pearson_depthres', diff=d.mean(), pos=f'{(d > 0).sum()}/{len(d)}', p=wilcoxon(d).pvalue))
        for kind in ['ssgsea', 'ulm']:
            R = rnd(enc, kind)
            if R is not None and (enc, kind) in G:
                for m in M:
                    d = (G[(enc, kind)][m] - R[m]).dropna()
                    rows.append(dict(contrast=f'{enc}: real - random [{kind}]', metric=m, H0=R[m].mean() if enc == 'H0' else np.nan,
                                     H1=R[m].mean() if enc == 'H1' else np.nan, diff=d.mean(), pos=f'{(d > 0).sum()}/{len(d)}', p=wilcoxon(d).pvalue))
    return pd.DataFrame(rows)


for train, tag in [('crc_only', 'crconly'), ('hest1k', 'hest1k')]:
    out = compare(train); out.to_csv(f'{OUT}/h0_vs_h1_{tag}_16groups.csv', index=False)
    print(tag); print(out[out.contrast.str.startswith('H1 - H0')].round(4).to_string(index=False))
