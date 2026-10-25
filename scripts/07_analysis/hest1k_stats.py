"""All HEST-1k-scale statistics (58 CRC test slides, 18 patient groups) from the long result tables.
Unit of analysis: per-slide mean over 36 programs -> mean over seeds -> mean per patient group; paired Wilcoxon over groups.
Run from the repo root:  python scripts/07_analysis/hest1k_stats.py   -> results/tables/hest1k_*.csv"""
import os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
M = ['pearson', 'pearson_centered', 'pearson_depthres']
RAW, OUT = 'results/raw', 'results/tables'; os.makedirs(OUT, exist_ok=True)
grp = pd.read_csv('metadata/crc_folds.csv').set_index('id')['group']
LEAK = {'COLONMAP_8270_AS', 'TENX_immune_P1'}          # test groups with linked slides in the training pool (docs/AUDIT.md)

# ---------------- wave 1 (hest1k_run.py: targets x spatial models x heads; plus random-gene-set nulls) ----------------
D = pd.read_parquet(f'{RAW}/hest1k_wave1_long.parquet')
KEY = ['train', 'target', 'model', 'head']
SL = D.groupby(KEY + ['seed', 'sample'])[M].mean().reset_index().groupby(KEY + ['sample'])[M].mean().reset_index()
G = SL.assign(group=SL['sample'].map(grp)).groupby(KEY + ['group'])[M].mean()
cond = lambda tr, t, mo='base', h='det': G.loc[(tr, t, mo, h)]
summ = D.groupby(KEY + ['seed', 'sample'])[M].mean().reset_index().groupby(KEY)[M + ['seed']].agg({**{m: 'mean' for m in M}, 'seed': 'nunique'}).rename(columns={'seed': 'n_seeds'})
summ.round(4).to_csv(f'{OUT}/hest1k_condition_summary.csv')
def test(A, B, drop=()):
    d = (A - B).dropna(); d = d[~d.index.isin(drop)]
    return dict(A=A[~A.index.isin(drop)].mean(), B=B[~B.index.isin(drop)].mean(), diff=d.mean(), groups_pos=f'{(d > 0).sum()}/{len(d)}', p_wilcoxon=wilcoxon(d).pvalue)
comps = [('targets', 'ULM vs ssGSEA', ('hest1k', 'ulm'), ('hest1k', 'ssgsea')), ('targets', 'ULM vs MLM', ('hest1k', 'ulm'), ('hest1k', 'mlm')),
         ('targets', 'MLM vs ssGSEA', ('hest1k', 'mlm'), ('hest1k', 'ssgsea')), ('targets', 'ULM vs ssGSEA-depthres', ('hest1k', 'ulm'), ('hest1k', 'ssgsea_depthres')),
         ('scale', 'ULM hest1k vs crc_only', ('hest1k', 'ulm'), ('crc_only', 'ulm')), ('scale', 'ssGSEA hest1k vs crc_only', ('hest1k', 'ssgsea'), ('crc_only', 'ssgsea')),
         ('scale', 'MLM hest1k vs crc_only', ('hest1k', 'mlm'), ('crc_only', 'mlm'))]
for t in ['ssgsea', 'ulm']:
    for mo in ['mlp', 'concert', 'gpk16', 'gpk48']: comps.append(('spatial', f'{t}: {mo} vs base', ('hest1k', t, mo), ('hest1k', t)))
    comps.append(('head', f'{t}: gauss vs det', ('hest1k', t, 'base', 'gauss'), ('hest1k', t)))
rows = []
for fam, nm, a, b in comps:
    for m in M: rows.append(dict(family=fam, comparison=nm, metric=m, **test(cond(*a)[m], cond(*b)[m])))
for tr in ['hest1k', 'crc_only']:
    for real, rands in [('ssgsea', ['rand0ssgsea', 'rand1ssgsea', 'rand2ssgsea']), ('ulm', ['rand0ulm'])]:
        B = sum(cond(tr, r) for r in rands) / len(rands)
        for m in M: rows.append(dict(family='random_null', comparison=f'{tr} {real} real vs random ({len(rands)} draws)', metric=m, **test(cond(tr, real)[m], B[m])))
W1 = pd.DataFrame(rows); W1.to_csv(f'{OUT}/hest1k_wave1_paired_tests.csv', index=False)
# per-program real - random (seed-averaged slide means; depth-res and centred)
pp = []
for tr in ['hest1k', 'crc_only']:
    for real, rands in [('ssgsea', ['rand0ssgsea', 'rand1ssgsea', 'rand2ssgsea']), ('ulm', ['rand0ulm'])]:
        x = D[(D.train == tr) & (D.model == 'base') & (D['head'] == 'det')]
        r = x[x.target == real].groupby('program')[M].mean(); z = x[x.target.isin(rands)].groupby('program')[M].mean()
        pp.append((r - z).assign(train=tr, target=real))
pd.concat(pp).reset_index().to_csv(f'{OUT}/hest1k_real_vs_random_per_program.csv', index=False)

# ---------------- wave 2 (hest1k_run2.py: upgrades, context, decoupling, external mouse) ----------------
R = pd.read_parquet(f'{RAW}/hest1k_wave2_long.parquet'); R['cond'] = R.tag.str.split('__').str[:2].str.join('__')
def gt(c):
    x = R[R.cond == c]; s = x.groupby(['seed', 'sample'])[M].mean().reset_index().groupby('sample')[M].mean()
    return s.groupby(s.index.map(grp)).mean()
W2C = [('ulm__tn-slide', 'ulm__tn-robust', 'slide z-score targets', 'upgrade'), ('ulm__tn-robust+tech', 'ulm__tn-robust', 'technology one-hot', 'upgrade'),
       ('ulm__tn-robust+progw', 'ulm__tn-robust', 'program reliability weights', 'upgrade'), ('ulm__tn-slide+tech+progw', 'ulm__tn-robust', 'all three combined', 'upgrade'),
       ('ulm__tn-robust+gene_aux', 'ulm__tn-robust', 'gene auxiliary loss', 'upgrade'), ('ulm__tn-robust+genes', 'ulm__tn-robust', 'gene decoder + ULM readout (2 seeds)', 'upgrade'),
       ('ulm__tn-robust+ctx', 'ulm__tn-robust', '448 um context (HEST-1k)', 'context'), ('ulm__tn-robust+ctx+crc_only', 'ulm__tn-robust+crc_only', '448 um context (CRC-only)', 'context'),
       ('ulm__tn-robust+decoup-res', 'ulm__tn-robust+decoup-randres', 'composition residual: real vs random', 'decoupling'),
       ('ulm__tn-robust+decoup-res', 'ulm__tn-robust+crc_only', 'composition residual vs full ULM (CRC-only)', 'decoupling')]
rows = []
for a, b, nm, fam in W2C:
    A, B = gt(a), gt(b)
    for m in M: rows.append(dict(family=fam, comparison=nm, metric=m, **test(A[m], B[m])))
W2 = pd.DataFrame(rows); W2.to_csv(f'{OUT}/hest1k_wave2_paired_tests.csv', index=False)
chk = gt('ulm__tn-robust')['pearson_depthres'].mean(), cond('hest1k', 'ulm')['pearson_depthres'].mean()
# decoupling per program
res = R[R.cond == 'ulm__tn-robust+decoup-res'].groupby('program')[M].mean(); rr = R[R.cond == 'ulm__tn-robust+decoup-randres'].groupby('program')[M].mean()
(res - rr).add_suffix('_real_minus_random').join(res.add_suffix('_real')).to_csv(f'{OUT}/decoupling_residual_per_program.csv')
# external mouse (human-trained) per organ
mm = pd.read_csv('metadata/mouse_test_slides_meta.csv').set_index('id')
ext = []
for c in ['ulm__tn-robust+ext-mouse', 'ssgsea__tn-robust+ext-mouse']:
    x = R[R.cond == c]; s = x.groupby(['seed', 'sample'])[M].mean().groupby('sample').mean(); s['organ'] = mm.organ.reindex(s.index).values
    ext.append(s.groupby('organ')[M].mean().assign(n=s.groupby('organ').size(), target=c.split('__')[0]))
pd.concat(ext).to_csv(f'{OUT}/mouse_external_by_organ.csv')

# ---------------- leakage sensitivity: drop the 2 affected test groups ----------------
rows = []
for fam, nm, a, b in comps[:7] + [c for c in comps if c[0] == 'spatial'][:4]:
    for lab, dr in [('all 18', ()), ('excluding 2 leaked groups', LEAK)]: rows.append(dict(comparison=nm, groups=lab, **test(cond(*a).pearson_depthres, cond(*b).pearson_depthres, dr)))
for real, rands in [('ssgsea', ['rand0ssgsea', 'rand1ssgsea', 'rand2ssgsea']), ('ulm', ['rand0ulm'])]:
    B = sum(cond('hest1k', r) for r in rands) / len(rands)
    for lab, dr in [('all 18', ()), ('excluding 2 leaked groups', LEAK)]: rows.append(dict(comparison=f'{real} real vs random (hest1k)', groups=lab, **test(cond('hest1k', real).pearson_depthres, B.pearson_depthres, dr)))
for a, b, nm, fam in W2C[:7]:
    for lab, dr in [('all 18', ()), ('excluding 2 leaked groups', LEAK)]: rows.append(dict(comparison=nm, groups=lab, **test(gt(a).pearson_depthres, gt(b).pearson_depthres, dr)))
pd.DataFrame(rows).to_csv(f'{OUT}/audit_leakage_sensitivity.csv', index=False)
print('run2 base vs run1 base (depth-res, group mean): %.4f vs %.4f' % chk)
print(W1[W1.metric == 'pearson_depthres'][['comparison', 'A', 'B', 'diff', 'groups_pos', 'p_wilcoxon']].round(4).to_string(index=False))
print(W2[W2.metric == 'pearson_depthres'][['comparison', 'A', 'B', 'diff', 'groups_pos', 'p_wilcoxon']].round(4).to_string(index=False))
