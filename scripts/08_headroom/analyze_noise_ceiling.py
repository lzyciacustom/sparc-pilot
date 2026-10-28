"""Noise ceiling vs achieved depth-adjusted accuracy (H-optimus-1, HEST-1k training), 16 patient groups."""
import numpy as np, pandas as pd, glob
from scipy.stats import wilcoxon
LEAK = {'COLONMAP_8270_AS', 'TENX_immune_P1'}
grp = pd.read_csv('crc_folds.csv').set_index('id')['group']
nc = pd.read_csv('results/noise_ceiling_long.csv')
r = nc.groupby(['sample', 'program', 'split', 'target']).r_half_depth.mean().reset_index()
r['rel'] = 2 * r.r_half_depth / (1 + r.r_half_depth)
r['ceiling'] = np.sqrt(r.rel.clip(lower=0))
W = r.pivot_table(index=['sample', 'program'], columns=['split', 'target'], values='ceiling')
W.columns = [f'ceil_{a}_{b}' for a, b in W.columns]
R = r.pivot_table(index=['sample', 'program'], columns=['split', 'target'], values='rel'); R.columns = [f'rel_{a}_{b}' for a, b in R.columns]
W = W.join(R)
def ach(t):
    f = [x for x in glob.glob(f'results/{t}__base__det__s[0-9].csv')]
    d = pd.concat([pd.read_csv(x) for x in f]); return d.groupby(['sample', 'program']).pearson_depthres.mean().rename(f'ach_{t}'), len(f)
for t in ['ulm', 'ssgsea', 'rand0ulm']:
    a, n = ach(t); W = W.join(a); print(t, 'seeds', n)
W = W.reset_index(); W['group'] = W['sample'].map(grp); W = W[~W.group.isin(LEAK)]
W.to_csv('results/noise_ceiling_by_section_program.csv', index=False)
cols = [c for c in W.columns if c.startswith(('ceil_', 'ach_', 'rel_'))]
G = W.groupby(['group', 'sample'])[cols].mean().groupby('group').mean()
print('16-group means:'); print(G.mean().round(3).to_string())
for t, c in [('ulm', 'ceil_count_ulm'), ('ulm', 'ceil_gene_ulm'), ('ssgsea', 'ceil_count_ssgsea'), ('rand0ulm', 'ceil_count_rand0ulm')]:
    print(f'achieved/ceiling {t} vs {c}: {G[f"ach_{t}"].mean()/G[c].mean():.2f}  headroom {G[c].mean()-G[f"ach_{t}"].mean():.3f}')
# per program (median over sections of 16 groups, after group averaging)
Pg = W.groupby(['group', 'program'])[cols].mean().groupby('program').mean()
Pg['frac_count'] = Pg.ach_ulm / Pg.ceil_count_ulm; Pg['frac_gene'] = Pg.ach_ulm / Pg.ceil_gene_ulm
Pg['headroom_count'] = Pg.ceil_count_ulm - Pg.ach_ulm
Pg = Pg.sort_values('ach_ulm', ascending=False); Pg.to_csv('results/noise_ceiling_by_program.csv')
print(Pg[['ceil_count_ulm', 'ceil_gene_ulm', 'ach_ulm', 'frac_count', 'headroom_count', 'ceil_count_ssgsea', 'ach_ssgsea']].round(3).to_string())
print('corr across programs: achieved ULM vs count ceiling', np.corrcoef(Pg.ach_ulm, Pg.ceil_count_ulm)[0, 1].round(3),
      ' vs gene ceiling', np.corrcoef(Pg.ach_ulm, Pg.ceil_gene_ulm)[0, 1].round(3))
G.to_csv('results/noise_ceiling_by_group.csv')
