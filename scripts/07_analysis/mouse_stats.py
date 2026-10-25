"""Mouse retraining pilot (scripts/05_mouse/mouse_ft.py): 22 non-PDX mouse Visium sections, ULM targets (3 seeds) and a
mouse random-gene-set null (rand0ulm, 1 seed). Unit: section (mean over programs, then seeds); paired Wilcoxon over sections.
Run from the repo root -> results/tables/mouse_ft_*.csv"""
import glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
M = ['pearson', 'pearson_centered', 'pearson_depthres']
mm = pd.read_csv('metadata/mouse_test_slides_meta.csv').set_index('id')
X = pd.concat([pd.read_csv(f) for f in glob.glob('results/raw/mouse_ft/*.csv')]); X['organ'] = X['sample'].map(mm.organ)
S = X.groupby(['target', 'condition', 'scheme', 'seed', 'sample', 'organ'])[M].mean().reset_index().groupby(['target', 'condition', 'scheme', 'sample', 'organ'])[M].mean().reset_index()
S.to_csv('results/tables/mouse_ft_per_section.csv', index=False)
S.groupby(['target', 'scheme', 'condition'])[M].mean().round(4).to_csv('results/tables/mouse_ft_summary.csv')
U = S[S.target == 'ulm'].set_index(['scheme', 'condition', 'sample']).pearson_depthres.sort_index()
N = S[S.target == 'rand0ulm'].set_index(['scheme', 'condition', 'sample']).pearson_depthres.sort_index()
def t(a, b, nm):
    d = (a - b).dropna(); return dict(comparison=nm, diff=d.mean(), sections_pos=f'{(d > 0).sum()}/{len(d)}', p_wilcoxon=wilcoxon(d).pvalue)
hu = U.loc[('none', 'human_only')]
rows = [t(hu, N.loc[('none', 'human_only')], 'human-only: real vs mouse random null')]
for sch in ['slide', 'dataset']:
    rows += [t(U.loc[(sch, 'mouse_only')], hu, f'{sch}: mouse-only vs human-only'), t(U.loc[(sch, 'human_ft')], hu, f'{sch}: human->mouse fine-tune vs human-only'),
             t(U.loc[(sch, 'human_ft')], U.loc[(sch, 'mouse_only')], f'{sch}: fine-tune vs mouse-only')]
    for c in ['mouse_only', 'human_ft']: rows.append(t(U.loc[(sch, c)], N.loc[(sch, c)], f'{sch}: {c} real vs random null'))
T = pd.DataFrame(rows); T.to_csv('results/tables/mouse_ft_tests.csv', index=False); print(T.round(4).to_string(index=False))
print(S.groupby(['target', 'scheme', 'condition']).pearson_depthres.mean().round(3).to_string())
