"""Aggregate CV results for A3/A4: per-slide means, by task, READ, paired Wilcoxon (slides and 16 patient groups)."""
import os, sys, glob, re; sys.path.append(os.getcwd())
import numpy as np, pandas as pd, sparcmap_lib as L
from scipy.stats import wilcoxon
folds = L.wt_folds(); MET = ['pearson', 'pearson_centered', 'pearson_depthres']
runs = []
b = pd.read_csv('baseline/baseline_sparcmap_ssgsea_cv.csv'); b['target'] = 'ssgsea'; b['seed'] = 0; runs.append(b)
for f in glob.glob('results/folds/cv_*_f[0-3].csv'):
    tgt, seed, k = re.match(r'.*cv_(.+)_s(\d+)_f(\d)\.csv', f).groups()
    if tgt == 'ssgsea' and seed == '0': continue
    r = pd.read_csv(f); r['target'] = tgt; r['seed'] = int(seed); runs.append(r)
for f in glob.glob('results/cv_*_s*.csv'):   # final CSVs from runs without per-fold files
    tgt, seed = re.match(r'.*cv_(.+)_s(\d+)\.csv', f).groups()
    if any(os.path.exists(x) for x in glob.glob(f'results/folds/cv_{tgt}_s{seed}_f*.csv')): continue
    runs.append(pd.read_csv(f))
R = pd.concat(runs, ignore_index=True)
R = R.drop(columns=[c for c in ['task', 'group'] if c in R]).merge(folds[['sample', 'task', 'group']], on='sample')
cnt = R.groupby(['target', 'seed']).fold.nunique(); complete = cnt[cnt == 4].index
R = R.set_index(['target', 'seed']).loc[complete].reset_index()
R.to_csv('results/A34_cv_long_per_slide_program.csv', index=False)
S = R.groupby(['target', 'seed', 'sample', 'task', 'group'])[MET].mean().reset_index()
S['kind'] = np.where(S.target.str.startswith('rand'), 'random', 'real')
S['method'] = S.target.str.replace(r'^rand\d_', '', regex=True)
S.to_csv('results/A34_cv_per_slide.csv', index=False)
# summary table per target (mean over seeds first)
Sm = S.groupby(['target', 'kind', 'method', 'sample', 'task', 'group'])[MET].mean().reset_index()
summ = []
for t, g in Sm.groupby('target'):
    row = dict(target=t, n_seeds=S[S.target == t].seed.nunique(), n_slides=g['sample'].nunique())
    for m in MET:
        row[f'{m}_all'] = g[m].mean(); row[f'{m}_groupmean'] = g.groupby('group')[m].mean().mean(); row[f'{m}_READ'] = g[g.task == 'READ'][m].mean()
    summ.append(row)
summ = pd.DataFrame(summ); summ.to_csv('results/A34_cv_summary.csv', index=False)
bytask = Sm.groupby(['target', 'task'])[MET].mean().reset_index(); bytask.to_csv('results/A34_cv_by_task.csv', index=False)
# random null: average over draws per method, real-minus-random gap
Rn = Sm[Sm.kind == 'random'].groupby(['method', 'sample', 'task', 'group'])[MET].mean().reset_index()
ndraw = Sm[Sm.kind == 'random'].groupby('method').target.nunique()
tests = []
def paired(a, b, label, metric, unit):
    a, b = a.align(b, join='inner'); d = a - b
    p = wilcoxon(a, b).pvalue if len(d) > 5 and (d != 0).any() else np.nan
    return dict(comparison=label, metric=metric, unit=unit, n=len(d), mean_a=a.mean(), mean_b=b.mean(), mean_diff=d.mean(), median_diff=d.median(), frac_a_gt_b=(d > 0).mean(), wilcoxon_p=p)
for m in Rn.method.unique():
    real = Sm[Sm.target == m].set_index('sample'); rnd = Rn[Rn.method == m].set_index('sample')
    for met in MET:
        tests.append(paired(real[met], rnd[met], f'{m}: real - random({ndraw[m]} draws)', met, 'slide'))
        tests.append(paired(real.groupby('group')[met].mean(), rnd.groupby('group')[met].mean(), f'{m}: real - random({ndraw[m]} draws)', met, 'group'))
        rr, nn = real[real.task == 'READ'][met], rnd[rnd.task == 'READ'][met]
        tests.append(dict(comparison=f'{m}: real - random({ndraw[m]} draws)', metric=met, unit='READ slides', n=len(rr), mean_a=rr.mean(), mean_b=nn.mean(), mean_diff=(rr - nn).mean(), median_diff=(rr - nn).median(), frac_a_gt_b=((rr - nn) > 0).mean(), wilcoxon_p=np.nan))
for t in ['mlm', 'ulm', 'ssgsea_depthres']:
    if t not in Sm.target.values: continue
    a = Sm[Sm.target == t].set_index('sample'); b0 = Sm[Sm.target == 'ssgsea'].set_index('sample')
    for met in MET:
        tests.append(paired(a[met], b0[met], f'{t} - ssgsea', met, 'slide'))
        tests.append(paired(a.groupby('group')[met].mean(), b0.groupby('group')[met].mean(), f'{t} - ssgsea', met, 'group'))
for a_, b_ in [('ulm', 'mlm'), ('ssgsea_depthres', 'mlm'), ('ulm', 'ssgsea_depthres')]:
    if a_ not in Sm.target.values or b_ not in Sm.target.values: continue
    A_ = Sm[Sm.target == a_].set_index('sample'); B_ = Sm[Sm.target == b_].set_index('sample')
    for met in MET:
        tests.append(paired(A_[met], B_[met], f'{a_} - {b_}', met, 'slide'))
        tests.append(paired(A_.groupby('group')[met].mean(), B_.groupby('group')[met].mean(), f'{a_} - {b_}', met, 'group'))
T = pd.DataFrame(tests); T.to_csv('results/A34_paired_tests.csv', index=False)
# READ per-program table
rp = R[R.task == 'READ'].groupby(['target', 'program'])[MET].mean().reset_index(); rp.to_csv('results/A4_READ_per_program.csv', index=False)
pd.set_option('display.width', 250)
print(summ.round(3).to_string()); print(T.round(4).to_string())
