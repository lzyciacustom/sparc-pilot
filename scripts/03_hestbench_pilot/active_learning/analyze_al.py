import os, sys, glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
sys.path.append(os.getcwd()); import sparcmap_lib as L
method = sys.argv[1] if len(sys.argv) > 1 else 'ssgsea'
D = f'al_out/{method}'; OUT = f'al_results/{method}'; os.makedirs(OUT, exist_ok=True)
f = L.wt_folds(); grp = dict(zip(f['sample'], f['group'])); task = dict(zip(f['sample'], f['task']))
runs = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(f'{D}/fold*_seed*_*.csv'))], ignore_index=True)
sel = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(f'{D}/sel_fold*.csv'))], ignore_index=True)
cnt = runs[runs.strategy.isin(['random', 'maxunc', 'epig'])].groupby(['fold', 'seed']).strategy.nunique(); full = cnt[cnt == 3].index
runs = runs.set_index(['fold', 'seed']).loc[full].reset_index()
sel = sel.set_index(['fold', 'seed']).loc[full].reset_index()
S3 = [x for x in ['random', 'maxunc', 'epig', 'featmatch'] if x in set(runs.strategy)]; MET = ['pearson', 'pearson_depthres']
print('complete (fold,seed) units:', len(full))

slide_cols = [c for c in runs.columns if c.startswith('slide_')]
long = runs.melt(id_vars=['fold', 'seed', 'strategy', 'round', 'n_slides', 'n_spots'], value_vars=slide_cols).dropna()
long['metric'] = np.where(long.variable.str.startswith('slide_pearson__'), 'pearson', 'pearson_depthres')
long['sample'] = long.variable.str.split('__').str[1]; long['group'] = long['sample'].map(grp); long['task'] = long['sample'].map(task)
long = long.drop(columns='variable')
long.to_csv(f'{OUT}/al_per_slide_long.csv', index=False)

lc = runs.groupby(['strategy', 'round'])[MET + ['n_spots', 'n_groups']].agg(['mean', 'sem'])
lc.columns = ['_'.join(c) for c in lc.columns]; lc = lc.reset_index(); lc.to_csv(f'{OUT}/al_learning_curves.csv', index=False)
lc_read = (long[long.task == 'READ'].groupby(['strategy', 'round', 'metric', 'fold', 'seed']).value.mean()
           .groupby(['strategy', 'round', 'metric']).agg(['mean', 'sem']).reset_index())
lc_read.to_csv(f'{OUT}/al_learning_curves_READ.csv', index=False)
lc_task = long.groupby(['task', 'strategy', 'round', 'metric']).value.mean().reset_index()
lc_task.to_csv(f'{OUT}/al_learning_curves_by_task.csv', index=False)


def auc_rounds(y):
    return np.trapezoid(y, dx=1) / (len(y) - 1)


rows = []
for (k, s, st), g in runs.groupby(['fold', 'seed', 'strategy']):
    g = g.sort_values('round')
    rows.append(dict(fold=k, seed=s, strategy=st, **{f'auc_{m}': auc_rounds(g[m].values) for m in MET},
                     **{f'final_{m}': g[m].values[-1] for m in MET}, final_n_spots=g.n_spots.values[-1],
                     final_n_groups=g.n_groups.values[-1]))
auc = pd.DataFrame(rows)

sp = []
for (k, s), g in runs.groupby(['fold', 'seed']):
    hi = g.groupby('strategy').n_spots.max().min(); lo = g.n_spots.min(); grid = np.linspace(lo, hi, 50)
    for st, h in g.groupby('strategy'):
        h = h.sort_values('n_spots')
        sp.append(dict(fold=k, seed=s, strategy=st, spot_budget=hi,
                       **{f'aucspot_{m}': np.trapezoid(np.interp(grid, h.n_spots, h[m]), grid) / (hi - lo) for m in MET},
                       **{f'atbudget_{m}': np.interp(hi, h.n_spots, h[m]) for m in MET}))
auc = auc.merge(pd.DataFrame(sp), on=['fold', 'seed', 'strategy'])
auc.to_csv(f'{OUT}/al_auc_per_run.csv', index=False)

ps = (long.groupby(['metric', 'sample', 'group', 'task', 'fold', 'seed', 'strategy'])
      .apply(lambda g: auc_rounds(g.sort_values('round').value.values), include_groups=False).rename('auc').reset_index())
ps_m = ps.groupby(['metric', 'sample', 'group', 'task', 'strategy']).auc.mean().unstack('strategy').reset_index()
ps_m.to_csv(f'{OUT}/al_auc_per_slide.csv', index=False)


def wx(d):
    d = np.asarray(d); d = d[~np.isnan(d)]
    return wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else np.nan


cmp = []
PAIRS = [('epig', 'random'), ('epig', 'maxunc'), ('maxunc', 'random')] + ([('epig', 'featmatch'), ('featmatch', 'random')] if 'featmatch' in S3 else [])
for a, b in PAIRS:
    W = auc.pivot_table(index=['fold', 'seed'], columns='strategy')
    for col in ['auc_pearson', 'auc_pearson_depthres', 'final_pearson', 'final_pearson_depthres',
                'aucspot_pearson', 'aucspot_pearson_depthres']:
        d = (W[(col, a)] - W[(col, b)]).values
        cmp.append(dict(comparison=f'{a}-{b}', metric=col, level='fold x seed run', n=int(np.isfinite(d).sum()), mean_diff=np.nanmean(d),
                        median_diff=np.nanmedian(d), n_pos=int((d > 0).sum()), wilcoxon_p=wx(d)))
    for m in MET:
        x = ps_m[ps_m.metric == m]
        d = (x[a] - x[b]).values
        cmp.append(dict(comparison=f'{a}-{b}', metric=f'auc_{m}', level='test slide (seed-avg)', n=int(np.isfinite(d).sum()),
                        mean_diff=np.nanmean(d), median_diff=np.nanmedian(d), n_pos=int((d > 0).sum()), wilcoxon_p=wx(d)))
        gg = x.groupby('group')[[a, b]].mean(); d = (gg[a] - gg[b]).values
        cmp.append(dict(comparison=f'{a}-{b}', metric=f'auc_{m}', level='patient group (16)', n=int(np.isfinite(d).sum()),
                        mean_diff=np.nanmean(d), median_diff=np.nanmedian(d), n_pos=int((d > 0).sum()), wilcoxon_p=wx(d)))
        xr = x[x.task == 'READ']; d = (xr[a] - xr[b]).values
        cmp.append(dict(comparison=f'{a}-{b}', metric=f'auc_{m}', level='READ slides (seed-avg)', n=int(np.isfinite(d).sum()),
                        mean_diff=np.nanmean(d), median_diff=np.nanmedian(d), n_pos=int((d > 0).sum()), wilcoxon_p=np.nan))
cmp = pd.DataFrame(cmp); cmp.to_csv(f'{OUT}/al_paired_comparisons.csv', index=False)

summ = auc.groupby('strategy')[[c for c in auc.columns if c.startswith(('auc', 'final', 'atbudget'))]].agg(['mean', 'sem'])
summ.columns = ['_'.join(c) for c in summ.columns]; summ = summ.reset_index()
rd = ps_m.groupby(['metric'])[S3].mean()
summ.to_csv(f'{OUT}/al_auc_table.csv', index=False)

sel['task'] = sel.slide.map(task)
comp = sel.groupby('strategy').agg(n_selected=('slide', 'size'), frac_group_already_covered=('group_already_covered', 'mean'),
                                   frac_batch_same_group=('same_group_as_batchmate', lambda x: x.sum() / (len(x) / 2)),
                                   mean_spots_per_slide=('n_spots', 'mean'), mean_epig_rank=('epig_rank', 'mean'),
                                   mean_maxunc_rank=('maxunc_rank', 'mean')).reset_index()
fg = auc.groupby('strategy').final_n_groups.mean().rename('mean_distinct_groups_final').reset_index()
comp = comp.merge(fg, on='strategy')
tc = sel.groupby(['strategy', 'task']).size().unstack(fill_value=0); tc = tc.div(tc.sum(1), axis=0).add_prefix('frac_task_')
comp = comp.merge(tc.reset_index(), on='strategy')
comp.to_csv(f'{OUT}/al_selection_composition.csv', index=False)

diag = []
for (k, s), g in runs[runs.strategy == 'random'].groupby(['fold', 'seed']):
    g = g.sort_values('round')
    for rdn in range(1, g['round'].max() + 1):
        sc = pd.read_csv(f'{D}/scores_fold{k}_seed{s}_random_r{rdn - 1}.csv')
        ch = sel[(sel.fold == k) & (sel.seed == s) & (sel.strategy == 'random') & (sel['round'] == rdn)].slide
        z = sc.set_index('slide')
        zr = (z[['epig', 'maxunc']].rank(pct=True)).loc[ch].mean()
        diag.append(dict(fold=k, seed=s, round=rdn, epig_pctile=zr.epig, maxunc_pctile=zr.maxunc,
                         gain_pearson=g.pearson.values[rdn] - g.pearson.values[rdn - 1],
                         gain_pdres=g.pearson_depthres.values[rdn] - g.pearson_depthres.values[rdn - 1]))
diag = pd.DataFrame(diag); diag.to_csv(f'{OUT}/al_score_vs_gain_random.csv', index=False)
dsum = []
for sc in ['epig_pctile', 'maxunc_pctile']:
    for gcol in ['gain_pearson', 'gain_pdres']:
        r, p = spearmanr(diag[sc], diag[gcol]); dsum.append(dict(score=sc, gain=gcol, n=len(diag), spearman=r, p=p))
allsc = pd.concat([pd.read_csv(x) for x in glob.glob(f'{D}/scores_*.csv')])
rr = allsc.groupby(['fold', 'seed', 'strategy', 'round']).apply(lambda g: spearmanr(g.epig, g.maxunc)[0], include_groups=False)
dsum.append(dict(score='epig vs maxunc (candidate ranks)', gain='-', n=len(rr), spearman=rr.median(), p=np.nan))
pd.DataFrame(dsum).to_csv(f'{OUT}/al_score_diagnostics.csv', index=False)
print(summ.round(3).to_string()); print(cmp.round(4).to_string()); print(comp.round(3).to_string()); print(pd.DataFrame(dsum).round(3))

tgt_tasks = {k: set(f[f.cv_fold == k].task) for k in f.cv_fold.unique()}
sel['task_in_target'] = [t in tgt_tasks[k] for t, k in zip(sel.task, sel.fold)]
pool_frac = []
for k in sorted(tgt_tasks):
    tr = f[f.cv_fold != k]
    pool_frac.append(tr.task.isin(tgt_tasks[k]).mean())
tm = sel.groupby('strategy').task_in_target.mean().rename('frac_selected_task_in_target').reset_index()
tm['pool_frac_task_in_target_mean'] = np.mean(pool_frac)
comp = pd.read_csv(f'{OUT}/al_selection_composition.csv').merge(tm, on='strategy')
comp.to_csv(f'{OUT}/al_selection_composition.csv', index=False)
pf = auc.groupby(['fold', 'strategy'])[['auc_pearson', 'auc_pearson_depthres', 'final_pearson', 'final_pearson_depthres']].mean().reset_index()
pf['target_tasks'] = pf.fold.map(lambda k: '+'.join(sorted(tgt_tasks[k])))
pf.to_csv(f'{OUT}/al_auc_by_fold.csv', index=False)
bt = ps.groupby(['metric', 'task', 'strategy']).auc.mean().unstack('strategy').reset_index()
bt.to_csv(f'{OUT}/al_auc_by_task.csv', index=False)
print(comp[['strategy', 'frac_selected_task_in_target', 'pool_frac_task_in_target_mean']].round(3).to_string())
print(pf.round(3).to_string()); print(bt.round(3).to_string())
