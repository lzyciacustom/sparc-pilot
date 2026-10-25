"""Aggregate B1 runs: per-slide metrics (seed-averaged), summaries by task / READ, seed-matched paired tests."""
import glob, os, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.append(os.getcwd())
import sparcmap_lib as L

MET = ['pearson', 'pearson_depthres', 'pearson_centered']
folds = L.wt_folds()


def load_runs(d='b1_runs'):
    fs = glob.glob(f'{d}/*.csv')
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else pd.DataFrame()


def per_slide(runs, seeds=None):
    r = runs if seeds is None else runs[runs.seed.isin(seeds)]
    s = r.groupby(['method', 'variant', 'seed', 'sample'])[MET].mean().reset_index()
    s = s.groupby(['method', 'variant', 'sample'])[MET].mean().reset_index()
    nseed = r.groupby(['method', 'variant']).seed.nunique().rename('n_seeds').reset_index()
    return s.merge(folds[['sample', 'task', 'group']], on='sample').merge(nseed, on=['method', 'variant'])


def summary(ps):
    out = []
    for (m, v), d in ps.groupby(['method', 'variant']):
        row = dict(method=m, variant=v, n_slides=len(d), n_seeds=int(d.n_seeds.iloc[0]))
        for k in MET: row[f'{k}_all'] = d[k].mean()
        g = d.groupby('group')[MET].mean()
        for k in MET[:2]: row[f'{k}_groupmean'] = g[k].mean()
        for t, dt in d.groupby('task'):
            for k in MET[:2]: row[f'{k}_{t}'] = dt[k].mean()
        out.append(row)
    return pd.DataFrame(out)


def paired(runs, a, b, method):
    """a vs b (b = reference), restricted to seeds both have; per-slide and per-group Wilcoxon."""
    ra, rb = runs[(runs.method == method) & (runs.variant == a)], runs[(runs.method == method) & (runs.variant == b)]
    seeds = sorted(set(ra.seed) & set(rb.seed))
    ka = set(zip(ra.fold, ra.seed)); kb = set(zip(rb.fold, rb.seed)); keys = ka & kb
    sel = lambda r: r[[(f, s) in keys for f, s in zip(r.fold, r.seed)]]
    pa, pb = per_slide(sel(ra)), per_slide(sel(rb))
    if len(pa) == 0: return []
    m = pa.merge(pb, on=['sample', 'task', 'group'], suffixes=('_a', '_b'))
    rows = []
    for k in MET[:2]:
        d = m[f'{k}_a'] - m[f'{k}_b']
        g = m.assign(d=d).groupby('group').d.mean()
        dr = m[m.task == 'READ']; drd = dr[f'{k}_a'] - dr[f'{k}_b']
        rows.append(dict(method=method, variant=a, reference=b, metric=k, seeds=','.join(map(str, seeds)), n_slides=len(m),
                         mean_a=m[f'{k}_a'].mean(), mean_b=m[f'{k}_b'].mean(), mean_diff=d.mean(), median_diff=d.median(),
                         slides_improved=int((d > 0).sum()), p_wilcoxon_slides=wilcoxon(d).pvalue if len(d) > 5 else np.nan,
                         n_groups=len(g), group_mean_diff=g.mean(), groups_improved=int((g > 0).sum()),
                         p_wilcoxon_groups=wilcoxon(g).pvalue if len(g) > 5 else np.nan,
                         READ_n=len(dr), READ_mean_a=dr[f'{k}_a'].mean(), READ_mean_b=dr[f'{k}_b'].mean(), READ_diff=drd.mean(),
                         READ_slides_improved=int((drd > 0).sum())))
    return rows


COMPARISONS = [('mlp', 'base_k16'), ('concert_k16', 'base_k16'), ('noDist_k16', 'base_k16'), ('noCos_k16', 'base_k16'),
               ('noComp_k16', 'base_k16'), ('noDist_k16', 'concert_k16'), ('noCos_k16', 'concert_k16'),
               ('noComp_k16', 'concert_k16'), ('base_k48', 'base_k16'), ('concert_k48', 'base_k16'),
               ('concert_k48', 'base_k48'), ('concert_k48', 'concert_k16'), ('concert_k16', 'mlp'), ('base_k16', 'mlp')]


def all_paired(runs):
    rows = []
    for m in runs.method.unique():
        for a, b in COMPARISONS:
            if {a, b} <= set(runs[runs.method == m].variant): rows += paired(runs, a, b, m)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    runs = load_runs()
    print(runs.groupby(['method', 'variant']).apply(lambda d: f"{d.fold.nunique()}f x {d.seed.nunique()}s, {len(d.groupby(['fold','seed']))} runs").to_string())
    ps = per_slide(runs); sm = summary(ps)
    pd.set_option('display.width', 250)
    print(sm[['method', 'variant', 'n_slides', 'n_seeds', 'pearson_all', 'pearson_depthres_all', 'pearson_READ', 'pearson_depthres_READ']].round(3).to_string())
    pt = all_paired(runs)
    if len(pt): print(pt[['method', 'variant', 'reference', 'metric', 'seeds', 'n_slides', 'mean_diff', 'slides_improved', 'p_wilcoxon_slides', 'group_mean_diff', 'groups_improved', 'p_wilcoxon_groups', 'READ_diff']].round(4).to_string())
