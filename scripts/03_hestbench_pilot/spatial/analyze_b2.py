"""Aggregate B2 head comparison. Adds a smoothed-compartment boundary metric computed post hoc from saved predictions."""
import glob, os, re, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
sys.path.append(os.getcwd())
import sparcmap_lib as L
from run_b2 import hex_neighbours, true_clusters
from run_b1 import folds, split

BB = open('b2_backbone.txt').read().strip()


def smooth_clusters(Y, nb, seed=0):
    Y = np.where(np.isnan(Y), np.nanmean(Y, 0), Y)
    Ys = np.array([Y[js].mean(0) for js in nb])
    Z = (Ys - Ys.mean(0)) / (Ys.std(0) + 1e-9)
    return KMeans(6, n_init=10, random_state=seed).fit_predict(Z)


def posthoc_boundary(cache='b2_smooth_boundary.csv'):
    done = pd.read_csv(cache) if os.path.exists(cache) else pd.DataFrame(columns=['method', 'head', 'fold', 'seed', 'sample'])
    have = set(zip(done['method'], done['head'], done['fold'], done['seed']))
    rows = []
    for m in ['ssgsea', 'mlm']:
        todo = []
        for f in glob.glob(f'b2_runs/{m}__{BB}__*__f*__s*.csv'):
            if f.endswith('.prog.csv'): continue
            h = f.split('__')[2]; k, s = map(int, re.search(r'__f(\d)__s(\d)\.csv$', f).groups())
            if (m, h, k, s) in have: continue
            pf = f'b1_preds/{m}__{BB}__f{k}__s{s}.npz' if h == 'det' else f.replace('.csv', '.pred.npz')
            todo.append((h, k, s, pf))
        if not todo: continue
        S = {}
        for h, k, s, pf in todo:
            z = np.load(pf)
            for sid in split(k)[2]:
                if sid not in S:
                    sl = L.load_slide(sid, m); nb = hex_neighbours(sl['xy'])
                    lab = smooth_clusters(sl['Y'], nb)
                    bnd = np.array([any(lab[j] != lab[i] for j in js) for i, js in enumerate(nb)])
                    lab0 = true_clusters(sl)
                    bnd0 = np.array([any(lab0[j] != lab0[i] for j in js) for i, js in enumerate(nb)])
                    S[sid] = (sl['Y'], sl['depth'], bnd, bnd0)
                Y, dep, bnd, bnd0 = S[sid]; P = z[sid]
                r0 = L._pearson_cols(P[bnd0], Y[bnd0]); r0i = L._pearson_cols(P[~bnd0], Y[~bnd0]) if (~bnd0).sum() > 20 else np.full(Y.shape[1], np.nan)
                rb = L._pearson_cols(P[bnd], Y[bnd]); ri = L._pearson_cols(P[~bnd], Y[~bnd])
                rbd = L._pearson_cols(L.residualize(P[bnd], dep[bnd, None]), L.residualize(Y[bnd], dep[bnd, None]))
                rows.append(dict(method=m, head=h, fold=k, seed=s, sample=sid, sm_boundary_frac=bnd.mean(),
                                 spec_boundary_frac=bnd0.mean(), pearson_spec_boundary=np.nanmean(r0), pearson_spec_interior=np.nanmean(r0i),
                                 pearson_sm_boundary=np.nanmean(rb), pearson_sm_interior=np.nanmean(ri),
                                 pearson_sm_boundary_depthres=np.nanmean(rbd)))
    out = pd.concat([done, pd.DataFrame(rows)]) if rows else done
    out.to_csv(cache, index=False); return out


def load():
    fs = [f for f in glob.glob(f'b2_runs/*__{BB}__*.csv') if not f.endswith('.prog.csv')]
    d = pd.concat([pd.read_csv(f) for f in fs])
    sb = posthoc_boundary()
    d = d.drop(columns=['pearson_boundary', 'pearson_interior', 'boundary_frac']).merge(sb, on=['method', 'head', 'fold', 'seed', 'sample'], how='left')
    d = d.rename(columns={'pearson_spec_boundary': 'pearson_boundary', 'pearson_spec_interior': 'pearson_interior', 'spec_boundary_frac': 'boundary_frac'})
    return d.merge(folds[['sample', 'task', 'group']], on='sample')


METRICS = ['pearson', 'pearson_depthres', 'pearson_boundary', 'pearson_interior', 'pearson_sm_boundary', 'pearson_sm_interior',
           'pearson_sm_boundary_depthres', 'corrF_point', 'corrF_sample', 'SF_point', 'SF_sample', 'S_r_point', 'S_r_sample',
           'meanoffcorr_point', 'meanoffcorr_sample', 'meanoffcorr_true', 'cov50_gauss', 'cov90_gauss', 'cov50_samp', 'cov90_samp',
           'crps_gauss', 'crps_samp', 'energy', 'spread_skill', 'boundary_frac', 'sm_boundary_frac']


def common_runs(d):
    """Keep (method, fold, seed) combinations for which all heads are present."""
    nh = d.groupby(['method', 'fold', 'seed'])['head'].nunique(); full = nh[nh == d['head'].nunique()].index
    return d.set_index(['method', 'fold', 'seed']).loc[lambda x: x.index.isin(full)].reset_index()


def per_slide(d):
    cols = [c for c in METRICS if c in d]
    return d.groupby(['method', 'head', 'sample', 'task', 'group'])[cols].mean().reset_index()


def summary(ps):
    rows = []
    for (m, h), x in ps.groupby(['method', 'head']):
        r = dict(method=m, head=h, n_slides=len(x))
        for c in METRICS:
            if c in x: r[c] = x[c].mean()
        xr = x[x.task == 'READ']
        for c in ['pearson', 'pearson_depthres', 'pearson_sm_boundary']: r[f'{c}_READ'] = xr[c].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def paired(ps, a, ref='det'):
    rows = []
    for m in ps.method.unique():
        x = ps[ps.method == m]; A, B = x[x['head'] == a].set_index('sample'), x[x['head'] == ref].set_index('sample')
        idx = A.index.intersection(B.index)
        if len(idx) < 6: continue
        for c in ['pearson', 'pearson_depthres', 'pearson_sm_boundary', 'pearson_boundary', 'corrF_point', 'SF_point', 'S_r_point',
                  'corrF_sample', 'SF_sample', 'S_r_sample', 'crps_samp', 'energy']:
            if c not in A or A[c].isna().all() or B[c].isna().all(): continue
            dd = (A.loc[idx, c] - B.loc[idx, c]).dropna(); g = dd.groupby(A.loc[dd.index, 'group']).mean()
            rows.append(dict(method=m, head=a, reference=ref, metric=c, n_slides=len(dd), mean_head=A.loc[dd.index, c].mean(),
                             mean_ref=B.loc[dd.index, c].mean(), mean_diff=dd.mean(), slides_higher=int((dd > 0).sum()),
                             p_wilcoxon_slides=wilcoxon(dd).pvalue, n_groups=len(g), groups_higher=int((g > 0).sum()),
                             p_wilcoxon_groups=wilcoxon(g).pvalue if len(g) > 5 else np.nan))
    return pd.DataFrame(rows)


def calib_by_group(ps):
    rows = []
    for (m, h, g), x in ps[ps['head'] != 'det'].groupby(['method', 'head', 'group']):
        r = dict(method=m, head=h, group=g, task=x.task.iloc[0], n_slides=len(x))
        for c in ['cov50_gauss', 'cov90_gauss', 'cov50_samp', 'cov90_samp', 'crps_samp', 'energy', 'spread_skill']:
            if c in x: r[c] = x[c].mean()
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    d = load(); print(d.groupby(['method', 'head']).apply(lambda x: sorted(set(zip(x.fold, x.seed)))).to_string())
    dc = common_runs(d); ps = per_slide(dc); sm = summary(ps)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(sm.round(3).T.to_string())
    pt = pd.concat([paired(ps, 'gauss'), paired(ps, 'flow'), paired(ps, 'flow', 'gauss')])
    print(pt.round(4).to_string())
