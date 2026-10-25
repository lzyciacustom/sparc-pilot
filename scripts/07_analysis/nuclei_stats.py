"""Nuclear-feature-adjusted evaluation (45 FFPE CRC slides, 10% spot subsample, HoVer-Net PanNuke features).
Default: patient-group tests from results/raw/nuclei_adjusted_per_slide.csv.
--from-preds (run in the flat workspace after 06_nuclei/crc_pred.py and hovernet_spots.py): recompute that table from
per-spot predictions: residualise prediction and target on log UMI (depth) or log UMI + 7 nuclear features within slide."""
import sys, os, glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
NF = ['nuc_density', 'nuc_mean_area', 'frac_neoplastic', 'frac_inflammatory', 'frac_connective', 'frac_dead', 'frac_epithelial']
def resid(Y, Z):
    Z = np.c_[np.ones(len(Z)), Z]; out = np.full_like(Y, np.nan)
    for j in range(Y.shape[1]):
        ok = np.isfinite(Y[:, j])
        if ok.sum() < 20: continue
        b, *_ = np.linalg.lstsq(Z[ok], Y[ok, j], rcond=None); out[ok, j] = Y[ok, j] - Z[ok] @ b
    return out
def colr(A, B):
    r = []
    for j in range(A.shape[1]):
        ok = np.isfinite(A[:, j]) & np.isfinite(B[:, j])
        r.append(np.corrcoef(A[ok, j], B[ok, j])[0, 1] if ok.sum() > 20 and A[ok, j].std() > 0 and B[ok, j].std() > 0 else np.nan)
    return np.array(r)
def evaluate(target, seeds):
    rows = []
    for f in sorted(glob.glob('nuclei/*.parquet')):
        sid = os.path.basename(f)[:-8]; n = pd.read_parquet(f).set_index('barcode')
        P = [np.load(f'preds_crc/{target}__s{s}/{sid}.npz', allow_pickle=True) for s in seeds]
        pred = np.mean([z['pred'] for z in P], 0); z = P[0]; idx = pd.Index(z['barcode'].astype(str)).get_indexer(n.index); ok = idx >= 0; ii = idx[ok]; n2 = n[ok]
        Zn = n2[NF].fillna(n2[NF].median()).fillna(0).values; Zn = (Zn - Zn.mean(0)) / (Zn.std(0) + 1e-9); Zd = z['depth'][ii][:, None]
        for lab, Z in [('depth', Zd), ('depth+nuclei', np.c_[Zd, Zn])]:
            rows.append(dict(sample=sid, adjust=lab, r=np.nanmean(colr(resid(z['pred'][ii] if len(seeds) == 1 else pred[ii], Z), resid(z['Y'][ii], Z))), n=len(ii)))
    return pd.DataFrame(rows)
if '--from-preds' in sys.argv:
    E = pd.concat([evaluate('ulm', [0]).assign(model='ulm_s0'), evaluate('rand0ulm', [0]).assign(model='rand0ulm_s0'), evaluate('ulm', [0, 1, 2]).assign(model='ulm_3seed')])
    E.to_csv('nuclei_adjusted_per_slide.csv', index=False); sys.exit(0)
grp = pd.read_csv('metadata/crc_folds.csv').set_index('id')['group']
E = pd.read_csv('results/raw/nuclei_adjusted_per_slide.csv')
G = {m: E[E.model == m].pivot(index='sample', columns='adjust', values='r').pipe(lambda t: t.groupby(t.index.map(grp)).mean()) for m in E.model.unique()}
rows = []
for adj in ['depth', 'depth+nuclei']:
    d = (G['ulm_s0'][adj] - G['rand0ulm_s0'][adj]).dropna()
    rows.append(dict(adjust=adj, ulm_single=G['ulm_s0'][adj].mean(), ulm_3seed=G['ulm_3seed'][adj].mean(), random=G['rand0ulm_s0'][adj].mean(), diff=d.mean(), groups_pos=f'{(d > 0).sum()}/{len(d)}', p_wilcoxon=wilcoxon(d).pvalue))
T = pd.DataFrame(rows)
for m in ['ulm_s0', 'rand0ulm_s0']:
    d = G[m]['depth+nuclei'] - G[m]['depth']; T[f'retained_{m}'] = G[m]['depth+nuclei'].mean() / G[m]['depth'].mean()
d = G['ulm_3seed']['depth'] - G['ulm_s0']['depth']; T['ensemble_gain'] = d.mean(); T['ensemble_p'] = wilcoxon(d).pvalue
T.to_csv('results/tables/nuclei_adjusted_eval.csv', index=False); print(T.round(4).to_string(index=False))
print('slides', E['sample'].nunique(), 'groups', len(G['ulm_s0']), 'spots', int(E[(E.model == 'ulm_s0') & (E.adjust == 'depth')].n.sum()))
