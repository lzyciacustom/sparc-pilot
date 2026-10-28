"""Composition / depth / residual decomposition of condition contrasts in the MD Anderson CRC metastasis Visium cohort.

b(pi, d) = a + c * b_M17(pi, d): 17-type colon+liver reference baseline; a, c fitted per program on NON-NEOPLASTIC spots only
(normal liver and colon), so calibration is independent of every tumour contrast tested.
Per spot: composition = b(pi, d_ref), depth = b(pi, d) - b(pi, d_ref), residual = y - b(pi, d); d_ref = median log depth.
All quantities in units of the pooled SD of y per program. Region means per section from the authors' pathology layers.
Contrasts (unit = section; patient IDs not available):
  micro_vs_macro_tumor / _stroma : liver micrometastasis minus macrometastasis, paired within sections that contain both
  primary_vs_liver_tumor         : primary carcinoma minus liver macrometastasis tumour (unpaired sections)
  front_vs_superficial_primary   : invasive-front tumour minus mucosa/submucosa tumour, paired within primary sections
Tests: Wilcoxon signed-rank (paired) or Mann-Whitney (unpaired) per program and component, BH over programs; the identical pipeline on
the 36 size-matched random gene sets (rand0 ULM, own b_M) gives the null hit rate.
Outputs: comp/results/mdacc_*.csv
"""
import os, sys, glob, numpy as np, pandas as pd, torch
from torch import nn
from scipy.stats import wilcoxon, mannwhitneyu
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, WS); sys.path.insert(0, f'{WS}/comp')
EXT = os.environ.get('SPARC_EXT', os.path.join(WS, 'ext'))  # external cohorts (scripts/00_data/download_external.py)
import sparcmap_lib as L
from build_ref17 import TYPES17
X_ = f'{EXT}/mdacc'; OUT = f'{WS}/comp/results'; K = len(TYPES17); P = len(L.PROGRAMS)
MIN = 10


def load_bM(target):
    b = torch.load(f'{WS}/comp/bM_{target}_ref17_mdacc.pt')
    net = nn.Sequential(nn.Linear(K + 1, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, P)); net.load_state_dict(b['state']); net.eval()
    lo, hi = (b['dmin'] - b['dmu']) / b['dsd'], (b['dmax'] - b['dmu']) / b['dsd']
    def f(pi, d):
        z = np.clip((np.broadcast_to(d, (len(pi),)) - b['dmu']) / b['dsd'], lo, hi)
        with torch.no_grad():
            return (net(torch.tensor(np.column_stack([np.nan_to_num(pi), z]), dtype=torch.float32)) * b['sd'] + b['mu']).numpy()
    return f


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty_like(p); n = len(p)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; return np.minimum(q, 1)


REG = {'liver_macro_tumor': ['Liver macrometastasis tumor'], 'liver_macro_stroma': ['Liver macrometastasis stroma'],
       'liver_micro_tumor': ['Liver micrometastasis tumor'], 'liver_micro_stroma': ['Liver micrometastasis stroma'],
       'primary_superficial_tumor': ['Tumor cells in mucosa and submucosa'],
       'primary_tumor': ['Tumor cells in mucosa and submucosa', 'Tumor cells in muscularis propria', 'Tumor cells in subserosa', 'Tumor cells at tumor invasive front/border'],
       'primary_front_tumor': ['Tumor cells at tumor invasive front/border'],
       'primary_front_stroma': ['Tumor stroma at tumor invasive front/border']}


def region_means(target, decon='comp'):
    bM = load_bM(target); D = pd.concat([pd.read_parquet(f).assign(sample=os.path.basename(f)[:-8]) for f in sorted(glob.glob(f'{X_}/proc/*.parquet'))])
    Y = D[[f'{target}__{p}' for p in L.PROGRAMS]].values.astype(float); d = np.log1p(D.total_counts.values.astype(float))
    pi = D[[f'{decon}__{t}' for t in TYPES17]].values.astype(float); ok = np.isfinite(pi).all(1) & np.isfinite(Y).all(1)
    D, Y, d, pi = D[ok].reset_index(drop=True), Y[ok], d[ok], pi[ok]
    B = bM(pi, d); dref = float(np.median(d)); Bref = bM(pi, dref)
    normal = D.Layer2.isin(['Non-neoplastic liver', 'Non-neoplastic colon']).values
    a = np.zeros(P); c = np.ones(P)
    for j in range(P):
        c[j], a[j] = np.polyfit(B[normal, j], Y[normal, j], 1)
    bfull, bref = a + c * B, a + c * Bref; sd = Y.std(0)
    parts = {'y': Y / sd, 'comp': bref / sd, 'depth': (bfull - bref) / sd, 'resid': (Y - bfull) / sd}
    calib_r = np.mean([np.corrcoef(bfull[normal, j], Y[normal, j])[0, 1] for j in range(P)])
    tumour = ~normal
    calib_r_t = np.mean([np.corrcoef(bfull[tumour, j], Y[tumour, j])[0, 1] for j in range(P)])
    rows = []
    for reg, labs in REG.items():
        m = D.Layer3.isin(labs).values
        for s in D['sample'].unique():
            mm = m & (D['sample'] == s).values
            if mm.sum() < MIN: continue
            r = dict(sample=s, region=reg, n=int(mm.sum()))
            for k, A in parts.items():
                for j, p in enumerate(L.PROGRAMS): r[f'{k}__{p}'] = A[mm, j].mean()
            for t_ in ['Hepatocyte', 'Epi_tumor', 'Fibroblast', 'T_NK', 'Mono_Mac', 'Kupffer', 'B', 'Plasma']:
                r[f'pi__{t_}'] = pi[mm, TYPES17.index(t_)].mean()
            rows.append(r)
    return pd.DataFrame(rows), calib_r, calib_r_t


def contrast(R, a, b, paired):
    A = R[R.region == a].set_index('sample'); B_ = R[R.region == b].set_index('sample')
    if paired:
        s = A.index.intersection(B_.index); A, B_ = A.loc[s], B_.loc[s]
    out = []
    for k in ['y', 'comp', 'depth', 'resid']:
        for p in L.PROGRAMS:
            x, z = A[f'{k}__{p}'].values, B_[f'{k}__{p}'].values
            if paired:
                dd = x - z; pv = wilcoxon(dd).pvalue if len(dd) >= 5 and np.any(dd != 0) else np.nan; eff = dd.mean(); pos = f'{(dd > 0).sum()}/{len(dd)}'
            else:
                pv = mannwhitneyu(x, z).pvalue if min(len(x), len(z)) >= 3 else np.nan; eff = x.mean() - z.mean(); pos = f'{len(x)} vs {len(z)}'
            out.append(dict(component=k, program=p, diff=eff, n=pos, p=pv))
    T = pd.DataFrame(out); T['q'] = T.groupby('component').p.transform(lambda s: bh(s.fillna(1)))
    pc = None
    if paired:
        pc = {t_: (A[f'pi__{t_}'] - B_[f'pi__{t_}']).mean() for t_ in ['Hepatocyte', 'Epi_tumor', 'Fibroblast', 'T_NK', 'Mono_Mac', 'Kupffer', 'B', 'Plasma']}
    else:
        pc = {t_: A[f'pi__{t_}'].mean() - B_[f'pi__{t_}'].mean() for t_ in ['Hepatocyte', 'Epi_tumor', 'Fibroblast', 'T_NK', 'Mono_Mac', 'Kupffer', 'B', 'Plasma']}
    return T, (len(A), len(B_)), pc


CONTRASTS = [('micro_vs_macro_tumor', 'liver_micro_tumor', 'liver_macro_tumor', True),
             ('micro_vs_macro_stroma', 'liver_micro_stroma', 'liver_macro_stroma', True),
             ('primary_vs_liver_tumor', 'primary_tumor', 'liver_macro_tumor', False),
             ('front_vs_superficial_primary', 'primary_front_tumor', 'primary_superficial_tumor', True)]

if __name__ == '__main__':
    allT, summ = [], []
    for target in ['ulm', 'rand0ulm']:
        for decon in (['comp', 'compdj'] if target == 'ulm' else ['comp']):
            R, cr, crt = region_means(target, decon)
            print(f'[{target}/{decon}] calibration r on normal spots {cr:.3f}, on tumour-containing spots {crt:.3f}', flush=True)
            for name, a, b, paired in CONTRASTS:
                T, n, pc = contrast(R, a, b, paired); T['contrast'] = name; T['target'] = target; T['decon'] = decon; allT.append(T)
                s = dict(target=target, decon=decon, contrast=name, n=str(n), comp_shift=' '.join(f'{k}:{v:+.2f}' for k, v in pc.items()))
                for k in ['y', 'comp', 'depth', 'resid']:
                    t = T[T.component == k]; s[f'{k}_q10'] = int((t.q < 0.1).sum()); s[f'{k}_absmean'] = round(t['diff'].abs().mean(), 3)
                summ.append(s)
    A = pd.concat(allT); A.to_csv(f'{OUT}/mdacc_contrasts_long.csv', index=False)
    S = pd.DataFrame(summ); S.to_csv(f'{OUT}/mdacc_contrasts_summary.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
    print(S.drop(columns='comp_shift').to_string())
    print(S[(S.target == 'ulm') & (S.decon == 'comp')][['contrast', 'comp_shift']].to_string())
    U = A[(A.target == 'ulm') & (A.decon == 'comp')]
    for name, *_ in CONTRASTS:
        t = U[(U.contrast == name)].pivot_table(index='program', columns='component', values=['diff', 'q'])
        top = t[('q', 'y')].sort_values().index[:10]
        print(f'\n== {name}: top programs by total difference (diff in SD units; q)')
        print(t.loc[top].round(3).to_string())
