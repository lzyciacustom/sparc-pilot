"""Stage-linked contrast on COLON MAP carcinomas (Heiser et al. 2023; pathologist spot annotations and clinical stage from OSF hftq2).

Per section (HEST Visium, matched to OSF ST_master by barcode overlap), measured program scores are decomposed exactly as in
niche_decomp.py: y = composition b(pi, d_ref) + depth [b(pi, d) - b(pi, d_ref)] + residual delta (fold-specific calibration).
Regions: pathologist labels; 'border' = carcinoma_border + carcinoma_edge, 'core' = carcinoma.
Per section and program: border-vs-core contrast (mean border - mean core) in units of the section SD of y; averaged per patient.
Test: stage III/IV vs I/II (adenomas and normal excluded), difference of patient means, permutation p with stage labels shuffled
within MSI strata (20,000 permutations); BH over programs per component. Null: the 36 size-matched random gene sets (rand0 ULM,
own b_M and calibration) through the identical pipeline.
Also reported: H&E-predicted decomposition (two-stage models, seed 0) for the same contrast on sections in the CRC test set.
Outputs: comp/results/colonmap_stage_*.csv
"""
import os, sys, numpy as np, pandas as pd
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, WS); sys.path.insert(0, f'{WS}/comp')
import sparcmap_lib as L
from decon_resid import TYPES
from analyze_comp import BM, run_preds
OUT = f'{WS}/comp/results'; rng = np.random.default_rng(2026); NPERM = 20000

obs = pd.read_parquet(f'{WS}/data/colonmap_osf/colonmap_obs.parquet')
meta = pd.read_csv(f'{WS}/hf_probe/MahmoodLab_hest/HEST_v1_2_0.csv'); meta = meta[meta.dataset_title.str.contains('COLON MAP', na=False)]
crc = pd.read_csv(f'{WS}/crc_folds.csv').set_index('id')

# match HEST section -> OSF master file by barcode overlap
match = []
for _, r in meta.iterrows():
    sid, sub = r['id'], r['subseries']; num = sub.replace('_AS_', '_').replace('_KL_', '_')
    t = pd.read_parquet(f'{WS}/targets_crc/{sid}.parquet', columns=['barcode']).barcode
    cand = obs[obs.Sample.astype(str) == num]
    best = max(cand.master_file.unique(), key=lambda f: cand[cand.master_file == f].barcode.isin(t).sum(), default=None)
    if best is None: continue
    ov = cand[cand.master_file == best].barcode.isin(t).mean()
    match.append(dict(id=sid, subseries=sub, master_file=best, overlap=ov))
match = pd.DataFrame(match); print(match.overlap.describe().round(3).to_dict())
match = match[match.overlap > 0.8]

CAL = {t: run_preds(f'delta__{t}__out_parquet__s0')[1] for t in ['ulm', 'rand0ulm']}
PRED = {'comp': run_preds('comp__ulm__out_parquet__s0'), 'ulm': run_preds('delta__ulm__out_parquet__s0'), 'rand0ulm': run_preds('delta__rand0ulm__out_parquet__s0')}


def section_contrast(sid, mf, target):
    k = int(crc.loc[sid, 'cv_fold']) if sid in crc.index else None
    if k is None: return None
    cal = CAL[target][k]; a, c, dref = cal['ab'][:, 0], cal['ab'][:, 1], float(cal['d_ref'])
    o = obs[obs.master_file == mf].set_index('barcode')
    t = pd.read_parquet(f'{WS}/targets_crc/{sid}.parquet').set_index('barcode')
    cp = pd.read_parquet(f'{WS}/comp/out_parquet/{sid}.parquet').set_index('barcode')
    bc = t.index.intersection(o.index); t = t.loc[bc]; cp = cp.reindex(bc); lab = o.loc[bc, 'pathology_annotation'].astype(str).values
    Y = t[[f'{target}__{p}' for p in L.PROGRAMS]].values.astype(float); d = np.log1p(t.total_counts.values.astype(float))
    pi = cp[[f'comp__{x}' for x in TYPES]].values.astype(float); ok = np.isfinite(pi).all(1) & np.isfinite(Y).all(1)
    Y, d, pi, lab, bc = Y[ok], d[ok], pi[ok], lab[ok], bc[ok]
    region = np.where(np.isin(lab, ['carcinoma_border', 'carcinoma_edge']), 'border', np.where(lab == 'carcinoma', 'core', 'other'))
    if (region == 'border').sum() < 20 or (region == 'core').sum() < 20: return None
    bref = a + c * BM[target](pi, dref); bfull = a + c * BM[target](pi, d)
    parts = dict(y=Y, comp=bref, depth=bfull - bref, resid=Y - bfull)
    if sid in PRED['comp'][0] and sid in PRED[target][0]:
        pz, dz = PRED['comp'][0][sid], PRED[target][0][sid]
        ix = pd.Index(pz['barcode']).get_indexer(bc)
        if (ix >= 0).all():
            parts['comp_hat'] = a + c * BM[target](pz['pi'][ix], dref); parts['resid_hat'] = dz['d'][pd.Index(dz['barcode']).get_indexer(bc)]
    sd = Y.std(0) + 1e-9; B, C = region == 'border', region == 'core'
    return {k_: (A[B].mean(0) - A[C].mean(0)) / sd for k_, A in parts.items()}, int(B.sum()), int(C.sum())


def perm_test(x, stage, msi):
    """difference of means III/IV - I/II; stage labels permuted within MSI strata."""
    obs_d = x[stage].mean() - x[~stage].mean(); cnt = 0
    strata = [np.where(msi == m)[0] for m in np.unique(msi)]
    for _ in range(NPERM):
        s = stage.copy()
        for ix in strata: s[ix] = rng.permutation(stage[ix])
        cnt += abs(x[s].mean() - x[~s].mean()) >= abs(obs_d) - 1e-12
    return obs_d, (cnt + 1) / (NPERM + 1)


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty_like(p); n = len(p)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; return np.minimum(q, 1)


pat_info = obs.drop_duplicates('master_file').set_index('master_file')[['Patient', 'Tumor Stage', 'Tumor Type']]
results = {}
for target in ['ulm', 'rand0ulm']:
    rows = []
    for _, m in match.iterrows():
        r = section_contrast(m['id'], m['master_file'], target)
        if r is None: continue
        E, nb, nc = r; pi_ = pat_info.loc[m['master_file']]
        for key, v in E.items():
            for j, p in enumerate(L.PROGRAMS):
                rows.append(dict(id=m['id'], patient=pi_['Patient'], stage=pi_['Tumor Stage'], msi=pi_['Tumor Type'], comp_=key, program=p, value=v[j], n_border=nb, n_core=nc))
    D = pd.DataFrame(rows); D = D[D.stage.isin(['I', 'II', 'III/IV'])]
    P = D.groupby(['patient', 'stage', 'msi', 'comp_', 'program']).value.mean().reset_index()
    pats = P.drop_duplicates('patient').set_index('patient')
    print(target, 'patients', len(pats), pats.stage.value_counts().to_dict(), pats.groupby(['stage', 'msi']).size().to_dict(),
          'sections', D.id.nunique(), flush=True)
    T = []
    for (key, p), g in P.groupby(['comp_', 'program']):
        g = g.set_index('patient').loc[pats.index]
        stage = (g.stage == 'III/IV').values; diff, pv = perm_test(g.value.values, stage, g.msi.values)
        T.append(dict(target=target, component=key, program=p, diff_III_IV_minus_I_II=diff, p=pv, n_late=int(stage.sum()), n_early=int((~stage).sum())))
    T = pd.DataFrame(T); T['q'] = T.groupby('component').p.transform(bh)
    results[target] = (D, P, T); T.to_csv(f'{OUT}/colonmap_stage_tests_{target}.csv', index=False); P.to_csv(f'{OUT}/colonmap_stage_patient_{target}.csv', index=False)
    print(T.groupby('component').agg(n=('p', 'size'), p_lt_05=('p', lambda s: int((s < 0.05).sum())), q_lt_10=('q', lambda s: int((s < 0.1).sum())),
                                     median_p=('p', 'median')).to_string(), flush=True)
Tc = results['ulm'][2]
print(Tc[Tc.component.isin(['y', 'comp', 'resid', 'resid_hat']) & (Tc.p < 0.05)].sort_values(['component', 'p'])[['component', 'program', 'diff_III_IV_minus_I_II', 'p', 'q']].round(4).to_string())
# agreement of H&E-predicted vs measured contrast (patient level, across programs)
for tgt in ['ulm', 'rand0ulm']:
    P = results[tgt][1]
    W = P.pivot_table(index=['patient', 'program'], columns='comp_', values='value').dropna(subset=['resid_hat'])
    per = W.groupby('patient').apply(lambda g: pd.Series(dict(r_comp=np.corrcoef(g.comp, g.comp_hat)[0, 1], r_resid=np.corrcoef(g.resid, g.resid_hat)[0, 1])))
    print(tgt, 'border-vs-core: H&E-predicted vs measured, per patient r across gene sets: composition', per.r_comp.median().round(3),
          'residual', per.r_resid.median().round(3), 'n', len(per))
    per.to_csv(f'{OUT}/colonmap_pred_vs_measured_by_patient_{tgt}.csv')
