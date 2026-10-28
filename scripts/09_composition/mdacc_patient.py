"""Patient-level version of the MDACC contrasts (Table S1 sample->patient map; IX_566xx sections, absent from Table S1, excluded).
Region means are averaged over a patient's sections before testing; paired contrasts use patients with both regions.
Random calibration as in mdacc_null.py: per patient, curated difference as z vs the 36 random-set differences; Wilcoxon across patients."""
import os, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, f'{WS}/comp'); sys.path.insert(0, WS)
from mdacc_decomp import region_means, bh, CONTRASTS
import sparcmap_lib as L
OUT = f'{WS}/comp/results'; KEYS = ['y', 'comp', 'depth', 'resid']
pm = pd.read_csv(f'{WS}/mdacc_patient_map.csv').set_index('sample').patient

def per_patient(R):
    R = R[R['sample'].isin(pm.index)].copy(); R['patient'] = R['sample'].map(pm)
    cols = [c for c in R.columns if '__' in c]
    return R.groupby(['patient', 'region'])[cols].mean().reset_index()

S = {lab: per_patient(region_means(t, dc)[0]) for lab, t, dc in [('curated', 'ulm', 'comp'), ('curated_disjoint', 'ulm', 'compdj'), ('random', 'rand0ulm', 'comp')]}

def mat(R, reg, k, pts): return R[R.region == reg].set_index('patient').loc[pts, [f'{k}__{p}' for p in L.PROGRAMS]].values

rows = []
for name, a, b, paired in CONTRASTS:
    pa = set(S['curated'][S['curated'].region == a].patient); pb = set(S['curated'][S['curated'].region == b].patient)
    for lab in ['curated', 'curated_disjoint', 'random_loo']:
        R = S['random' if lab == 'random_loo' else lab]
        for k in KEYS:
            if paired:
                pts = sorted(pa & pb)
                if len(pts) < 5: continue
                Dc = mat(R, a, k, pts) - mat(R, b, k, pts); Dr = mat(S['random'], a, k, pts) - mat(S['random'], b, k, pts)
                for j, p in enumerate(L.PROGRAMS):
                    ref = np.delete(Dr, j, axis=1) if lab == 'random_loo' else Dr
                    z = (Dc[:, j] - ref.mean(1)) / (ref.std(1) + 1e-9)
                    rows.append(dict(contrast=name, set=lab, component=k, program=p, n=len(pts), raw_diff=Dc[:, j].mean(), z_mean=z.mean(),
                                     pos=int((Dc[:, j] > 0).sum()), p_raw=wilcoxon(Dc[:, j]).pvalue, p_z=wilcoxon(z).pvalue))
            else:
                A = mat(R, a, k, sorted(pa)); B = mat(R, b, k, sorted(pb))
                for j, p in enumerate(L.PROGRAMS):
                    rows.append(dict(contrast=name, set=lab, component=k, program=p, n=f'{len(pa)} vs {len(pb)}', raw_diff=A[:, j].mean() - B[:, j].mean(),
                                     z_mean=np.nan, pos=np.nan, p_raw=mannwhitneyu(A[:, j], B[:, j]).pvalue, p_z=np.nan))
T = pd.DataFrame(rows)
for c in ['p_raw', 'p_z']: T['q' + c[1:]] = T.groupby(['contrast', 'set', 'component'])[c].transform(lambda s: bh(s.fillna(1)))
T.to_csv(f'{OUT}/mdacc_patient_level.csv', index=False)
print('patients per contrast:', T.groupby('contrast').n.first().to_dict())
print(T.groupby(['contrast', 'set', 'component']).q_z.apply(lambda s: int((s < 0.1).sum())).unstack('component').to_string())
pd.set_option('display.width', 250)
key = ['GSE26495_PD1HIGH_VS_PD1LOW_CD8_TCELL_UP', 'REACTOME_COLLAGEN_FORMATION', 'HALLMARK_INTERFERON_GAMMA_RESPONSE', 'HALLMARK_TNFA_SIGNALING_VIA_NFKB', 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION']
X = T[(T.contrast == 'micro_vs_macro_stroma') & T.set.isin(['curated', 'curated_disjoint']) & T.program.isin(key)]
print(X[['set', 'component', 'program', 'n', 'raw_diff', 'z_mean', 'pos', 'q_raw', 'q_z']].round(3).to_string())
r = T[(T.contrast == 'micro_vs_macro_stroma') & (T.set == 'random_loo') & (T.component == 'resid')].z_mean.abs()
c = T[(T.contrast == 'micro_vs_macro_stroma') & (T.set == 'curated') & (T.component == 'resid')].set_index('program').z_mean
print('random |z| max', round(r.max(), 2), '| curated residual effects beyond random max:', c[c.abs() > r.max()].round(2).to_dict())
