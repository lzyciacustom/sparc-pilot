"""Evaluate composition-aware variants on the CRC test sections (16 patient groups; 2 overlap groups excluded).

Measured quantities per test spot (fold-specific calibration fitted on training sections only):
  b(pi, d)      = a + c * b_M(pi_deconv, d_spot)            composition + depth expectation
  delta         = y - b(pi, d)                               reference-relative residual
Predictions (H&E only): direct y_hat; composition-only b(pi_hat, d_ref); two-stage b(pi_hat, d_ref) + delta_hat (separate models);
joint b(pi_hat, d_ref) + delta_hat (one model). All accuracies are depth-adjusted Pearson (both sides residualised on log1p UMI).
Outputs: comp/results/comp_long.csv, comp/results/comp_summary.csv
"""
import os, sys, glob, numpy as np, pandas as pd, torch
from torch import nn
from scipy.stats import wilcoxon
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, WS)
import sparcmap_lib as L
from decon_resid import TYPES
L.TGT_DIR = f'{WS}/targets_crc'
LEAK = {'COLONMAP_8270_AS', 'TENX_immune_P1'}
crc = pd.read_csv(f'{WS}/crc_folds.csv').set_index('id'); grp = crc.group
PRD = f'{WS}/comp/preds'; OUT = f'{WS}/comp/results'; os.makedirs(OUT, exist_ok=True)
K = len(TYPES); P = len(L.PROGRAMS)


def load_bM(target):
    b = torch.load(f'{WS}/comp/bM_{target}.pt')
    net = nn.Sequential(nn.Linear(K + 1, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, P)); net.load_state_dict(b['state']); net.eval()
    lo, hi = (b['dmin'] - b['dmu']) / b['dsd'], (b['dmax'] - b['dmu']) / b['dsd']
    def f(pi, d):
        z = np.clip((np.broadcast_to(d, (len(pi),)) - b['dmu']) / b['dsd'], lo, hi)
        with torch.no_grad():
            return (net(torch.tensor(np.column_stack([np.nan_to_num(pi), z]), dtype=torch.float32)) * b['sd'] + b['mu']).numpy()
    return f
BM = {t: load_bM(t) for t in ['ulm', 'rand0ulm']}

_cache = {}
def truth(sid, target, decon):
    key = (sid, target, decon)
    if key not in _cache:
        t = pd.read_parquet(f'{WS}/targets_crc/{sid}.parquet').set_index('barcode')
        c = pd.read_parquet(f'{WS}/comp/{decon}/{sid}.parquet').set_index('barcode')
        _cache[key] = (t, c)
    return _cache[key]


def run_preds(tag):
    d = f'{PRD}/{tag}'
    if not os.path.isdir(d): return None
    out = {}
    for f in glob.glob(f'{d}/*.npz'):
        if os.path.basename(f).startswith('calib'): continue
        z = np.load(f, allow_pickle=True); out[os.path.basename(f)[:-4]] = {k: z[k] for k in z.files}
    cal = {int(os.path.basename(f)[7:-4]): dict(np.load(f)) for f in glob.glob(f'{d}/calib_f*.npz')}
    return out, cal


def dr(A, B, depth):
    return L._pearson_cols(L.residualize(A, depth[:, None]), L.residualize(B, depth[:, None]))


def evaluate(target, decon, seed):
    base = f'{target}__{decon}__s{seed}'
    R = {m: run_preds(f'{m}__{base}' + ('__a1b1' if m == 'joint' else '')) for m in ['direct', 'comp', 'delta', 'joint']}
    if R['direct'] is None and target == 'ulm' and decon != 'out_parquet':
        R['direct'] = run_preds(f'direct__ulm__out_parquet__s{seed}')
    ref = next((r for r in R.values() if r is not None), None)
    if ref is None: return []
    rows = []
    for sid in ref[0]:
        k = int(ref[0][sid]['fold']); bc = ref[0][sid]['barcode']
        t, c = truth(sid, target, decon); t = t.reindex(bc); c = c.reindex(bc)
        Y = t[[f'{target}__{p}' for p in L.PROGRAMS]].values.astype(float); depth = np.log1p(t.total_counts.values.astype(float))
        pi = c[[f'comp__{x}' for x in TYPES]].values.astype(float); okpi = np.isfinite(pi).all(1)
        calr = next(r[1][k] for r in [R['delta'], R['joint'], R['comp'], R['direct']] if r is not None and k in r[1])
        a, cc = calr['ab'][:, 0], calr['ab'][:, 1]; dref = float(calr['d_ref'])
        btrue = a + cc * BM[target](pi, depth); btrue[~okpi] = np.nan
        delta = Y - btrue
        rec = dict(sample=sid, group=grp[sid], fold=k, target=target, decon=decon, seed=seed)
        mean = lambda v: float(np.nanmean(v))
        rec['meas_b_vs_y'] = mean(dr(btrue, Y, depth))                                  # composition+depth expectation vs y (depth-adjusted)
        rec['meas_bref_vs_y'] = mean(dr(a + cc * BM[target](pi, dref), Y, depth))       # composition alone at fixed depth
        if R['direct'] is not None and sid in R['direct'][0]:
            rec['total_direct'] = mean(dr(R['direct'][0][sid]['y'], Y, depth))
        if R['comp'] is not None and sid in R['comp'][0]:
            ph = R['comp'][0][sid]['pi']; bhat = a + cc * BM[target](ph, dref)
            rec['total_componly'] = mean(dr(bhat, Y, depth))
            rec['pi_r_comp'] = mean([np.corrcoef(ph[okpi, j], pi[okpi, j])[0, 1] for j in range(K) if pi[okpi, j].std() > 0])
            for j, x in enumerate(TYPES):
                rec[f'pi_r_comp__{x}'] = np.corrcoef(ph[okpi, j], pi[okpi, j])[0, 1] if pi[okpi, j].std() > 0 else np.nan
            if R['direct'] is not None and sid in R['direct'][0]:
                rec['resid_direct_implicit'] = mean(dr(R['direct'][0][sid]['y'] - bhat, delta, depth))
            if R['delta'] is not None and sid in R['delta'][0]:
                dh = R['delta'][0][sid]['d']
                rec['total_twostage'] = mean(dr(bhat + dh, Y, depth))
        if R['delta'] is not None and sid in R['delta'][0]:
            rec['resid_twostage'] = mean(dr(R['delta'][0][sid]['d'], delta, depth))
        if R['joint'] is not None and sid in R['joint'][0]:
            j_ = R['joint'][0][sid]
            rec['total_joint'] = mean(dr(j_['y'], Y, depth)); rec['resid_joint'] = mean(dr(j_['d'], delta, depth))
            rec['pi_r_joint'] = mean([np.corrcoef(j_['pi'][okpi, q], pi[okpi, q])[0, 1] for q in range(K) if pi[okpi, q].std() > 0])
        rows.append(rec)
    return rows



if __name__ == '__main__':
    rows = []
    for target, decon in [('ulm', 'out_parquet'), ('rand0ulm', 'out_parquet'), ('ulm', 'out_disjoint')]:
        for seed in [0, 1, 2]:
            rows += evaluate(target, decon, seed)
    df = pd.DataFrame(rows); df.to_csv(f'{OUT}/comp_long.csv', index=False)
    df = df[~df.group.isin(LEAK)]
    met = [c for c in df.columns if c.startswith(('total_', 'resid_', 'pi_r_', 'meas_')) and '__' not in c]
    G = df.groupby(['target', 'decon', 'seed', 'group'])[met].mean().groupby(['target', 'decon', 'group']).mean()
    summ = G.groupby(['target', 'decon']).mean().T
    print(summ.round(3).to_string())
    nseed = df.groupby(['target', 'decon']).seed.nunique(); print('seeds', nseed.to_dict())


    def contrast(sel_a, col_a, sel_b, col_b, label):
        try:
            A = G.loc[sel_a][col_a]; B = G.loc[sel_b][col_b]
        except KeyError:
            return None
        d = (A - B).dropna()
        if len(d) < 5: return None
        return dict(contrast=label, a=A.mean(), b=B.mean(), diff=d.mean(), pos=f'{(d > 0).sum()}/{len(d)}', p=wilcoxon(d).pvalue)
    U = ('ulm', 'out_parquet'); Rn = ('rand0ulm', 'out_parquet'); Dj = ('ulm', 'out_disjoint')
    C = [contrast(U, 'total_joint', U, 'total_direct', 'total: joint - direct'),
         contrast(U, 'total_twostage', U, 'total_direct', 'total: two-stage - direct'),
         contrast(U, 'total_joint', U, 'total_twostage', 'total: joint - two-stage'),
         contrast(U, 'total_componly', U, 'total_direct', 'total: composition-only - direct'),
         contrast(U, 'resid_joint', U, 'resid_twostage', 'residual: joint - two-stage'),
         contrast(U, 'resid_twostage', U, 'resid_direct_implicit', 'residual: two-stage - direct(implicit)'),
         contrast(U, 'resid_joint', U, 'resid_direct_implicit', 'residual: joint - direct(implicit)'),
         contrast(U, 'pi_r_joint', U, 'pi_r_comp', 'composition r: joint - comp-only'),
         contrast(U, 'resid_twostage', Rn, 'resid_twostage', 'residual two-stage: real - random'),
         contrast(U, 'resid_joint', Rn, 'resid_joint', 'residual joint: real - random'),
         contrast(Dj, 'resid_twostage', Rn, 'resid_twostage', 'residual two-stage (gene-disjoint decon): real - random'),
         contrast(Dj, 'resid_twostage', U, 'resid_twostage', 'residual two-stage: gene-disjoint - original decon')]
    C = pd.DataFrame([c for c in C if c]); print(C.round(4).to_string())
    C.to_csv(f'{OUT}/comp_contrasts.csv', index=False); summ.to_csv(f'{OUT}/comp_summary.csv')
