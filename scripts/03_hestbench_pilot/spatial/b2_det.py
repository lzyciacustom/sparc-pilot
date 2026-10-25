"""B2 deterministic head metrics from saved B1 predictions of the chosen backbone. Usage: python b2_det.py <backbone>"""
import os, sys, glob; sys.path.append(os.getcwd())
import numpy as np, pandas as pd
import sparcmap_lib as L
from run_b2 import slide_metrics, hex_neighbours, true_clusters, OUT
from run_b1 import VARIANTS, split, folds
bbv = sys.argv[1]; kk = VARIANTS[bbv][0]
for m in ['ssgsea', 'mlm']:
    S = {sid: L.load_slide(sid, m, k=kk) for sid in folds['sample']}; geo = {}
    for f in sorted(glob.glob(f'b1_preds/{m}__{bbv}__f*__s*.npz')):
        k, s = int(f.split('__f')[1][0]), int(f.split('__s')[-1][0])
        tag = f'{m}__{bbv}__det__f{k}__s{s}'
        if os.path.exists(f'{OUT}/{tag}.csv'): continue
        z = np.load(f); med, iqr = z['__norm_med'], z['__norm_iqr']; srows, prows = [], []
        for sid in split(k)[2]:
            sl = S[sid]
            if sid not in geo: geo[sid] = (hex_neighbours(sl['xy']), true_clusters(sl))
            pt = z[sid]; P = dict(point=pt, point_n=(pt - med) / iqr, mu=None, sd=None, samples=None)
            met, r_raw, r_d, r_b = slide_metrics(sl, P, (med, iqr), *geo[sid])
            met.pop('_cov50_prog', None); met.pop('_cov90_prog', None); srows.append(dict(sample=sid, **met))
            for j, p in enumerate(L.PROGRAMS):
                prows.append(dict(sample=sid, program=p, pearson=r_raw[j], pearson_depthres=r_d[j], pearson_boundary=r_b[j], cov50=np.nan, cov90=np.nan))
        meta = dict(method=m, backbone=bbv, head='det', fold=k, seed=s, best_epoch=-1)
        pd.DataFrame(prows).assign(**meta).to_csv(f'{OUT}/{tag}.prog.csv', index=False)
        pd.DataFrame(srows).assign(**meta).to_csv(f'{OUT}/{tag}.csv', index=False); print(tag, flush=True)
