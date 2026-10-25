import os, sys, numpy as np, pandas as pd, torch; sys.path.append(os.getcwd())
import sparcmap_lib as L
L.TGT_DIR='targets_crc'; FD='features/hoptimus0_crc'
crc=pd.read_csv('crc_folds.csv'); pool=pd.read_csv('hest1k_train_pool.csv')
crc_ids=set(crc.id); ids=[s for s in pool.id if s not in crc_ids]+crc[(crc.cv_fold!=0)].id.tolist()
S=[]
for s in ids:
    x=L.load_slide(s,'rand0ssgsea',feat_dir=FD)
    if np.isfinite(x['Y']).any(axis=0).sum()>=30: S.append(x)
med,iqr=L.robust_stats([s['Y'] for s in S])
print('n',len(S),'iqr min',iqr.min(),'med range',med.min(),med.max())
Yn=[(s['Y']-med)/iqr for s in S]
mx=[(s['sid'],np.nanmax(np.abs(y))) for s,y in zip(S,Yn)]; mx.sort(key=lambda t:-t[1]); print('largest |normalised y|',mx[:6])
fx=[(s['sid'],np.abs(s['X']).max(), np.isfinite(s['X']).all(), np.isfinite(s['rel']).all(), np.abs(s['rel']).max()) for s in S]
print('nonfinite X/rel:',[f for f in fx if not (f[2] and f[3])][:5]); fx.sort(key=lambda t:-t[4]); print('largest rel',fx[:4])
