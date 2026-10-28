import os, sys, time, numpy as np, pandas as pd, torch; sys.path.append(os.getcwd())
import sparcmap_lib as L
import encoders as ENC
L.TGT_DIR='targets_crc'; FD=ENC.feat_dir('crc')
cand=pd.read_csv('hest1k_crc_candidates.csv')
cand=cand[cand.id.map(lambda s: os.path.exists(f'{FD}/{s}.h5') and os.path.exists(f'targets_crc/{s}.parquet'))]
g=cand.groupby('group').spots_under_tissue.sum().sort_values(ascending=False)
K=6; load=np.zeros(K); fold={}
for grp in g.index: k=int(np.argmin(load)); fold[grp]=k; load[k]+=g[grp]
cand['cv_fold']=cand.group.map(fold); cand.to_csv('crc_folds.csv',index=False)
job=sys.argv[1]; seed=int(sys.argv[2]) if len(sys.argv)>2 else 0
def load(sid):
    m='ssgsea' if job=='ssgsea_depthres' else job
    s=L.load_slide(sid, m, feat_dir=FD)
    if job=='ssgsea_depthres': s['Y']=L.residualize(s['Y'], s['depth'][:,None]).astype(np.float32)
    return s
S={s:load(s) for s in cand.id}
out=[]
for k in range(K):
    te=cand[cand.cv_fold==k].id.tolist(); tr_all=cand[cand.cv_fold!=k]
    grs=tr_all.groupby('group').spots_under_tissue.sum().sort_values(); vg=grs.index[len(grs)//2]
    va=tr_all[tr_all.group==vg].id.tolist(); tr=tr_all[tr_all.group!=vg].id.tolist(); t=time.time()
    m,norm,h=L.train([S[s] for s in tr],[S[s] for s in va],epochs=30,seed=100*seed+k)
    r=L.evaluate(m,[S[s] for s in te],norm); r['fold']=k; r['target']=job; r['seed']=seed; out.append(r)
    print(job,seed,k,vg,f'{time.time()-t:.0f}s',r[['pearson','pearson_centered','pearson_depthres']].mean().round(3).to_dict(),flush=True)
    del m; torch.cuda.empty_cache()
pd.concat(out).to_csv(f'crc_cv_{job}_s{seed}.csv',index=False)
