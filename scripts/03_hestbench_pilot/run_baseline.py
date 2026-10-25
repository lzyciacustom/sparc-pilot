import os, sys; sys.path.append(os.getcwd())
import sparcmap_lib as L, pandas as pd, numpy as np, torch, time
folds=L.wt_folds(); out=[]
S={s:L.load_slide(s,'ssgsea') for s in folds['sample']}
for k in sorted(folds.cv_fold.unique()):
    te=folds[folds.cv_fold==k]['sample'].tolist(); tr_all=folds[folds.cv_fold!=k]
    # inner validation: one patient group from training folds for early stopping
    vg=tr_all.groupby('group').n_spots.sum().sort_values().index[len(tr_all.group.unique())//2]
    va=tr_all[tr_all.group==vg]['sample'].tolist(); tr=tr_all[tr_all.group!=vg]['sample'].tolist()
    t=time.time()
    m,norm,h=L.train([S[s] for s in tr],[S[s] for s in va],epochs=30,seed=k)
    r=L.evaluate(m,[S[s] for s in te],norm); r['fold']=k; out.append(r)
    torch.save({'state':m.state_dict(),'norm':norm},f'baseline_fold{k}.pt')
    print(k, vg, f'{time.time()-t:.0f}s', r[['pearson','pearson_centered','pearson_depthres']].mean().round(3).to_dict(), flush=True)
res=pd.concat(out).merge(folds[['sample','task']],on='sample'); res.to_csv('baseline_sparcmap_ssgsea_cv.csv',index=False)
print(res.groupby('task')[['pearson','pearson_centered','pearson_depthres']].mean().round(3))
print('ALL', res[['pearson','pearson_centered','pearson_depthres']].mean().round(3).to_dict())
