import os, sys, numpy as np, pandas as pd, torch; sys.path.append(os.getcwd())
import sparcmap_lib as L
L.TGT_DIR='targets_crc'; FD='features/hoptimus0_crc'
crc=pd.read_csv('crc_folds.csv'); pool=pd.read_csv('hest1k_train_pool.csv'); crc_ids=set(crc.id)
ids=[s for s in pool.id if s not in crc_ids]+crc[(crc.cv_fold!=0)].id.tolist()
S=[x for x in (L.load_slide(s,'rand0ssgsea',feat_dir=FD) for s in ids) if np.isfinite(x['Y']).any(axis=0).sum()>=30]
med,iqr=L.robust_stats([s['Y'] for s in S]); tb=L.SlideBatcher(S,med,iqr); rng=np.random.default_rng(0); torch.manual_seed(0)
m=L.build_model(36).cuda(); opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-2)
for step in range(3000):
    c,n,r,y,mk=tb.random_batch(1024,rng)
    with torch.autocast('cuda',dtype=torch.bfloat16): p=m(c,n,r).float()
    mse=((p-y).pow(2)*mk).sum()/mk.sum().clamp(min=1); pl=L.pearson_loss(p,y,mk); loss=mse+0.5*pl
    opt.zero_grad(); loss.backward()
    gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
    if not torch.isfinite(loss) or not torch.isfinite(gn):
        print('NONFINITE at step',step,'loss',loss.item(),'mse',mse.item(),'pl',pl.item(),'grad',gn.item(),'pred std min',p.std(0).min().item()); break
    opt.step()
    if step%500==0: print(step, round(loss.item(),4), 'pred std min', round(p.std(0).min().item(),5), flush=True)
else: print('no nonfinite in 3000 steps')
