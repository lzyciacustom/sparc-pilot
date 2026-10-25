import sys, glob, os, json, warnings, numpy as np, pandas as pd, anndata as ad, h5py, scipy.sparse as sp, decoupler as dc, scanpy as sc
warnings.filterwarnings('ignore'); R='data/hest_bench'
found=json.load(open('sparc_programs_msigdb.json'))['sets']; PN=list(found)
sys.path.append(os.getcwd()); from ssgsea import ssgsea_gpu
log=[]
for p in sorted(glob.glob(f'{R}/*/adata/*.h5ad')):
    task=p.split('/')[2]; sid=os.path.basename(p)[:-5]; outp=f'targets/{sid}.parquet'
    if os.path.exists(outp): continue
    a=ad.read_h5ad(p); a.var_names_make_unique()
    with h5py.File(f'{R}/{task}/patches/{sid}.h5') as f:
        bc=[b[0].decode() if isinstance(b[0],bytes) else str(b[0]) for b in f['barcode'][:]]; xy=f['coords'][:]
    obs=pd.Index(a.obs_names); hit=obs.get_indexer(bc); ok=hit>=0
    a=a[hit[ok]].copy(); bc=np.array(bc)[ok]; xy=xy[ok]
    X=a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    keep=np.asarray((X>0).sum(0)).ravel()>0
    es,names=ssgsea_gpu(X[:,keep], a.var_names[keep], found)
    df=pd.DataFrame({'barcode':bc,'x':xy[:,0],'y':xy[:,1],'total_counts':np.asarray(X.sum(1)).ravel()})
    for j,n in enumerate(names): df['ssgsea__'+n]=es[:,j]
    b=a[:,keep].copy(); sc.pp.normalize_total(b,target_sum=1e4); sc.pp.log1p(b)
    net=pd.DataFrame([(k,g,1.0) for k,v in found.items() for g in set(v)],columns=['source','target','weight'])
    for m in ['ulm','mlm']:
        try:
            getattr(dc.mt,m)(data=b,net=net,tmin=5,verbose=False)
            est=b.obsm[f'score_{m}']
            for n in est.columns: df[f'{m}__{n}']=est[n].values
        except Exception as e: log.append((sid,m,repr(e)[:150]))
    df.to_parquet(outp); log.append((sid,'ok',len(df),len(names)))
    print(sid,task,len(df),len(names),flush=True)
json.dump(log,open('targets/log.json','w'))
