import json, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, decoupler as dc, scanpy as sc, warnings
warnings.filterwarnings('ignore'); R='data/hest_bench'
found=json.load(open('sparc_programs_msigdb.json'))['sets']
inv=pd.read_csv('hest_bench_inventory.csv').set_index('sample')
bad=sorted({x[0] for x in json.load(open('targets/log.json')) if x[1] in('ulm','mlm')})
net=pd.DataFrame([(k,g,1.0) for k,v in found.items() for g in set(v)],columns=['source','target','weight'])
for sid in bad:
    df=pd.read_parquet(f'targets/{sid}.parquet')
    a=ad.read_h5ad(f"{R}/{inv.loc[sid,'task']}/adata/{sid}.h5ad"); a.var_names_make_unique()
    a=a[pd.Index(a.obs_names).get_indexer(df.barcode)].copy()
    X=a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X); keep=np.asarray((X>0).sum(0)).ravel()>0
    b=a[:,keep].copy(); sc.pp.normalize_total(b,target_sum=1e4); sc.pp.log1p(b)
    for m in ['ulm','mlm']:
        r=getattr(dc.mt,m)(data=b,net=net,tmin=5,verbose=False); src=r if r is not None else b
        est=src.obsm[f'score_{m}']
        est=est.reindex(b.obs_names)
        for n in est.columns: df[f'{m}__{n}']=est[n].values
    df.to_parquet(f'targets/{sid}.parquet'); print(sid, df.shape, flush=True)
