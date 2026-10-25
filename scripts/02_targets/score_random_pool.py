import os,sys; sys.path.append(os.getcwd()); from ens2sym import to_symbols
import glob, os, json, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch, decoupler as dc, scanpy as sc, warnings
warnings.filterwarnings('ignore'); R='data/hest1k_crc'; dev='cuda'
src=open('compute_targets_crc.py').read(); exec(src[src.index('def ssgsea_gpu'):src.index('log=[]')])
RS=json.load(open('random_gene_sets.json'))['sets']
import sys
SH,NS=int(sys.argv[1]),int(sys.argv[2]); CRC=set(pd.read_csv('hest1k_crc_candidates.csv').id)
for ii,p in enumerate(sorted(glob.glob('targets_crc/*.parquet'))):
    sid=os.path.basename(p)[:-8]
    if ii%NS!=SH or sid in CRC: continue
    df=pd.read_parquet(p)
    if 'rand2ssgsea__HALLMARK_HYPOXIA' in df: continue
    a=ad.read_h5ad(f'{R}/st/{sid}.h5ad'); a=to_symbols(a); a.var_names_make_unique()
    a=a[pd.Index(a.obs_names).get_indexer(df.barcode)].copy()
    X=a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X); keep=np.asarray((X>0).sum(0)).ravel()>0
    for k in ['rand0','rand1','rand2']:
        es,names=ssgsea_gpu(X[:,keep], a.var_names[keep], RS[k])
        for j,n in enumerate(names): df[f'{k}ssgsea__{n}']=es[:,j]
    b=a[:,keep].copy(); sc.pp.normalize_total(b,target_sum=1e4); sc.pp.log1p(b)
    net=pd.DataFrame([(kk,g,1.0) for kk,v in RS['rand0'].items() for g in set(v)],columns=['source','target','weight'])
    try:
        rr=dc.mt.ulm(data=b,net=net,tmin=5,verbose=False); srco=rr if rr is not None else b
        est=srco.obsm['score_ulm'].reindex(b.obs_names)
        for n in est.columns: df[f'rand0ulm__{n}']=est[n].values
    except AssertionError as e: print(sid,'ulm skipped',flush=True)
    df.to_parquet(p+'.tmp'); os.replace(p+'.tmp',p); print(sid, df.shape, flush=True)
