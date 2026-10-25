import glob, os, sys, json, warnings, numpy as np, pandas as pd, anndata as ad, h5py, scipy.sparse as sp, decoupler as dc, scanpy as sc
warnings.filterwarnings('ignore'); R='data/hest1k_crc'
found=json.load(open('sparc_programs_msigdb.json'))['sets']; PN=list(found)
sys.path.append(os.getcwd()); from ens2sym import to_symbols; from ssgsea import ssgsea_gpu
log=[]
SH,NS=int(sys.argv[1]),int(sys.argv[2])
for ii,p in enumerate(sorted(glob.glob(f'{R}/st/*.h5ad'))):
    if ii%NS!=SH: continue
    sid=os.path.basename(p)[:-5]; outp=f'targets_crc/{sid}.parquet'
    if os.path.exists(outp): continue
    a=ad.read_h5ad(p); a=to_symbols(a); a.var_names_make_unique()
    with h5py.File(f'{R}/patches/{sid}.h5') as f:
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
            rr=getattr(dc.mt,m)(data=b,net=net,tmin=5,verbose=False); src=rr if rr is not None else b
            est=src.obsm[f'score_{m}'].reindex(b.obs_names)
            for n in est.columns: df[f'{m}__{n}']=est[n].values
        except Exception as e: log.append((sid,m,repr(e)[:150]))
    df.to_parquet(outp+'.tmp'); os.replace(outp+'.tmp',outp); log.append((sid,'ok',len(df),len(names)))
    print(sid,len(df),len(names),flush=True)
json.dump(log,open(f'targets_crc/log_shard{SH}.json','w'))
