import os,sys; sys.path.append(os.getcwd()); from ens2sym import to_symbols
import glob, os, json, re, warnings, numpy as np, pandas as pd, anndata as ad, h5py, scipy.sparse as sp, torch, decoupler as dc, scanpy as sc
warnings.filterwarnings('ignore'); dev='cuda'; R='data/hest1k_crc'
found=json.load(open('sparc_programs_msigdb.json'))['sets']; PN=list(found)
def ssgsea_gpu(X, genes, sets, w=0.25, min_size=5, bs=1024):
    gidx={g:i for i,g in enumerate(genes)}; names=[];cols=[]
    for p,g in sets.items():
        ii=[gidx[x] for x in set(g) if x in gidx]
        if len(ii)>=min_size: names.append(p); cols.append(ii)
    N=len(genes); M=torch.zeros(N,len(names),device=dev)
    for j,ii in enumerate(cols): M[ii,j]=1
    nS=M.sum(0); out=[]
    for s in range(0,X.shape[0],bs):
        xb=X[s:s+bs]; xb=xb.toarray() if sp.issparse(xb) else np.asarray(xb)
        r=torch.tensor(pd.DataFrame(xb.T).rank(method='average').values.T,device=dev,dtype=torch.float32)
        wts=r**w
        out.append((((r*wts)@M)/(wts@M) - (N*(N+1)/2 - r@M)/(N-nS)).cpu().numpy())
    return np.vstack(out), names
log=[]
import sys
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
