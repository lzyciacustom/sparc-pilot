import sys, os, json, warnings, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch, decoupler as dc, scanpy as sc
warnings.filterwarnings('ignore'); R='data/hest1k_mouse'; OUT='targets_mouse'; os.makedirs(OUT,exist_ok=True)
P=json.load(open('sparc_programs_msigdb.json'))['sets']
h2m=pd.read_csv('mgi_one2one_human_mouse.csv'); h2m=dict(zip(h2m.human_symbol,h2m.mouse_symbol))
PM={p:sorted({h2m[g] for g in set(v) if g in h2m}) for p,v in P.items()}
json.dump(PM,open(f'{OUT}/programs_mouse_orthologs.json','w'))
sys.path.append(os.getcwd()); from ssgsea import ssgsea_gpu
def score(a, sets, df, prefix, log, sid):
    """a: AnnData (raw counts) restricted to one species' genes, var_names = symbols"""
    X=a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    keep=np.asarray((X>0).sum(0)).ravel()>0
    es,names=ssgsea_gpu(X[:,keep], a.var_names[keep], sets)
    for j,n in enumerate(names): df[f'{prefix}ssgsea__{n}']=es[:,j]
    b=a[:,keep].copy(); b.X=X[:,keep]; sc.pp.normalize_total(b,target_sum=1e4); sc.pp.log1p(b)
    net=pd.DataFrame([(k,g,1.0) for k,v in sets.items() for g in set(v)],columns=['source','target','weight'])
    for m in ['ulm','mlm']:
        try:
            rr=getattr(dc.mt,m)(data=b,net=net,tmin=5,verbose=False); src=rr if rr is not None else b
            est=src.obsm[f'score_{m}'].reindex(b.obs_names)
            for n in est.columns: df[f'{prefix}{m}__{n}']=est[n].values
        except Exception as e: log.append((sid,prefix+m,repr(e)[:150]))
    present={p:int(len(set(v)&set(a.var_names[keep]))) for p,v in sets.items()}
    return names, present
ids=pd.read_csv('mouse_test_slides.csv')['id'].tolist(); log=[]; cov=[]
for sid in ids:
    outp=f'{OUT}/{sid}.parquet'
    a=ad.read_h5ad(f'{R}/st/{sid}.h5ad'); a.var_names_make_unique()
    X=a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X); a.X=X
    gid=(a.var['gene_ids'] if 'gene_ids' in a.var else pd.Series(['ENSMUSG_assumed']*a.n_vars,index=a.var_names)).astype(str).str.replace(r'^GRCm38_','',regex=True)
    is_mm=gid.str.startswith('ENSMUSG').values; is_hs=gid.str.startswith('ENSG').values
    sym=pd.Index(a.var_names).str.replace(r'^(GRCm38_|mm10___|mm10_)','',regex=True)
    df=pd.DataFrame({'barcode':a.obs_names.astype(str)})
    df['total_counts']=np.asarray(X.sum(1)).ravel()
    if 'spatial' in a.obsm: xy=np.asarray(a.obsm['spatial']); df['x']=xy[:,0]; df['y']=xy[:,1]
    if 'in_tissue' in a.obs: df['in_tissue']=a.obs['in_tissue'].astype(int).values
    am=a[:,is_mm].copy(); am.var_names=sym[is_mm]; am.var_names_make_unique()
    df['total_counts_mouse']=np.asarray(am.X.sum(1)).ravel()
    names,present=score(am,PM,df,'',log,sid)
    for p in P: cov.append((sid,p,len(set(P[p])),len(PM[p]),present[p],p in names))
    if is_hs.sum()>1000:  # PDX / mixed-reference slide: also score human genes with the human programs
        ah=a[:,is_hs].copy(); ah.var_names=pd.Index(a.var_names)[is_hs]; ah.var_names_make_unique()
        df['total_counts_human']=np.asarray(ah.X.sum(1)).ravel()
        score(ah,P,df,'hs_',log,sid)
    df.to_parquet(outp,index=False); log.append((sid,'ok',len(df),len(names),int(is_mm.sum()),int(is_hs.sum())))
    print(sid,len(df),len(names),int(is_mm.sum()),int(is_hs.sum()),df.shape[1],flush=True)
    torch.cuda.empty_cache()
pd.DataFrame(cov,columns=['slide','program','human_set_size','mapped_one2one_orthologs','orthologs_detected_in_slide','scored_ssgsea']).to_csv(f'{OUT}/coverage_per_slide.csv',index=False)
json.dump(log,open(f'{OUT}/log.json','w'))
