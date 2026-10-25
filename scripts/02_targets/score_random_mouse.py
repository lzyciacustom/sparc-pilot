import json, warnings, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, decoupler as dc, scanpy as sc
warnings.filterwarnings('ignore')
h2m=pd.read_csv('mgi_one2one_human_mouse.csv'); h2m=dict(zip(h2m.human_symbol,h2m.mouse_symbol))
RS=json.load(open('random_gene_sets.json'))['sets']['rand0']
RM={p:sorted({h2m[g] for g in set(v) if g in h2m}) for p,v in RS.items()}
net=pd.DataFrame([(k,g,1.0) for k,v in RM.items() for g in v],columns=['source','target','weight'])
for sid in pd.read_csv('mouse_test_slides_meta.csv').id:
    a=ad.read_h5ad(f'data/hest1k_mouse/st/{sid}.h5ad'); a.var_names_make_unique()
    X=a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X); a.X=X
    gid=(a.var['gene_ids'] if 'gene_ids' in a.var else pd.Series(['ENSMUSG_assumed']*a.n_vars,index=a.var_names)).astype(str).str.replace(r'^GRCm38_','',regex=True)
    is_mm=gid.str.startswith('ENSMUSG').values
    sym=pd.Index(a.var_names).str.replace(r'^(GRCm38_|mm10___|mm10_)','',regex=True)
    am=a[:,is_mm].copy(); am.var_names=sym[is_mm]; am.var_names_make_unique()
    Xm=am.X.tocsr(); keep=np.asarray((Xm>0).sum(0)).ravel()>0
    b=am[:,keep].copy(); sc.pp.normalize_total(b,target_sum=1e4); sc.pp.log1p(b)
    rr=dc.mt.ulm(data=b,net=net,tmin=5,verbose=False); src=rr if rr is not None else b
    est=src.obsm['score_ulm'].reindex(b.obs_names)
    t=pd.read_parquet(f'targets_mouse/{sid}.parquet')
    assert (t.barcode.values==a.obs_names.astype(str).values).all()
    for n in est.columns: t[f'rand0ulm__{n}']=est[n].values
    t.to_parquet(f'targets_mouse/{sid}.parquet',index=False); print(sid, est.shape[1], flush=True)
