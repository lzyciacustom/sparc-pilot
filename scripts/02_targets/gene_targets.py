import os, sys, glob, json, h5py, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, warnings; warnings.filterwarnings('ignore')
sys.path.append(os.getcwd()); from ens2sym import to_symbols
import encoders as ENC; FD = ENC.feat_dir('crc')
R='data/hest1k_crc'; SH,NS=int(sys.argv[1]),int(sys.argv[2])
panel=json.load(open('gene_panel.json'))['panel']
ids=sorted(os.path.basename(f)[:-3] for f in glob.glob(f'{FD}/*.h5'))
for ii,s in enumerate(ids):
    if ii%NS!=SH or os.path.exists(f'genes_crc/{s}.h5'): continue
    with h5py.File(f'{FD}/{s}.h5') as f: bc=np.array([b[0].decode() if isinstance(b[0],bytes) else str(b[0]) for b in f['barcode'][:]])
    a=to_symbols(ad.read_h5ad(f'{R}/st/{s}.h5ad')); a.var_names_make_unique()
    idx=pd.Index(a.obs_names).get_indexer(bc)
    X=a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    lib=np.asarray(X.sum(1)).ravel()
    gi=pd.Index(a.var_names).get_indexer(panel)
    out=np.full((len(bc),len(panel)),np.nan,np.float32)
    ok=idx>=0; have=gi>=0
    sub=X[idx[ok]][:,gi[have]].toarray()
    sub=np.log1p(sub/np.maximum(lib[idx[ok]],1)[:,None]*1e4)
    tmp=np.full((ok.sum(),len(panel)),np.nan,np.float32); tmp[:,have]=sub; out[ok]=tmp
    with h5py.File(f'genes_crc/{s}.h5.tmp','w') as o: o['X']=out.astype(np.float16); o['barcode']=bc.astype('S')
    os.replace(f'genes_crc/{s}.h5.tmp',f'genes_crc/{s}.h5'); print(s,out.shape,round(have.mean(),3),flush=True)
