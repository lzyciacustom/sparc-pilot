import pandas as pd, numpy as np, scipy.sparse as sp, anndata as ad
_g=pd.read_parquet('data/hest1k_crc/human_gene_db.parquet',columns=['ensembl_gene_id','symbol']).dropna()
E2S=dict(zip(_g.ensembl_gene_id,_g.symbol))
import re
def to_symbols(a):
    v0=str(a.var_names[0])
    if v0.startswith('GRCh38_'):
        hum=np.array([str(v).startswith('GRCh38_') for v in a.var_names])
        a=a[:,hum].copy(); a.var_names=[re.sub(r'^GRCh38_+','',str(v)) for v in a.var_names]; return a
    if not v0.startswith('ENSG'): return a
    sym=pd.Series([E2S.get(str(v).split('.')[0]) for v in a.var_names],index=a.var_names)
    keep=sym.notna().values; a=a[:,keep]; s=sym[keep].values
    X=a.X.tocsc() if sp.issparse(a.X) else sp.csc_matrix(a.X)
    u,inv=np.unique(s,return_inverse=True)
    A=sp.csr_matrix((np.ones(len(inv)),(np.arange(len(inv)),inv)),shape=(len(inv),len(u)))
    Xn=(X@A).tocsr()
    return ad.AnnData(X=Xn,obs=a.obs.copy(),var=pd.DataFrame(index=u))
