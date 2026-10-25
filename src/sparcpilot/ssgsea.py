"""Closed-form single-sample GSEA on GPU (rank-normalised, weight 0.25), equivalent to gseapy.ssgsea."""
import numpy as np, pandas as pd, scipy.sparse as sp, torch


def ssgsea_gpu(X, genes, sets, w=0.25, min_size=5, bs=1024, dev='cuda'):
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
