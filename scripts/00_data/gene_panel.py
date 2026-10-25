import os, sys, glob, json, re, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, warnings; warnings.filterwarnings('ignore')
sys.path.append(os.getcwd()); from ens2sym import to_symbols, E2S
R='data/hest1k_crc'; ids=sorted(os.path.basename(f)[:-3] for f in glob.glob('features/hoptimus0_crc/*.h5'))
def names(sid):
    v=ad.read_h5ad(f'{R}/st/{sid}.h5ad',backed='r').var_names
    v0=str(v[0])
    if v0.startswith('GRCh38_'): return {re.sub(r'^GRCh38_+','',x) for x in v if x.startswith('GRCh38_')}
    if v0.startswith('ENSG'): return {E2S[x.split('.')[0]] for x in v if x.split('.')[0] in E2S}
    return set(v)
from collections import Counter
cnt=Counter(); 
for s in ids: cnt.update(names(s))
common=[g for g,c in cnt.items() if c>=0.95*len(ids)]
progs=json.load(open('sparc_programs_msigdb.json'))['sets']; pg=set(g for v in progs.values() for g in v)
prog_in=sorted(set(common)&pg)
rng=np.random.default_rng(0); acc=[]
for s in ids[::3]:
    a=to_symbols(ad.read_h5ad(f'{R}/st/{s}.h5ad')); a.var_names_make_unique()
    a=a[rng.choice(a.n_obs,min(150,a.n_obs),replace=False)]
    g=[x for x in common if x in a.var_names]; X=a[:,g].X; X=X.toarray() if sp.issparse(X) else np.asarray(X)
    X=np.log1p(X/np.maximum(np.asarray(a.X.sum(1)).reshape(-1,1),1)*1e4); acc.append(pd.DataFrame(X,columns=g))
M=pd.concat(acc).fillna(0); disp=M.var()/(M.mean()+1e-3)
hvg=[g for g in disp.sort_values(ascending=False).index if g not in prog_in and M[g].mean()>0.05][:max(0,4000-len(prog_in))]
panel=prog_in+hvg
json.dump({'panel':panel,'n_prog_genes':len(prog_in),'n_common':len(common),'n_slides':len(ids)},open('gene_panel.json','w'))
print('common',len(common),'program genes in common',len(prog_in),'of',len(pg),'panel',len(panel))
