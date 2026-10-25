"""Spectra (scSpectra 0.2.1, CPU) on a slide subset. Global model (use_cell_types=False), L = 36 anchored + 5 free
factors, lam=0.01, 2000 epochs (default 10000 reduced for time), gene panel = program genes detected in >=5% spots
+ 500 HVGs, log1p(CP10k). Factor j (<36) is initialised from program j; we keep that anchoring."""
import os, sys, json, time, warnings; warnings.filterwarnings('ignore'); os.environ['TQDM_DISABLE'] = '1'
sys.path.append(os.getcwd())
import numpy as np, pandas as pd, anndata as ad, scanpy as sc, torch; torch.set_num_threads(12)
import sparcmap_lib as L, score_lib as SL
from Spectra import Spectra as SPX
sets = json.load(open('sparc_programs_msigdb.json'))['sets']; P = L.PROGRAMS; allg = set(g for p in P for g in sets[p])
SUB = ['ZEN36', 'ZEN40', 'ZEN48', 'ZEN49'] + sys.argv[1:]
for sid in SUB:
    of = f'results/spectra/{sid}.parquet'
    if os.path.exists(of): continue
    t0 = time.time(); C = SL.load_counts(sid); good = np.asarray(C['X'].sum(1)).ravel() > 0
    a = ad.AnnData(C['X'][good].copy()); a.var_names = C['genes']; a.obs_names = C['barcode'][good]
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    frac = np.asarray((a.X > 0).mean(0)).ravel(); sc.pp.highly_variable_genes(a, n_top_genes=500)
    keep = (a.var_names.isin(allg) & (frac >= 0.05)) | a.var.highly_variable.values
    b = a[:, keep].copy(); b.X = np.asarray(b.X.toarray(), dtype=np.float32); vs = set(b.var_names)
    gsd = {p: [g for g in sets[p] if g in vs] for p in P}
    torch.manual_seed(0); np.random.seed(0)
    SPX.est_spectra(adata=b, gene_set_dictionary=gsd, L=41, use_highly_variable=False, use_cell_types=False, lam=0.01,
                    num_epochs=2000, verbose=False, label_factors=False, filter_sets=False)
    S = b.obsm['SPECTRA_cell_scores']; F = b.uns['SPECTRA_factors']; vocab = np.array(b.var_names)
    ov = []
    for j, p in enumerate(P):
        top = set(vocab[np.argsort(-F[j])[:50]]); ov.append(len(top & set(gsd[p])) / min(50, len(gsd[p])))
    df = pd.DataFrame(S, columns=[f'spectra__{p}' for p in P] + [f'spectra__free{k}' for k in range(S.shape[1] - 36)])
    df.insert(0, 'barcode', b.obs_names); df.to_parquet(of)
    pd.DataFrame({'program': P, 'top50_overlap_coef': ov, 'n_genes_panel': [len(gsd[p]) for p in P]}).to_csv(f'results/spectra/{sid}_overlap.csv', index=False)
    print(sid, b.shape, f'{time.time()-t0:.0f}s', 'median overlap', np.median(ov), flush=True)
