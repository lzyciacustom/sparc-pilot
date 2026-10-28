"""Reference-derived composition baseline b_M(pi, depth): expected ULM program score of a spot with transcript-fraction
composition pi and log1p total UMI depth, learned from simulated pseudospots of known composition.

Pseudospots: 1 + Poisson(9) GSE132465 cells drawn by type from pi ~ mixture of Dirichlet(1), Dirichlet(0.2) and
dominant-type draws; summed raw UMI, thinned to a log-uniform depth in [1e2, 1e5] (capped at the pseudospot total); library-size normalised (1e4),
log1p, ULM over a fixed gene universe (reference genes detected in >=90% of the 58 CRC sections).
Composition label = realised transcript fraction by type (the quantity NNLS on mean-fraction profiles estimates).
b_M: MLP (12 + standardised log depth) -> 256 -> 256 -> 36, fitted on 90% of pseudospots; held-out R^2 reported per program.
Outputs: comp/bM_<targets>.pt, comp/bM_fit.csv
"""
import os, sys, json, numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, torch
from torch import nn
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, WS)
from decon_resid import TYPES
from noise_ceiling import ulm, lognorm
import sparcmap_lib as L

C = f'{WS}/comp'; rng = np.random.default_rng(2026); torch.manual_seed(0)
G = sp.load_npz(f'{C}/gse132465_counts.npz').tocsr(); cells = pd.read_csv(f'{C}/gse132465_cells.csv')
genes = pd.read_csv(f'{C}/gse132465_genes.csv').iloc[:, 0].values
SETS = {'ulm': json.load(open(f'{WS}/sparc_programs_msigdb.json'))['sets'],
        'rand0ulm': json.load(open(f'{WS}/random_gene_sets.json'))['sets']['rand0']}

# gene universe
if not os.path.exists(f'{C}/universe.txt'):
    cnt = {}
    ids = sorted(pd.read_csv(f'{WS}/crc_folds.csv').id)
    for s in ids:
        a = ad.read_h5ad(f'{WS}/data/hest1k_crc/st/{s}.h5ad'); X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
        for g in set(np.array(a.var_names)[np.asarray((X > 0).sum(0)).ravel() > 0]): cnt[g] = cnt.get(g, 0) + 1
    U = sorted(g for g, c in cnt.items() if c >= 0.9 * len(ids)); open(f'{C}/universe.txt', 'w').write('\n'.join(U))
U = set(open(f'{C}/universe.txt').read().split('\n'))
gi = np.array([i for i, g in enumerate(genes) if g in U]); ug = genes[gi]
G = G[:, gi].tocsr()
lab = cells.type.values; byt = {t: np.where(lab == t)[0] for t in TYPES}
print('universe', len(ug), {t: len(v) for t, v in byt.items()}, flush=True)

N = 80000; K = len(TYPES)
mode = rng.integers(0, 3, N); pi = np.empty((N, K))
pi[mode == 0] = rng.dirichlet(np.ones(K), (mode == 0).sum())
pi[mode == 1] = rng.dirichlet(0.2 * np.ones(K), (mode == 1).sum())
m2 = mode == 2; dom = rng.integers(0, K, m2.sum()); base = rng.dirichlet(0.5 * np.ones(K), m2.sum()) * 0.3
base[np.arange(m2.sum()), dom] += 0.7; pi[m2] = base
ncell = 1 + rng.poisson(9, N)
rows, cols = [], []
for i in range(N):
    k = rng.multinomial(ncell[i], pi[i])
    for t, c in zip(TYPES, k):
        if c: rows += [i] * c; cols += list(rng.choice(byt[t], c))
A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, G.shape[0]))
lib = np.asarray(G.sum(1)).ravel()
T = sp.csr_matrix((lib[cols], (rows, [TYPES.index(lab[c]) for c in cols])), shape=(N, K)).toarray()
frac = T / T.sum(1, keepdims=True)
X = (A @ G).tocsr().astype(np.float64)
tot = np.asarray(X.sum(1)).ravel(); D = np.exp(rng.uniform(np.log(1e2), np.log(1e5), N))
p = np.minimum(1, D / tot); X.data = rng.binomial(X.data.astype(np.int64), np.repeat(p, np.diff(X.indptr))).astype(np.float64)
depth = np.log1p(np.asarray(X.sum(1)).ravel())
Xl = lognorm(X)
P = L.PROGRAMS; fit = []
ntr = int(0.9 * N)
for name, sets in SETS.items():
    Y = ulm(Xl, ug, sets).reindex(columns=P).values.astype(np.float32)
    ok = np.isfinite(Y).all(1); dmu, dsd = depth[ok][:ntr].mean(), depth[ok][:ntr].std()
    Xt = torch.tensor(np.column_stack([frac[ok], (depth[ok] - dmu) / dsd]), dtype=torch.float32); Yt = torch.tensor(Y[ok])
    mu, sd = Yt[:ntr].mean(0), Yt[:ntr].std(0)
    net = nn.Sequential(nn.Linear(K + 1, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, len(P)))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(60):
        perm = torch.randperm(ntr)
        for b in range(0, ntr, 1024):
            j = perm[b:b + 1024]; loss = ((net(Xt[j]) - (Yt[j] - mu) / sd) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): pr = net(Xt[ntr:]) * sd + mu
    yt = Yt[ntr:]; r2 = 1 - ((pr - yt) ** 2).sum(0) / ((yt - yt.mean(0)) ** 2).sum(0)
    # linear-in-pi reference and depth share for comparison
    Z = np.column_stack([np.ones(ok.sum()), frac[ok]]); b, *_ = np.linalg.lstsq(Z[:ntr], Y[ok][:ntr], rcond=None)
    r2lin = 1 - ((Z[ntr:] @ b - Y[ok][ntr:]) ** 2).sum(0) / ((Y[ok][ntr:] - Y[ok][ntr:].mean(0)) ** 2).sum(0)
    Zd = np.column_stack([Z, depth[ok]]); bd, *_ = np.linalg.lstsq(Zd[:ntr], Y[ok][:ntr], rcond=None)
    r2d = 1 - ((Zd[ntr:] @ bd - Y[ok][ntr:]) ** 2).sum(0) / ((Y[ok][ntr:] - Y[ok][ntr:].mean(0)) ** 2).sum(0)
    torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, dmu=float(dmu), dsd=float(dsd), dmin=float(depth[ok].min()), dmax=float(depth[ok].max()), types=TYPES, programs=P), f'{C}/bM_{name}.pt')
    for j, pp in enumerate(P): fit.append(dict(targets=name, program=pp, r2_mlp_pi_depth=float(r2[j]), r2_linear=r2lin[j], r2_linear_plus_depth=r2d[j]))
    print(name, 'held-out R2 mlp median', float(r2.median()), 'linear', np.median(r2lin), flush=True)
pd.DataFrame(fit).to_csv(f'{C}/bM_fit.csv', index=False)
