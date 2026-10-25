"""B2 prediction-head comparison. Usage: python run_b2.py <worker_id> <backbone_variant> [bench]
Backbone variants follow run_b1.VARIANTS names restricted to transformer kinds; 'base_k16' is realised as
ConcertTransformer with all biases off (architecture-equivalent to SPARC-Map's OriginalSpatialTransformer)."""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.append(os.getcwd())
import numpy as np, pandas as pd, torch
from torch import nn
from scipy.stats import norm as N01
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
torch.set_num_threads(2)
import sparcmap_lib as L, spatial_models as SM
from run_b1 import VARIANTS, split, get_comp

HEADS = ['gauss', 'flow']
EXTRA = ['gauss_pure', 'flow_pure']
METHODS = ['ssgsea', 'mlm']; SEEDS = [0, 1]; EPOCHS = 30; NS, NSTEP = 16, 32
OUT = 'b2_runs'; os.makedirs(OUT, exist_ok=True); os.makedirs('b2_locks', exist_ok=True)
Z50, Z90 = N01.ppf(0.75), N01.ppf(0.95)


class HeadModel(nn.Module):
    def __init__(self, backbone, head, P, hidden=256):
        super().__init__()
        self.bb = backbone; self.kind = head; self.P = P
        self.det = backbone.head
        if head.startswith('gauss'): self.g = SM.GaussianHead(hidden, P, rank=4)
        if head.startswith('flow'): self.f = SM.FlowHead(hidden, P)

    def forward(self, c, n, r):
        return self.bb.embed(c, n, r)


def bf(fn, *a):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return fn(*a)


def train_loss(model, c, n, r, y, m, lam=0.5):
    h = bf(model, c, n, r).float()
    base = lambda p: ((p - y).pow(2) * m).sum() / m.sum().clamp(min=1) + lam * L.pearson_loss(p, y, m)
    k = model.kind
    if k == 'det': return base(model.det(h))
    if k.startswith('gauss'):
        mu, Fm, d = model.g(h); l = SM.GaussianHead.nll(mu, Fm, d, y, m)
        return l + (base(mu) if k == 'gauss' else 0.)
    if k.startswith('flow'):
        l = model.f.loss(h, y, m)
        return l + (base(model.det(h)) if k == 'flow' else 0.)


@torch.no_grad()
def predict_full(model, s, normst, ns=NS, steps=NSTEP, bs=1024, seed=0):
    """Returns point prediction (orig units), and in normalised units: mu, sd (gauss), samples (N,ns,P)."""
    model.eval(); med, iqr = normst; g = torch.Generator(device='cuda').manual_seed(seed)
    X = torch.tensor(s['X'], device='cuda'); nn_ = torch.tensor(s['nn'], device='cuda'); rel = torch.tensor(s['rel'], device='cuda')
    pts, mus, sds, sams = [], [], [], []
    for a in range(0, len(X), bs):
        i = torch.arange(a, min(a + bs, len(X)), device='cuda')
        h = bf(model, X[i], X[nn_[i]], rel[i]).float()
        if model.kind == 'det':
            pts.append(model.det(h))
        elif model.kind.startswith('gauss'):
            mu, Fm, d = model.g(h); sd = (Fm.pow(2).sum(-1) + d).sqrt()
            eps = torch.randn(len(i), ns, Fm.shape[-1], device='cuda', generator=g)
            e2 = torch.randn(len(i), ns, model.P, device='cuda', generator=g)
            smp = mu[:, None] + torch.einsum('bpr,bsr->bsp', Fm, eps) + d.sqrt()[:, None] * e2
            pts.append(mu); mus.append(mu); sds.append(sd); sams.append(smp)
        else:
            torch.manual_seed(seed + a)
            smp = bf(model.f.sample, h, ns, steps).float()
            pts.append(smp.mean(1)); sams.append(smp)
    cat = lambda z: torch.cat(z).cpu().numpy() if z else None
    pn = cat(pts)
    return dict(point=pn * iqr + med, point_n=pn, mu=cat(mus), sd=cat(sds), samples=cat(sams))


def val_score(model, slides, normst):
    sc = []
    for s in slides:
        pr = predict_full(model, s, normst, ns=8, steps=16)['point']
        sc.append(np.nanmean(L._pearson_cols(pr, s['Y'])))
    return float(np.mean(sc))


def fit(model, tr, va, seed, epochs=EPOCHS, bs=1024, lr=2e-4, wd=1e-2):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    med, iqr = L.robust_stats([s['Y'] for s in tr]); model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    tb = L.SlideBatcher(tr, med, iqr); spe = max(50, int(tb.sizes.sum() / bs))
    best, best_state, hist = -1e9, None, []
    for ep in range(epochs):
        model.train()
        for _ in range(spe):
            c, n, r, y, m = tb.random_batch(bs, rng)
            loss = train_loss(model, c, n, r, y, m)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sc = val_score(model, va, (med, iqr)); hist.append(sc); print(ep, round(sc, 4), flush=True)
        if sc > best: best, best_state = sc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); del tb; torch.cuda.empty_cache()
    return model, (med, iqr), hist


def hex_neighbours(xy):
    tree = cKDTree(xy); d, _ = tree.query(xy, k=2); r = 1.5 * np.median(d[:, 1])
    return tree.query_ball_point(xy, r)


def nbr_mean(A, nb):
    out = np.full_like(A, np.nan)
    for i, js in enumerate(nb):
        js = [j for j in js if j != i]
        if js: out[i] = np.nanmean(A[js], 0)
    return out


def rankcols(A):
    return pd.DataFrame(A).rank().values


def corrmat(A, B=None):
    """Pearson correlation between columns of A and columns of B (NaN rows dropped pairwise via fill)."""
    A = np.where(np.isnan(A), np.nanmean(A, 0), A); B = A if B is None else np.where(np.isnan(B), np.nanmean(B, 0), B)
    Az = (A - A.mean(0)) / (A.std(0) + 1e-12); Bz = (B - B.mean(0)) / (B.std(0) + 1e-12)
    return Az.T @ Bz / len(A)


def S_matrix(A, nb):
    M = nbr_mean(A, nb); ok = ~np.isnan(M).any(1)
    return corrmat(rankcols(A[ok]), rankcols(M[ok]))


def offdiag(C):
    return C[~np.eye(len(C), dtype=bool)]


def crps_gauss(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * N01.cdf(z) - 1) + 2 * N01.pdf(z) - 1 / np.sqrt(np.pi))


def crps_samples(y, S):
    t1 = np.abs(S - y[:, None]).mean(1)
    M = S.shape[1]; t2 = np.abs(S[:, :, None] - S[:, None]).sum((1, 2)) / (2 * M * (M - 1))
    return t1 - t2


def energy_score(y, S):
    t1 = np.linalg.norm(S - y[:, None], axis=-1).mean(1)
    M = S.shape[1]; t2 = np.linalg.norm(S[:, :, None] - S[:, None], axis=-1).sum((1, 2)) / (2 * M * (M - 1))
    return t1 - t2


def slide_metrics(s, P, normst, nb, lab_true):
    med, iqr = normst; Y = s['Y']; Yn = (Y - med) / iqr; pt = P['point']; Pn = P['point_n']
    r_raw = L._pearson_cols(pt, Y)
    r_d = L._pearson_cols(L.residualize(pt, s['depth'][:, None]), L.residualize(Y, s['depth'][:, None]))
    bnd = np.array([any(lab_true[j] != lab_true[i] for j in js if j != i) for i, js in enumerate(nb)])
    r_b = L._pearson_cols(pt[bnd], Y[bnd]) if bnd.sum() > 20 else np.full(len(L.PROGRAMS), np.nan)
    r_i = L._pearson_cols(pt[~bnd], Y[~bnd]) if (~bnd).sum() > 20 else np.full(len(L.PROGRAMS), np.nan)
    Ct = corrmat(Y); St = S_matrix(Y, nb)
    out = dict(pearson=np.nanmean(r_raw), pearson_depthres=np.nanmean(r_d), pearson_boundary=np.nanmean(r_b),
               pearson_interior=np.nanmean(r_i), boundary_frac=bnd.mean())

    def cov_stats(A, tag):
        Cp = corrmat(A); Sp = S_matrix(A, nb)
        out[f'corrF_{tag}'] = np.linalg.norm(Cp - Ct); out[f'SF_{tag}'] = np.linalg.norm(Sp - St)
        out[f'S_r_{tag}'] = np.corrcoef(Sp.ravel(), St.ravel())[0, 1]
        out[f'meanoffcorr_{tag}'] = offdiag(Cp).mean()
    cov_stats(pt, 'point')
    out['meanoffcorr_true'] = offdiag(Ct).mean()
    ok = ~np.isnan(Yn)
    if P['samples'] is not None:
        Sm = P['samples']; cov_stats(Sm[:, 0] * iqr + med, 'sample')
        lo50, hi50 = np.quantile(Sm, [0.25, 0.75], axis=1); lo90, hi90 = np.quantile(Sm, [0.05, 0.95], axis=1)
        c50 = ((Yn >= lo50) & (Yn <= hi50)); c90 = ((Yn >= lo90) & (Yn <= hi90))
        out['cov50_samp'] = np.nanmean(np.where(ok, c50, np.nan)); out['cov90_samp'] = np.nanmean(np.where(ok, c90, np.nan))
        Yf = np.where(ok, Yn, Sm.mean(1))
        out['crps_samp'] = np.nanmean(np.where(ok, crps_samples(Yf, Sm), np.nan)); out['energy'] = energy_score(Yf, Sm).mean()
        sd_s = Sm.std(1)
        out['spread_skill'] = np.nanmean([pd.Series(sd_s[:, j]).corr(pd.Series(np.abs(Yn[:, j] - Pn[:, j])), method='spearman') for j in range(Yn.shape[1])])
        out['_cov50_prog'] = np.nanmean(np.where(ok, c50, np.nan), 0); out['_cov90_prog'] = np.nanmean(np.where(ok, c90, np.nan), 0)
    if P['mu'] is not None:
        mu, sd = P['mu'], P['sd']; e = np.abs(Yn - mu)
        out['cov50_gauss'] = np.nanmean(np.where(ok, e <= Z50 * sd, np.nan)); out['cov90_gauss'] = np.nanmean(np.where(ok, e <= Z90 * sd, np.nan))
        out['crps_gauss'] = np.nanmean(crps_gauss(Yn, mu, sd))
        out['_cov50_prog'] = np.nanmean(np.where(ok, e <= Z50 * sd, np.nan), 0); out['_cov90_prog'] = np.nanmean(np.where(ok, e <= Z90 * sd, np.nan), 0)
    if P['samples'] is None:
        Yf = np.where(ok, Yn, Pn)
        out['crps_samp'] = np.nanmean(np.abs(Yn - Pn)); out['energy'] = np.linalg.norm(Yf - Pn, axis=1).mean()
    return out, r_raw, r_d, r_b


def true_clusters(s, seed=0):
    Y = s['Y']; Y = np.where(np.isnan(Y), np.nanmean(Y, 0), Y); Z = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    return KMeans(6, n_init=10, random_state=seed).fit_predict(Z)


GEO = {}


def run_one(wid, m, h, k, s, bbv, slides_fn):
    kk, kind, kw = VARIANTS[bbv]
    bbkw = kw if kind == 'concert' else dict(use_dist=False, use_cos=False, use_comp=False)
    tag = f'{m}__{bbv}__{h}__f{k}__s{s}'; out = f'{OUT}/{tag}.csv'
    S = slides_fn(m, kk); tr, va, te, vg = split(k); comp = get_comp(k, S)
    torch.manual_seed(100 * s + k)
    model = HeadModel(SM.ConcertTransformer(1536, len(L.PROGRAMS), comp=comp, **bbkw), h, len(L.PROGRAMS))
    t = time.time()
    model, normst, hist = fit(model, [S[x] for x in tr], [S[x] for x in va], seed=100 * s + k)
    srows, prows, preds = [], [], {}
    for sid in te:
        sl = S[sid]
        if sid not in GEO: GEO[sid] = (hex_neighbours(sl['xy']), true_clusters(sl))
        nb, lab = GEO[sid]
        P = predict_full(model, sl, normst, seed=1000 + s)
        met, r_raw, r_d, r_b = slide_metrics(sl, P, normst, nb, lab)
        c50p, c90p = met.pop('_cov50_prog', None), met.pop('_cov90_prog', None)
        srows.append(dict(sample=sid, **met))
        for j, p in enumerate(L.PROGRAMS):
            prows.append(dict(sample=sid, program=p, pearson=r_raw[j], pearson_depthres=r_d[j], pearson_boundary=r_b[j],
                              cov50=np.nan if c50p is None else c50p[j], cov90=np.nan if c90p is None else c90p[j]))
        preds[sid] = P['point'].astype(np.float32); del P
    meta = dict(method=m, backbone=bbv, head=h, fold=k, seed=s, best_epoch=int(np.argmax(hist)))
    sdf = pd.DataFrame(srows).assign(**meta); pdf = pd.DataFrame(prows).assign(**meta)
    pdf.to_csv(f'{OUT}/{tag}.prog.csv', index=False); np.savez_compressed(f'{OUT}/{tag}.pred.npz', **preds)
    sdf.to_csv(out + '.tmp', index=False); os.replace(out + '.tmp', out)
    print(wid, 'B2', tag, f'{time.time()-t:.0f}s', sdf[[c for c in ['pearson', 'pearson_depthres', 'cov50_gauss', 'cov90_gauss', 'cov50_samp', 'cov90_samp', 'crps_samp'] if c in sdf]].mean().round(3).to_dict(), flush=True)
    del model
