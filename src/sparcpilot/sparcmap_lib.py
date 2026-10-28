"""Shared SPARC-Map reproduction library for the pilot studies.

Architecture imported at runtime from the public SPARC repo (preprocessing/map_models.py,
OriginalSpatialTransformer): patch + k=16 nearest neighbours, relative-coordinate
MLP (2->64->H), 3 transformer blocks, H=256, 8 heads; loss MSE + 0.5*(1-Pearson);
AdamW lr 1e-4; per-program robust (median/IQR) label normalisation, matching the
released checkpoint name `knn_transformer_robust_normalized_mse_pearson`.
Deviation: relative coordinates are scaled by the slide's median 1-NN distance
(`median_knn`) because HEST-Bench slides come from different scanners.
"""
from __future__ import annotations
import os, sys, json, h5py, numpy as np, pandas as pd, torch
from scipy.spatial import cKDTree

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WS, 'sparc_repo'))
from preprocessing.map_models import OriginalSpatialTransformer  # noqa: E402

sys.path.insert(0, WS)
from encoders import feat_dir as _feat_dir  # noqa: E402
FEAT_DIR = _feat_dir('', root=WS)  # HEST-Bench features of the selected encoder (SPARC_ENCODER, default h1)
TGT_DIR = os.path.join(WS, 'targets')
INV = pd.read_csv(os.path.join(WS, 'hest_bench_inventory.csv'))
PROGRAMS = list(json.load(open(os.path.join(WS, 'sparc_programs_msigdb.json')))['sets'])
METHODS = ['ssgsea', 'ulm', 'mlm']


def load_slide(sid, method='ssgsea', k=16, feat_dir=FEAT_DIR):
    """Return dict with features (N,1536 float32), targets (N,P), coords, knn idx (N,k), rel coords (N,k,2), depth."""
    with h5py.File(os.path.join(feat_dir, f'{sid}.h5')) as f:
        X = f['features'][:].astype(np.float32); xy = f['coords'][:].astype(np.float32)
        bc = np.array([b[0].decode() if isinstance(b[0], bytes) else str(b[0]) for b in f['barcode'][:]])
    t = pd.read_parquet(os.path.join(TGT_DIR, f'{sid}.parquet')).set_index('barcode')
    idx = pd.Index(t.index).get_indexer(bc); ok = idx >= 0
    X, xy, bc = X[ok], xy[ok], bc[ok]; t = t.iloc[idx[ok]]
    cols = [f'{method}__{p}' for p in PROGRAMS]
    Y = t.reindex(columns=cols).values.astype(np.float32)
    tree = cKDTree(xy); d, nn_ = tree.query(xy, k=k + 1)
    nn_ = nn_[:, 1:]; scale = np.median(d[:, 1])
    rel = (xy[nn_] - xy[:, None, :]) / max(scale, 1e-6)
    return dict(sid=sid, X=X, Y=Y, xy=xy, nn=nn_, rel=rel.astype(np.float32), barcode=bc,
                depth=np.log1p(t['total_counts'].values.astype(np.float32)))


def robust_stats(Ys):
    Y = np.concatenate(Ys); med = np.nanmedian(Y, 0)
    iqr = np.nanpercentile(Y, 75, 0) - np.nanpercentile(Y, 25, 0)
    return med, np.where(iqr > 1e-8, iqr, 1.0)


def pearson_loss(pred, y, mask, w=None):
    pm = (pred * mask).sum(0) / mask.sum(0).clamp(min=1)
    ym = (y * mask).sum(0) / mask.sum(0).clamp(min=1)
    pc, yc = (pred - pm) * mask, (y - ym) * mask
    # eps inside sqrt avoids NaN gradients for fully masked columns
    r = (pc * yc).sum(0) / ((pc.pow(2).sum(0) + 1e-8).sqrt() * (yc.pow(2).sum(0) + 1e-8).sqrt())
    valid = mask.sum(0) > 1
    if not bool(valid.any()): return (pred * 0).sum()
    if w is None: return 1 - r[valid].mean()
    return 1 - (r[valid] * w[valid]).sum() / w[valid].sum()


class SlideBatcher:
    """Samples (center, neighbours, rel, y) batches from a list of loaded slides held on GPU."""
    def __init__(self, slides, med, iqr, device='cuda'):
        self.dev = device; self.s = []
        for s in slides:
            Yn = (s['Y'] - med) / iqr
            self.s.append(dict(X=torch.tensor(s['X'], device=device), Y=torch.tensor(np.nan_to_num(Yn), device=device),
                               M=torch.tensor(~np.isnan(Yn), device=device).float(),
                               nn=torch.tensor(s['nn'], device=device), rel=torch.tensor(s['rel'], device=device)))
        self.sizes = np.array([len(s['X']) for s in slides])

    def get(self, si, idx):
        s = self.s[si]; idx = torch.as_tensor(idx, device=self.dev)
        return s['X'][idx], s['X'][s['nn'][idx]], s['rel'][idx], s['Y'][idx], s['M'][idx]

    def random_batch(self, bs, rng):
        si = rng.choice(len(self.s), p=self.sizes / self.sizes.sum())
        return self.get(si, rng.integers(0, self.sizes[si], bs))


def build_model(P, input_dim=1536, **kw):
    return OriginalSpatialTransformer(input_dim=input_dim, output_dim=P, **kw)


def default_forward(model, c, n, r):
    return model(c, n, r)


def train(train_slides, val_slides=None, epochs=30, bs=256, lr=1e-4, wd=1e-2, lam=0.5, seed=0,
          model=None, forward=default_forward, loss_extra=None, steps_per_epoch=None, verbose=False, prog_w=None, norm=None, batcher_cls=None):
    """Train a SPARC-Map-style model. Returns model, (med, iqr), history."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    med, iqr = norm if norm is not None else robust_stats([s['Y'] for s in train_slides])
    P = train_slides[0]['Y'].shape[1]
    W = None if prog_w is None else torch.tensor(prog_w, device='cuda', dtype=torch.float32)
    model = (model or build_model(P)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    tb = (batcher_cls or SlideBatcher)(train_slides, med, iqr)
    spe = steps_per_epoch or max(50, int(tb.sizes.sum() / bs))
    best, best_state, hist = -1e9, None, []
    for ep in range(epochs):
        model.train()
        for _ in range(spe):
            c, n, r, y, m = tb.random_batch(bs, rng)
            p = forward(model, c, n, r)
            mw = m if W is None else m * W
            loss = ((p - y).pow(2) * mw).sum() / mw.sum().clamp(min=1e-6) + lam * pearson_loss(p, y, m, W)
            if loss_extra is not None: loss = loss + loss_extra(model, p, y, m)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if val_slides:
            res = evaluate(model, val_slides, (med, iqr), forward=forward)
            score = res['pearson'].mean(); hist.append(score)
            if score > best: best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
            if verbose: print(ep, round(score, 4), flush=True)
    if best_state is not None: model.load_state_dict(best_state)
    return model, (med, iqr), hist


@torch.no_grad()
def predict(model, slide, norm, forward=default_forward, bs=2048):
    model.eval(); med, iqr = norm
    X = torch.tensor(slide['X'], device='cuda'); nn_ = torch.tensor(slide['nn'], device='cuda')
    rel = torch.tensor(slide['rel'], device='cuda'); out = []
    for s in range(0, len(X), bs):
        i = torch.arange(s, min(s + bs, len(X)), device='cuda')
        out.append(forward(model, X[i], X[nn_[i]], rel[i]).float().cpu().numpy())
    return np.concatenate(out) * iqr + med


def _pearson_cols(A, B):
    out = []
    for j in range(A.shape[1]):
        ok = ~(np.isnan(A[:, j]) | np.isnan(B[:, j]))
        out.append(np.corrcoef(A[ok, j], B[ok, j])[0, 1] if ok.sum() > 10 and B[ok, j].std() > 0 else np.nan)
    return np.array(out)


def residualize(Y, Z):
    """Regress columns of Y on covariates Z (N,q) with intercept; return residuals (NaN-safe per column)."""
    Zc = np.column_stack([np.ones(len(Z)), Z]); R = np.full_like(Y, np.nan)
    for j in range(Y.shape[1]):
        ok = ~np.isnan(Y[:, j])
        b, *_ = np.linalg.lstsq(Zc[ok], Y[ok, j], rcond=None); R[ok, j] = Y[ok, j] - Zc[ok] @ b
    return R


def evaluate(model, slides, norm, forward=default_forward):
    """Per-slide, per-program metrics: raw Pearson, centred Pearson (per-patch mean across programs removed
    from z-scored maps, as in SPARC's spatial analysis), and depth-residualised Pearson."""
    rows = []
    for s in slides:
        pr = predict(model, s, norm, forward); Y = s['Y']
        zs = lambda A: (A - np.nanmean(A, 0)) / (np.nanstd(A, 0) + 1e-9)
        Pz, Yz = zs(pr), zs(Y)
        Pc = Pz - np.nanmean(Pz, 1, keepdims=True); Yc = Yz - np.nanmean(Yz, 1, keepdims=True)
        r_raw = _pearson_cols(pr, Y); r_c = _pearson_cols(Pc, Yc)
        r_d = _pearson_cols(residualize(pr, s['depth'][:, None]), residualize(Y, s['depth'][:, None]))
        for j, p in enumerate(PROGRAMS):
            rows.append(dict(sample=s['sid'], program=p, pearson=r_raw[j], pearson_centered=r_c[j], pearson_depthres=r_d[j]))
    return pd.DataFrame(rows)


def wt_folds(n_folds=4):
    """Patient-group folds over Visium whole-transcriptome slides; each HEST-Bench patient group stays in one fold.
    READ's two patient groups are forced into different folds."""
    wt = INV[INV.platform == 'visium_wt'].copy()
    g = wt.groupby('group')['n_spots'].sum().sort_values(ascending=False)
    load = np.zeros(n_folds); assign = {}
    for grp in g.index:
        k = int(np.argmin(load))
        if grp.startswith('READ') and any(v == k for gg, v in assign.items() if gg.startswith('READ')):
            k = int(np.argsort(load)[1])
        assign[grp] = k; load[k] += g[grp]
    wt['cv_fold'] = wt.group.map(assign)
    return wt[['sample', 'task', 'group', 'cv_fold', 'n_spots']]
