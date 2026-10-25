"""Active selection of sections (Track C): random vs max-uncertainty vs EPIG batch selection.

Uses the shared SPARC-Map architecture (L.build_model) and loss (MSE + 0.5*(1-Pearson)), robust label
normalisation, AdamW lr 1e-4 wd 1e-2, bs 256, grad-clip 1.0 -- identical to L.train -- but with
(a) a cheap schedule (EPOCHS epochs, early stopping on a fixed inner-validation patient group),
(b) slide-level Bayesian-bootstrap sampling weights per ensemble member (Dirichlet(1)), and
(c) bf16 autocast + fp16 feature storage on GPU to keep memory/time low.
"""
import os, sys, json, time, numpy as np, pandas as pd, torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import sparcmap_lib as L

EPOCHS = int(os.environ.get('AL_EPOCHS', 8))
M = int(os.environ.get('AL_M', 5))
DEV = 'cuda'


class GPUSlides:
    """All slides of one fold on GPU (features fp16)."""
    def __init__(self, S):
        self.S = S; self.g = {}
        for sid, s in S.items():
            self.g[sid] = dict(X=torch.tensor(s['X'], device=DEV, dtype=torch.float16),
                               nn=torch.tensor(s['nn'], device=DEV), rel=torch.tensor(s['rel'], device=DEV))

    def batch(self, sid, idx):
        g = self.g[sid]
        return g['X'][idx], g['X'][g['nn'][idx]], g['rel'][idx]


def fwd(model, gs, sid, idx):
    c, n, r = gs.batch(sid, idx)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(c, n, r).float()


@torch.no_grad()
def predict_norm(model, gs, sid, idx=None, bs=4096):
    """Predictions in robust-normalised units for slide sid (optionally a subset idx of spots)."""
    model.eval(); N = len(gs.S[sid]['X'])
    idx = torch.arange(N, device=DEV) if idx is None else torch.as_tensor(idx, device=DEV)
    return torch.cat([fwd(model, gs, sid, idx[i:i + bs]) for i in range(0, len(idx), bs)]).cpu().numpy()


def train_member(gs, train_ids, val_ids, seed, weights=None, epochs=EPOCHS, bs=256, lr=1e-4, wd=1e-2, lam=0.5):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    S = gs.S
    med, iqr = L.robust_stats([S[s]['Y'] for s in train_ids])
    Yn = {s: torch.tensor(np.nan_to_num((S[s]['Y'] - med) / iqr), device=DEV) for s in train_ids}
    Mk = {s: torch.tensor(~np.isnan(S[s]['Y']), device=DEV).float() for s in train_ids}
    sizes = np.array([len(S[s]['X']) for s in train_ids], float)
    p = sizes / sizes.sum() if weights is None else (sizes * weights) / (sizes * weights).sum()
    model = L.build_model(len(L.PROGRAMS)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    spe = max(50, int(sizes.sum() / bs))
    Yv = {s: S[s]['Y'] for s in val_ids}
    best, best_state = -1e9, None
    for ep in range(epochs):
        model.train()
        for _ in range(spe):
            si = train_ids[rng.choice(len(train_ids), p=p)]
            idx = torch.as_tensor(rng.integers(0, int(sizes[train_ids.index(si)]), bs), device=DEV)
            pr = fwd(model, gs, si, idx); y = Yn[si][idx]; m = Mk[si][idx]
            loss = ((pr - y).pow(2) * m).sum() / m.sum().clamp(min=1) + lam * L.pearson_loss(pr, y, m)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        # early stopping on inner validation group: mean per-slide per-program Pearson
        sc = np.nanmean([np.nanmean(L._pearson_cols(predict_norm(model, gs, v), Yv[v])) for v in val_ids])
        if sc > best: best, best_state = sc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    return model, (med, iqr), best


def train_ensemble(gs, train_ids, val_ids, base_seed):
    ens = []
    for m in range(M):
        rng = np.random.default_rng(base_seed * 100 + m)
        w = rng.dirichlet(np.ones(len(train_ids)))  # Bayesian bootstrap over labelled slides
        ens.append(train_member(gs, train_ids, val_ids, seed=base_seed * 100 + m, weights=w))
    return ens


def ens_predict(ens, gs, sid, idx=None, unit='norm'):
    """(M, n, P) member predictions. unit='norm' -> common normalisation of member 0; 'raw' -> original target units."""
    out = []
    for model, (med, iqr), _ in ens:
        pn = predict_norm(model, gs, sid, idx)
        raw = pn * iqr + med
        out.append(raw)
    out = np.stack(out)
    if unit == 'norm':  # express all members on one common robust scale so variances are comparable
        med0, iqr0 = ens[0][1]
        out = (out - med0) / iqr0
    return out


def evaluate_ensemble(ens, gs, test_ids):
    rows = []
    for sid in test_ids:
        s = gs.S[sid]; pr = ens_predict(ens, gs, sid, unit='raw').mean(0); Y = s['Y']
        r_raw = L._pearson_cols(pr, Y)
        r_d = L._pearson_cols(L.residualize(pr, s['depth'][:, None]), L.residualize(Y, s['depth'][:, None]))
        for j, p in enumerate(L.PROGRAMS):
            rows.append(dict(sample=sid, program=p, pearson=r_raw[j], pearson_depthres=r_d[j]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ acquisition
RHO2_CLIP = 0.999


def member_dev(A):
    """A (M, n, P) -> per-program unit vectors over members, shape (P, n, M)."""
    D = A - A.mean(0, keepdims=True)
    D = np.transpose(D, (2, 1, 0))
    return D / (np.linalg.norm(D, axis=2, keepdims=True) + 1e-12)


def pair_mi(Dc, Dt):
    """Mean over (candidate, target) pairs and programs of -0.5 log(1 - rho^2). Dc (P,nc,M), Dt (P,nt,M)."""
    rho = np.einsum('pim,pjm->pij', Dc, Dt)
    r2 = np.clip(rho ** 2, 0, RHO2_CLIP)
    return float((-0.5 * np.log1p(-r2)).mean()), float(r2.mean())


def mean_r2(Da, Db):
    rho = np.einsum('pim,pjm->pij', Da, Db)
    return float(np.clip(rho ** 2, 0, RHO2_CLIP).mean())


def score_candidates(ens, gs, cand_ids, target_ids, rng, n_cand=300, n_tgt_per_slide=100):
    """Return per-candidate: maxunc (mean over all spots of summed ensemble variance), epig, and member-deviation
    matrices on sampled spots (for the redundancy penalty)."""
    tgt = []
    for t in target_ids:
        n = len(gs.S[t]['X']); idx = np.sort(rng.choice(n, min(n, n_tgt_per_slide), replace=False))
        tgt.append(ens_predict(ens, gs, t, idx))
    Dt = member_dev(np.concatenate(tgt, 1))
    out = {}
    for c in cand_ids:
        A = ens_predict(ens, gs, c)                       # all spots, common normalised units
        mu = float(A.var(0, ddof=1).sum(1).mean())        # summed-over-programs variance, mean over spots
        n = A.shape[1]; idx = np.sort(rng.choice(n, min(n, n_cand), replace=False))
        Dc = member_dev(A[:, idx])
        epig, r2 = pair_mi(Dc, Dt)
        out[c] = dict(maxunc=mu, epig=epig, epig_meanr2=r2, D=Dc)
    return out


def select_batch(strategy, scores, cand_ids, b, rng):
    """Batch rules.
    random : b slides uniformly without replacement.
    maxunc : top-b by mean summed ensemble variance (no diversity term).
    epig   : greedy; first = argmax EPIG; each next = argmax EPIG(s) * (1 - max_{c in chosen} R(s,c)),
             R(s,c) = mean over programs and sampled spot pairs of rho^2 between member-deviation vectors of s and c,
             i.e. candidates whose ensemble disagreement pattern duplicates an already-chosen slide are discounted."""
    cand = list(cand_ids)
    if strategy == 'random':
        return list(rng.choice(cand, b, replace=False))
    if strategy == 'maxunc':
        return sorted(cand, key=lambda c: -scores[c]['maxunc'])[:b]
    if strategy == 'featmatch':  # label-free, model-free control: top-b by H&E-embedding similarity to target slides
        return sorted(cand, key=lambda c: -scores[c]['featsim'])[:b]
    chosen = []
    for _ in range(b):
        best, bs = None, -np.inf
        for c in cand:
            if c in chosen: continue
            red = max([mean_r2(scores[c]['D'], scores[d]['D']) for d in chosen], default=0.0)
            s = scores[c]['epig'] * (1 - red)
            if s > bs: best, bs = c, s
        chosen.append(best)
    return chosen
