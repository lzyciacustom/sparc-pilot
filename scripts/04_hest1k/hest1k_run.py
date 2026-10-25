"""HEST-1k-scale pilot driver: train on the HEST-1k human Visium pool, test on 58 CRC slides (6 group folds).

Usage: python hest1k_run.py --target ssgsea --model base --head det --seed 0 [--folds 0,1,2,3,4,5] [--smoke]
targets : ssgsea | ulm | mlm | ssgsea_depthres | rand{0,1,2}ssgsea | rand0ulm
models  : base (SPARC-Map, distance-only k=16) | mlp (centre only) | concert (attention-bias, Track B) |
          gpk16 / gpk48 (non-stationary Gaussian-kernel aggregation, k=16/48; CONCERT-style spatial kernel)
heads   : det | gauss (low-rank Gaussian, rank 4)
All runs: bs 1024, lr 2e-4, bf16 autocast, 30 epochs (sqrt-scaled from SPARC's bs256/lr1e-4; identical for all variants).
Outputs: results/<target>__<model>__<head>__s<seed>.csv (per CRC test slide x program metrics).
"""
import os, sys, time, argparse, math, numpy as np, pandas as pd, torch
from torch import nn
WS = os.path.dirname(os.path.abspath(__file__)); sys.path.append(WS); sys.path.append(os.path.join(WS, 'trackB'))
import sparcmap_lib as L
from spatial_models import CentreMLP, ConcertTransformer, GaussianHead, fit_compartments

ap = argparse.ArgumentParser()
ap.add_argument('--target', required=True); ap.add_argument('--model', default='base'); ap.add_argument('--head', default='det')
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--folds', default='0,1,2,3,4,5')
ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--smoke', action='store_true')
ap.add_argument('--crc_only', action='store_true', help='train on CRC training folds only (no HEST-1k pool)')
ap.add_argument('--feat_dir', default=os.path.join(WS, 'features', 'hoptimus0_crc'))
ap.add_argument('--tgt_dir', default=os.path.join(WS, 'targets_crc'))
args = ap.parse_args()
L.TGT_DIR = args.tgt_dir
OUT = os.path.join(WS, 'results'); os.makedirs(OUT, exist_ok=True)
tag = f'{args.target}__{args.model}__{args.head}__s{args.seed}' + ('__crconly' if args.crc_only else '') + ('__smoke' if args.smoke else '')
K_NB = 48 if args.model == 'gpk48' else 16

crc = pd.read_csv(os.path.join(WS, 'crc_folds.csv'))
pool = pd.read_csv(os.path.join(WS, 'hest1k_train_pool.csv'))
ok = lambda s: os.path.exists(f'{args.feat_dir}/{s}.h5') and os.path.exists(f'{args.tgt_dir}/{s}.parquet')
crc = crc[crc.id.map(ok)]; crc_ids = set(crc.id)
pool_ids = [s for s in pool.id if ok(s) and s not in crc_ids]
if args.smoke: pool_ids = pool_ids[:6]
if args.crc_only: pool_ids = []


def load(sid):
    m = 'ssgsea' if args.target == 'ssgsea_depthres' else args.target
    s = L.load_slide(sid, m, k=K_NB, feat_dir=args.feat_dir)
    if args.target == 'ssgsea_depthres': s['Y'] = L.residualize(s['Y'], s['depth'][:, None]).astype(np.float32)
    return s


t0 = time.time()
S = {}
for s in pool_ids + sorted(crc_ids):
    try:
        x = load(s)
        if np.isfinite(x['Y']).any(axis=0).sum() >= 30: S[s] = x
    except Exception as e:
        print('skip', s, repr(e)[:80], flush=True)
pool_ids = [s for s in pool_ids if s in S]
print(f'loaded {len(S)} slides ({len(pool_ids)} pool non-CRC) in {time.time()-t0:.0f}s', flush=True)


class GPKernelAgg(nn.Module):
    """Non-stationary Gaussian-kernel aggregation (CONCERT-style spatial kernel, posterior-mean form):
    z_i = [g(x_i), sum_j k_m(d_ij; l_im) g(x_j) / sum_j k_m(d_ij; l_im)  for m=1..M],
    k_m = exp(-d^2 / (2 l^2)), log l_im = base_m + a_m(g(x_i)) + b_m(compartment_i)."""
    def __init__(self, input_dim, P, comp=None, hidden=256, M=4, dropout=0.1):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.log_l = nn.Parameter(torch.log(torch.tensor([0.75, 1.5, 3.0, 6.0][:M])))
        self.l_mod = nn.Linear(hidden, M); nn.init.zeros_(self.l_mod.weight); nn.init.zeros_(self.l_mod.bias)
        self.use_comp = comp is not None
        if self.use_comp:
            self.register_buffer('feat_mu', torch.tensor(comp['mu'])); self.register_buffer('pca_W', torch.tensor(comp['W']))
            self.register_buffer('cent', torch.tensor(comp['C'])); self.l_comp = nn.Embedding(comp['k'], M); nn.init.zeros_(self.l_comp.weight)
        self.head = nn.Sequential(nn.LayerNorm(hidden * (M + 1)), nn.Linear(hidden * (M + 1), hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, P))
        self.hidden_dim = hidden * (M + 1)

    def embed(self, c, n, r):
        hc, hn = self.enc(c), self.enc(n)
        logl = self.log_l + self.l_mod(hc)
        if self.use_comp:
            with torch.no_grad():
                z = (c.float() - self.feat_mu) @ self.pca_W
                cid = torch.cdist(z.unsqueeze(1), self.cent.unsqueeze(0).expand(z.shape[0], -1, -1)).squeeze(1).argmin(-1)
            logl = logl + self.l_comp(cid)
        l2 = torch.exp(2 * logl.float())
        d2 = r.float().pow(2).sum(-1)
        w = torch.exp(-d2.unsqueeze(-1) / (2 * l2.unsqueeze(1)))
        agg = torch.einsum('bkm,bkh->bmh', w.to(hn.dtype), hn) / (w.sum(1).unsqueeze(-1).to(hn.dtype) + 1e-6)
        return torch.cat([hc, agg.flatten(1)], -1)

    def forward(self, c, n, r):
        return self.head(self.embed(c, n, r))


class WithGauss(nn.Module):
    """Wrap a backbone exposing embed(c,n,r) -> h with a low-rank Gaussian head; forward returns the mean and
    caches (F, d) for the NLL term."""
    def __init__(self, backbone, hidden, P):
        super().__init__(); self.bb = backbone; self.gh = GaussianHead(hidden, P, rank=4); self._last = None
    def forward(self, c, n, r):
        mu, Fm, d = self.gh(self.bb.embed(c, n, r)); self._last = (Fm, d); return mu


class BaseEmbed(nn.Module):
    """SPARC-Map OriginalSpatialTransformer exposing embed() (head removed for the Gaussian variant)."""
    def __init__(self, P):
        super().__init__(); self.m = L.build_model(P)
    def embed(self, c, n, r):
        m = self.m
        tok = torch.cat([m.input_proj(c).unsqueeze(1), m.input_proj(n) + m.coord_encoder(r)], 1)
        for blk in m.transformer_blocks: tok = blk(tok)
        return tok[:, 0]
    def forward(self, c, n, r): return self.m(c, n, r)


def bf16(model, c, n, r):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(c, n, r).float()


def make_model(P, train_slides):
    comp = fit_compartments(train_slides, k=6, seed=args.seed) if args.model in ('concert', 'gpk16', 'gpk48') else None
    if args.model == 'base': bb, hid = BaseEmbed(P), 256
    elif args.model == 'mlp': bb, hid = CentreMLP(1536, P), None
    elif args.model == 'concert': bb, hid = ConcertTransformer(1536, P, comp=comp), 256
    elif args.model in ('gpk16', 'gpk48'): bb = GPKernelAgg(1536, P, comp=comp); hid = bb.hidden_dim
    else: raise ValueError(args.model)
    if args.head == 'gauss':
        assert hid is not None, 'gauss head needs a backbone with embed()'
        return WithGauss(bb, hid, P)
    return bb


def gauss_extra(model, p, y, m):
    Fm, d = model._last
    return GaussianHead.nll(p, Fm.float(), d.float(), y, m)


@torch.no_grad()
def gauss_calib(model, slide, norm, bs=4096):
    """Per-slide 90% marginal coverage and mean marginal CRPS (normalised units)."""
    model.eval(); med, iqr = norm
    X = torch.tensor(slide['X'], device='cuda'); nn_ = torch.tensor(slide['nn'], device='cuda'); rel = torch.tensor(slide['rel'], device='cuda')
    Y = torch.tensor(np.nan_to_num((slide['Y'] - med) / iqr), device='cuda'); M = torch.tensor(~np.isnan(slide['Y']), device='cuda').float()
    cov = crps = cnt = 0.
    for s in range(0, len(X), bs):
        i = torch.arange(s, min(s + bs, len(X)), device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            mu = model(X[i], X[nn_[i]], rel[i]).float()
        Fm, d = model._last; sd = (Fm.float().pow(2).sum(-1) + d.float()).sqrt()
        z = (Y[i] - mu) / sd; m = M[i]
        cov += ((z.abs() <= 1.6449).float() * m).sum().item()
        nd = torch.distributions.Normal(0, 1)
        cr = sd * (z * (2 * nd.cdf(z) - 1) + 2 * torch.exp(nd.log_prob(z)) - 1 / math.sqrt(math.pi))
        crps += (cr * m).sum().item(); cnt += m.sum().item()
    return cov / cnt, crps / cnt


folds = [int(f) for f in args.folds.split(',')]
out = []
for k in folds:
    te = crc[crc.cv_fold == k].id.tolist(); tr_crc = crc[crc.cv_fold != k]
    grs = tr_crc.groupby('group').spots_under_tissue.sum().sort_values(); vg = grs.index[len(grs) // 2]
    va = [s for s in tr_crc[tr_crc.group == vg].id if s in S]
    tr = pool_ids + [s for s in tr_crc[tr_crc.group != vg].id if s in S]
    te = [s for s in te if s in S]
    t = time.time()
    trs = [S[s] for s in tr]; P = trs[0]['Y'].shape[1]
    model = make_model(P, trs)
    ep = 1 if args.smoke else args.epochs
    m, norm, hist = L.train(trs, [S[s] for s in va], epochs=ep, bs=1024, lr=2e-4, seed=100 * args.seed + k, model=model,
                            forward=bf16, loss_extra=gauss_extra if args.head == 'gauss' else None,
                            steps_per_epoch=20 if args.smoke else None)
    r = L.evaluate(m, [S[s] for s in te], norm, forward=bf16)
    if args.head == 'gauss':
        cal = {s: gauss_calib(m, S[s], norm) for s in te}
        r['cov90'] = r['sample'].map(lambda s, cal=cal: cal[s][0]); r['crps'] = r['sample'].map(lambda s, cal=cal: cal[s][1])
    r['fold'] = k; r['target'] = args.target; r['model'] = args.model; r['head'] = args.head; r['seed'] = args.seed
    r['n_train_slides'] = len(tr); r['crc_only'] = args.crc_only; r['best_epoch'] = int(np.argmax(hist)) if hist else -1
    out.append(r)
    print(tag, 'fold', k, 'val', vg, f'{time.time()-t:.0f}s', f'ntrain={len(tr)}',
          r[['pearson', 'pearson_centered', 'pearson_depthres']].mean().round(3).to_dict(),
          f'mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB', flush=True)
    del m, model; torch.cuda.empty_cache()
    pd.concat(out).to_csv(f'{OUT}/{tag}.csv.tmp', index=False); os.replace(f'{OUT}/{tag}.csv.tmp', f'{OUT}/{tag}.csv')
print('DONE', tag, f'{time.time()-t0:.0f}s')
