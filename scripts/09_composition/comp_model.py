"""Composition-aware SPARC-Map variants, CRC-only training, 6 patient-group folds, H-optimus-1 features.

Targets per spot: y = ULM program scores (36); pi = NNLS transcript-fraction composition (12 types; GSE132465);
b_M = reference-derived composition-and-depth baseline (comp/build_bM.py), affinely calibrated per program on training spots:
    b(pi, d) = a_p + c_p * b_M,p(pi, d);   reference-relative residual  delta = y - b(pi_deconv, d_spot)  (composition and depth removed).
H&E-side predictions use the fold's median training depth d_ref:  y_hat = b(pi_hat, d_ref) + delta_hat.
Modes (shared SPARC-Map backbone = OriginalSpatialTransformer token 0):
  direct : head -> y                                        (baseline, MSE + 0.5 Pearson)
  comp   : softmax head -> pi_hat, soft cross-entropy to pi  (composition-only model; y_hat = b(pi_hat))
  delta  : head -> delta_hat, MSE + 0.5 Pearson on delta     (stage 2 of the two-stage model)
  joint  : y_hat = b(pi_hat, d_ref) + delta_hat;  loss = L(y_hat, y) + alpha * CE(pi, pi_hat) + beta * L(delta_hat, delta)
Per-spot test predictions -> comp/preds/<tag>/<sid>.npz ; calibration per fold -> comp/preds/<tag>/calib_f<k>.npz
"""
import os, sys, argparse, time, json, numpy as np, pandas as pd, torch
from torch import nn
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, WS)
import sparcmap_lib as L
from decon_resid import TYPES

ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True, choices=['direct', 'comp', 'delta', 'joint'])
ap.add_argument('--target', default='ulm', choices=['ulm', 'rand0ulm'])
ap.add_argument('--decon', default='out_parquet', help='comp/<dir> with comp__<type> columns (out_parquet | out_disjoint)')
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--alpha', type=float, default=1.0); ap.add_argument('--beta', type=float, default=1.0)
ap.add_argument('--folds', default='0,1,2,3,4,5'); ap.add_argument('--smoke', action='store_true')
ap.add_argument('--feat_dir', default=f'{WS}/features/hoptimus1_crc')
args = ap.parse_args()
L.TGT_DIR = f'{WS}/targets_crc'
tag = f'{args.mode}__{args.target}__{args.decon}__s{args.seed}' + (f'__a{args.alpha:g}b{args.beta:g}' if args.mode == 'joint' else '') + ('__smoke' if args.smoke else '')
PD = f'{WS}/comp/preds/{tag}'; os.makedirs(PD, exist_ok=True)
K = len(TYPES); P = len(L.PROGRAMS)

bm = torch.load(f'{WS}/comp/bM_{args.target}.pt')
bM = nn.Sequential(nn.Linear(K + 1, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, P))
bM.load_state_dict(bm['state']); bM = bM.cuda().eval(); [q.requires_grad_(False) for q in bM.parameters()]
bmu, bsd = bm['mu'].cuda(), bm['sd'].cuda()
def bM_f(pi, d):
    z = ((d.float() - bm['dmu']) / bm['dsd']).clamp((bm['dmin'] - bm['dmu']) / bm['dsd'], (bm['dmax'] - bm['dmu']) / bm['dsd'])
    return bM(torch.cat([pi.float(), z.reshape(-1, 1).expand(len(pi), 1)], -1)) * bsd + bmu

crc = pd.read_csv(f'{WS}/crc_folds.csv')
S = {}
for sid in crc.id:
    s = L.load_slide(sid, args.target, k=16, feat_dir=args.feat_dir)
    c = pd.read_parquet(f'{WS}/comp/{args.decon}/{sid}.parquet').set_index('barcode').reindex(s['barcode'])
    s['pi'] = c[[f'comp__{t}' for t in TYPES]].values.astype(np.float32)
    S[sid] = s
with torch.no_grad():
    for s in S.values():
        pi = np.nan_to_num(s['pi']); s['bM'] = bM_f(torch.tensor(pi, device='cuda'), torch.tensor(s['depth'], device='cuda')).cpu().numpy()
        s['bM'][~np.isfinite(s['pi']).all(1)] = np.nan
print('loaded', len(S), flush=True)


def calibrate(slides):
    """Per-program a + c * b_M on pooled training spots."""
    Y = np.concatenate([s['Y'] for s in slides]); B = np.concatenate([s['bM'] for s in slides]); ab = np.zeros((P, 2))
    for j in range(P):
        ok = np.isfinite(Y[:, j]) & np.isfinite(B[:, j])
        ab[j] = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), B[ok, j]]), Y[ok, j], rcond=None)[0]
    return ab


class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.m = L.build_model(P)
        h = self.m.hidden_dim
        self.pi_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(h // 2, K))
    def forward(self, c, n, r):
        m = self.m
        tok = torch.cat([m.input_proj(c).unsqueeze(1), m.input_proj(n) + m.coord_encoder(r)], 1)
        for blk in m.transformer_blocks: tok = blk(tok)
        z = tok[:, 0]
        return m.head(z), self.pi_head(z)


def pearson_l(p, y, msk): return L.pearson_loss(p, y, msk)
def mse(p, y, msk): return ((p - y) ** 2 * msk).sum() / msk.sum().clamp(min=1)


class Batcher:
    def __init__(self, slides, med, iqr, ab):
        self.s = []
        for s in slides:
            Yn = (s['Y'] - med) / iqr; D = (s['Y'] - (ab[:, 0] + ab[:, 1] * s['bM'])) / iqr
            T = lambda a: torch.tensor(np.nan_to_num(a), device='cuda', dtype=torch.float32)
            pi_ok = np.isfinite(s['pi']).all(1)
            self.s.append(dict(X=T(s['X']), nn=torch.tensor(s['nn'], device='cuda'), rel=T(s['rel']), Y=T(Yn), MY=T(np.isfinite(Yn)),
                               D=T(D), MD=T(np.isfinite(D)), PI=T(s['pi']), MPI=T(pi_ok)))
        self.sizes = np.array([len(s['X']) for s in slides])
    def get(self, si, idx):
        s = self.s[si]; idx = torch.as_tensor(idx, device='cuda')
        return [s['X'][idx], s['X'][s['nn'][idx]], s['rel'][idx]] + [s[k][idx] for k in ['Y', 'MY', 'D', 'MD', 'PI', 'MPI']]
    def random(self, bs, rng):
        si = rng.choice(len(self.s), p=self.sizes / self.sizes.sum()); return self.get(si, rng.integers(0, self.sizes[si], bs))


D_REF = None


def outputs(net, c, n, r, med_t, iqr_t, ab_t):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h, lg = net(c, n, r)
    h, lg = h.float(), lg.float(); pi_hat = torch.softmax(lg, -1)
    if args.mode == 'joint':
        yraw = ab_t[:, 0] + ab_t[:, 1] * bM_f(pi_hat, D_REF) + iqr_t * h
        return (yraw - med_t) / iqr_t, h, lg, pi_hat
    return h, h, lg, pi_hat


def loss_fn(o, b):
    yhat, dhat, lg, _ = o; _, _, _, Y, MY, D, MD, PI, MPI = b
    ce = -(PI * torch.log_softmax(lg, -1)).sum(-1); ce = (ce * MPI).sum() / MPI.sum().clamp(min=1)
    if args.mode == 'direct': return mse(yhat, Y, MY) + 0.5 * pearson_l(yhat, Y, MY)
    if args.mode == 'comp': return ce
    if args.mode == 'delta': return mse(dhat, D, MD) + 0.5 * pearson_l(dhat, D, MD)
    return (mse(yhat, Y, MY) + 0.5 * pearson_l(yhat, Y, MY) + args.alpha * ce
            + args.beta * (mse(dhat, D, MD) + 0.5 * pearson_l(dhat, D, MD)))


@torch.no_grad()
def predict(net, s, med_t, iqr_t, ab_t, bs=4096):
    net.eval(); X = torch.tensor(s['X'], device='cuda'); nn_ = torch.tensor(s['nn'], device='cuda'); rel = torch.tensor(s['rel'], device='cuda')
    outs = {'y': [], 'd': [], 'pi': []}
    for b in range(0, len(X), bs):
        i = torch.arange(b, min(b + bs, len(X)), device='cuda'); yh, dh, lg, ph = outputs(net, X[i], X[nn_[i]], rel[i], med_t, iqr_t, ab_t)
        outs['y'].append((yh * iqr_t + med_t).cpu().numpy()); outs['d'].append((dh * iqr_t).cpu().numpy()); outs['pi'].append(ph.cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def val_score(net, slides, med_t, iqr_t, ab_t, ab):
    sc = []
    for s in slides:
        o = predict(net, s, med_t, iqr_t, ab_t)
        if args.mode == 'comp':
            ok = np.isfinite(s['pi']).all(1); sc.append(-(-(s['pi'][ok] * np.log(o['pi'][ok] + 1e-8)).sum(1).mean()))
        elif args.mode == 'delta':
            sc.append(np.nanmean(L._pearson_cols(o['d'], s['Y'] - (ab[:, 0] + ab[:, 1] * s['bM']))))
        else:
            sc.append(np.nanmean(L._pearson_cols(o['y'], s['Y'])))
    return float(np.mean(sc))


t0 = time.time()
for k in [int(f) for f in args.folds.split(',')]:
    te = crc[crc.cv_fold == k].id.tolist(); tr_crc = crc[crc.cv_fold != k]
    grs = tr_crc.groupby('group').spots_under_tissue.sum().sort_values(); vg = grs.index[len(grs) // 2]
    va = tr_crc[tr_crc.group == vg].id.tolist(); tr = tr_crc[tr_crc.group != vg].id.tolist()
    trs = [S[s] for s in tr]; med, iqr = L.robust_stats([s['Y'] for s in trs]); ab = calibrate(trs)
    med_t, iqr_t, ab_t = [torch.tensor(a, device='cuda', dtype=torch.float32) for a in (med, iqr, ab)]
    D_REF = torch.tensor(float(np.median(np.concatenate([s['depth'] for s in trs]))), device='cuda')
    np.savez(f'{PD}/calib_f{k}.npz', med=med, iqr=iqr, ab=ab, d_ref=float(D_REF))
    torch.manual_seed(100 * args.seed + k); rng = np.random.default_rng(100 * args.seed + k)
    net = Net().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-2)
    tb = Batcher(trs, med, iqr, ab); spe = 20 if args.smoke else max(50, int(tb.sizes.sum() / 1024))
    best, state = -1e9, None
    for ep in range(1 if args.smoke else args.epochs):
        net.train()
        for _ in range(spe):
            b = tb.random(1024, rng); o = outputs(net, b[0], b[1], b[2], med_t, iqr_t, ab_t); loss = loss_fn(o, b)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        v = val_score(net, [S[s] for s in va], med_t, iqr_t, ab_t, ab)
        if v > best: best, state, be = v, {kk: vv.detach().clone() for kk, vv in net.state_dict().items()}, ep
    net.load_state_dict(state)
    for s in te:
        o = predict(net, S[s], med_t, iqr_t, ab_t)
        np.savez_compressed(f'{PD}/{s}.npz', y=o['y'].astype(np.float32), d=o['d'].astype(np.float32), pi=o['pi'].astype(np.float32),
                            barcode=S[s]['barcode'], fold=k)
    print(tag, 'fold', k, 'val', vg, f'best_ep={be} val={best:.3f}', f'{time.time()-t0:.0f}s', flush=True)
    del net, tb; torch.cuda.empty_cache()
print('DONE', tag, f'{time.time()-t0:.0f}s')
