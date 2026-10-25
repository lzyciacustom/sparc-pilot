"""Wave-2 driver (extends hest1k_run.py; base model = SPARC-Map distance-only transformer, bs1024/lr2e-4/bf16/30 ep).
Options:
  --target_norm robust|slide   slide: z-score every program within each slide (removes between-slide offsets)
  --tech                       append platform one-hot (ffpe_probe / ff_wt / visium_hd) to every token's features
  --progw                      weight programs by split-half reliability (ULM, depth-residualised, HEST-Bench Track A)
  --ctx                        concatenate multi-scale context features (448 um @ 2 um/px) -> input 3072
  --genes                      predict a 4,000-gene panel (low-rank decoder), programs read out by ULM on predicted genes
  --gene_aux                   36-program head + 4,000-gene auxiliary head (gene loss weight 0.5 of program loss)
  --decoup comp|res|randres    CRC-only: predict cell-type composition / composition-adjusted ULM / adjusted random ULM
  --external mouse             train on all human slides (pool + CRC), test on 27 mouse slides (ortholog-mapped targets)
Output: results2/<tag>.csv (per test slide x program metrics).
"""
import os, sys, time, argparse, json, numpy as np, pandas as pd, torch, h5py
from torch import nn
WS = os.path.dirname(os.path.abspath(__file__)); sys.path.append(WS); sys.path.append(os.path.join(WS, 'trackB'))
import sparcmap_lib as L

ap = argparse.ArgumentParser()
ap.add_argument('--target', default='ulm'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--folds', default='0,1,2,3,4,5')
ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--smoke', action='store_true')
ap.add_argument('--target_norm', default='robust'); ap.add_argument('--tech', action='store_true'); ap.add_argument('--progw', action='store_true')
ap.add_argument('--ctx', action='store_true'); ap.add_argument('--genes', action='store_true'); ap.add_argument('--gene_aux', action='store_true')
ap.add_argument('--decoup', default=''); ap.add_argument('--crc_only', action='store_true'); ap.add_argument('--external', default='')
args = ap.parse_args()
FD = os.path.join(WS, 'features', 'hoptimus0_crc'); CTX = os.path.join(WS, 'features', 'hoptimus0_ctx')
L.TGT_DIR = os.path.join(WS, 'targets_crc')
OUT = os.path.join(WS, 'results2'); os.makedirs(OUT, exist_ok=True)
flags = [f'tn-{args.target_norm}'] + [k for k in ['tech', 'progw', 'ctx', 'genes', 'gene_aux', 'crc_only'] if getattr(args, k)]
if args.decoup: flags.append(f'decoup-{args.decoup}')
if args.external: flags.append(f'ext-{args.external}')
tag = f'{args.target}__' + '+'.join(flags) + f'__s{args.seed}' + ('' if args.folds == '0,1,2,3,4,5' else '__f' + args.folds.replace(',', '-')) + ('__smoke' if args.smoke else '')
PROGS = L.PROGRAMS

crc = pd.read_csv(os.path.join(WS, 'crc_folds.csv'))
pool = pd.read_csv(os.path.join(WS, 'hest1k_train_pool.csv'))
crc_ids = set(crc.id)
meta = pd.concat([pool[['id', 'st_technology', 'preservation_method']],
                  pd.read_csv(os.path.join(WS, 'hest1k_crc_candidates.csv')).rename(columns={})[['id', 'st_technology', 'preservation_method']]]).drop_duplicates('id').set_index('id')
if args.external == 'mouse':
    mm = pd.read_csv(os.path.join(WS, 'mouse_test_slides_meta.csv')).set_index('id'); meta = pd.concat([meta, mm[['st_technology', 'preservation_method']]])


def tech_onehot(sid):
    r = meta.loc[sid] if sid in meta.index else None
    v = np.zeros(3, np.float32)
    if r is None: return v
    if 'HD' in str(r.st_technology): v[2] = 1
    elif 'FFPE' in str(r.preservation_method): v[0] = 1
    else: v[1] = 1
    return v


panel = json.load(open(os.path.join(WS, 'gene_panel.json')))['panel'] if (args.genes or args.gene_aux) else None
relw = None
if args.progw:
    sh = pd.read_csv(os.path.join(WS, 'A2_splithalf_per_program.csv'))
    sh = sh[sh.method == 'ulm'].set_index('program')['r_depthres'].reindex(PROGS)
    relw = np.clip(sh.fillna(sh.median()).values, 0.05, None); relw = (relw / relw.mean()).astype(np.float32)


def load(sid, tgt_dir=None, feat_dir=FD, ctx_dir=CTX, genes_dir=os.path.join(WS, 'genes_crc')):
    old = L.TGT_DIR
    if tgt_dir: L.TGT_DIR = tgt_dir
    m = 'ssgsea' if args.target == 'ssgsea_depthres' else args.target
    if args.decoup == 'randres': m = 'rand0ulm'
    s = L.load_slide(sid, m, k=16, feat_dir=feat_dir); L.TGT_DIR = old
    if args.target == 'ssgsea_depthres': s['Y'] = L.residualize(s['Y'], s['depth'][:, None]).astype(np.float32)
    if args.decoup:
        dc = pd.read_parquet(os.path.join(WS, 'decoup', f'{sid}.parquet')).set_index('barcode')
        comp = dc.reindex(s['barcode'])[[c for c in dc.columns if c.startswith('comp__')]].values.astype(np.float32)
        s['comp_names'] = [c for c in dc.columns if c.startswith('comp__')]
        if args.decoup == 'comp': s['Y'] = comp
        else:
            ok = np.isfinite(comp).all(1); Y = np.full_like(s['Y'], np.nan)
            Y[ok] = L.residualize(s['Y'][ok], comp[ok][:, :-1]); s['Y'] = Y.astype(np.float32)
    if args.target_norm == 'slide':
        Y = s['Y']; s['Y'] = ((Y - np.nanmean(Y, 0)) / (np.nanstd(Y, 0) + 1e-6)).astype(np.float32)
    if args.ctx:
        with h5py.File(os.path.join(ctx_dir, f'{sid}.h5')) as f:
            cb = np.array([b[0].decode() if isinstance(b[0], bytes) else str(b[0]) for b in f['barcode'][:]]); cX = f['features'][:].astype(np.float32)
        s['X'] = np.concatenate([s['X'], cX[pd.Index(cb).get_indexer(s['barcode'])]], 1)
    if args.tech:
        s['X'] = np.concatenate([s['X'], np.tile(tech_onehot(sid), (len(s['X']), 1))], 1)
    if panel is not None and not args.external:
        with h5py.File(os.path.join(genes_dir, f'{sid}.h5')) as f:
            gb = f['barcode'][:].astype(str); G = f['X'][:]
        s['G'] = G[pd.Index(gb).get_indexer(s['barcode'])]
        s['Yprog'] = s['Y']
        s['Y'] = s['G'] if args.genes else np.concatenate([s['Y'], s['G'].astype(np.float32)], 1)
    return s


if args.decoup:
    have = {f[:-8] for f in os.listdir(os.path.join(WS, 'decoup'))}
    crc = crc[crc.id.isin(have)]
ok = lambda s: os.path.exists(f'{FD}/{s}.h5') and os.path.exists(f'{L.TGT_DIR}/{s}.parquet') and (not args.ctx or os.path.exists(f'{CTX}/{s}.h5'))
crc = crc[crc.id.map(ok)]
pool_ids = [] if (args.decoup or args.crc_only) else [s for s in pool.id if ok(s) and s not in crc_ids]
if args.smoke: pool_ids = pool_ids[:6]
t0 = time.time(); S = {}
for s in pool_ids + list(crc.id):
    try:
        x = load(s)
        if np.isfinite(x.get('Yprog', x['Y'])).any(axis=0).sum() >= min(30, x.get('Yprog', x['Y']).shape[1]): S[s] = x
    except Exception as e: print('skip', s, repr(e)[:100], flush=True)
pool_ids = [s for s in pool_ids if s in S]
EXT = {}
if args.external == 'mouse':
    for s in pd.read_csv(os.path.join(WS, 'mouse_test_slides_meta.csv')).id:
        try: EXT[s] = load(s, tgt_dir=os.path.join(WS, 'targets_mouse'), feat_dir=os.path.join(WS, 'features', 'hoptimus0_mouse'))
        except Exception as e: print('skip ext', s, repr(e)[:100], flush=True)
print(f'loaded {len(S)} slides ({len(pool_ids)} pool) ext={len(EXT)} in {time.time()-t0:.0f}s', flush=True)
D_IN = next(iter(S.values()))['X'].shape[1]; P_OUT = next(iter(S.values()))['Y'].shape[1]


class BaseEmbed(nn.Module):
    def __init__(self, P, din):
        super().__init__(); self.m = L.build_model(P, input_dim=din)
    def embed(self, c, n, r):
        m = self.m; tok = torch.cat([m.input_proj(c).unsqueeze(1), m.input_proj(n) + m.coord_encoder(r)], 1)
        for blk in m.transformer_blocks: tok = blk(tok)
        return tok[:, 0]
    def forward(self, c, n, r): return self.m(c, n, r)


class GeneDecoder(nn.Module):
    """Transformer embedding -> 128-d latent -> 4,000 genes (low-rank, shared gene loadings)."""
    def __init__(self, G, din, rank=128):
        super().__init__(); self.bb = BaseEmbed(1, din)
        self.lat = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, rank), nn.GELU()); self.dec = nn.Linear(rank, G)
    def forward(self, c, n, r): return self.dec(self.lat(self.bb.embed(c, n, r)))


def bf16(model, c, n, r):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(c, n, r).float()


class F16Batcher(L.SlideBatcher):
    """Memory-light batcher for wide (gene) targets: Y kept fp16 with NaNs; mask computed per batch."""
    def __init__(self, slides, med, iqr, device='cuda'):
        self.dev = device; self.s = []
        for s in slides:
            Yn = ((s['Y'].astype(np.float32) - med) / iqr).astype(np.float16)
            self.s.append(dict(X=torch.tensor(s['X'], device=device, dtype=torch.float16), Y=torch.tensor(Yn, device=device),
                               nn=torch.tensor(s['nn'], device=device), rel=torch.tensor(s['rel'], device=device)))
        self.sizes = np.array([len(s['X']) for s in slides])
    def get(self, si, idx):
        s = self.s[si]; idx = torch.as_tensor(idx, device=self.dev)
        Y = s['Y'][idx].float(); M = torch.isfinite(Y).float()
        return s['X'][idx].float(), s['X'][s['nn'][idx]].float(), s['rel'][idx], torch.nan_to_num(Y), M


@torch.no_grad()
def predict_raw(model, slide, norm, bs=2048):
    model.eval(); med, iqr = norm
    X = torch.tensor(slide['X'], device='cuda'); nn_ = torch.tensor(slide['nn'], device='cuda'); rel = torch.tensor(slide['rel'], device='cuda')
    out = []
    for s in range(0, len(X), bs):
        i = torch.arange(s, min(s + bs, len(X)), device='cuda'); out.append(bf16(model, X[i], X[nn_[i]], rel[i]).cpu().numpy())
    return np.concatenate(out) * iqr + med


def ulm_readout(E):
    """decoupler-style ULM t-values of each program on predicted log-expression E (N x G) over the gene panel."""
    progs = json.load(open(os.path.join(WS, 'sparc_programs_msigdb.json')))['sets']
    gi = {g: i for i, g in enumerate(panel)}
    Mb = np.zeros((len(panel), len(PROGS)), np.float32)
    for j, p in enumerate(PROGS):
        for g in progs[p]:
            if g in gi: Mb[gi[g], j] = 1
    G = E.shape[1]; xm = Mb.mean(0); Mc = Mb - xm; vm = (Mc ** 2).sum(0)
    Ec = E - E.mean(1, keepdims=True); C = Ec @ Mc; b = C / vm
    sse = (Ec ** 2).sum(1, keepdims=True) - b ** 2 * vm
    return b / np.sqrt(np.clip(sse, 1e-8, None) / (G - 2) / vm)


def metrics(pred, s, names):
    Y = s.get('Yprog', s['Y']) if pred.shape[1] == len(PROGS) else s['Y']
    zs = lambda A: (A - np.nanmean(A, 0)) / (np.nanstd(A, 0) + 1e-9)
    Pz, Yz = zs(pred), zs(Y); Pc = Pz - np.nanmean(Pz, 1, keepdims=True); Yc = Yz - np.nanmean(Yz, 1, keepdims=True)
    r_raw = L._pearson_cols(pred, Y); r_c = L._pearson_cols(Pc, Yc)
    r_d = L._pearson_cols(L.residualize(pred, s['depth'][:, None]), L.residualize(Y, s['depth'][:, None]))
    return pd.DataFrame(dict(sample=s['sid'], program=names, pearson=r_raw, pearson_centered=r_c, pearson_depthres=r_d))


def evaluate(model, slides, norm):
    rows = []
    for s in slides:
        pr = predict_raw(model, s, norm)
        if args.genes: pr = ulm_readout(pr)
        elif args.gene_aux: pr = pr[:, :len(PROGS)]
        names = s.get('comp_names') if args.decoup == 'comp' else PROGS
        rows.append(metrics(pr, s, names))
    return pd.concat(rows)


def run(train, val, test, k):
    trs = [S[s] for s in train]
    if args.genes or args.gene_aux:
        Yall = np.concatenate([x['Y'][::5].astype(np.float32) for x in trs])
        med = np.nanmean(Yall, 0); iqr = np.nanstd(Yall, 0); iqr = np.where(iqr > 1e-6, iqr, 1.0)
        if args.gene_aux:
            pm, pi = L.robust_stats([x['Y'][:, :len(PROGS)] for x in trs]); med[:len(PROGS)] = pm; iqr[:len(PROGS)] = pi
        norm = (med.astype(np.float32), iqr.astype(np.float32)); bcls = F16Batcher
    else:
        norm = None; bcls = None
    if args.genes: model = GeneDecoder(P_OUT, D_IN)
    else: model = BaseEmbed(P_OUT, D_IN)
    pw = relw
    if args.gene_aux:
        pw = np.concatenate([np.ones(len(PROGS), np.float32) if relw is None else relw,
                             np.full(P_OUT - len(PROGS), 0.5 * len(PROGS) / (P_OUT - len(PROGS)), np.float32)])
    use_val = None if (args.genes or args.gene_aux or args.decoup == 'comp') else [S[s] for s in val]
    m, norm, hist = L.train(trs, use_val, epochs=1 if args.smoke else args.epochs, bs=1024, lr=2e-4, seed=100 * args.seed + k,
                            model=model, forward=bf16, steps_per_epoch=20 if args.smoke else None, prog_w=pw, norm=norm, batcher_cls=bcls)
    return m, norm, hist


out = []
if args.external:
    grs = crc.groupby('group').spots_under_tissue.sum().sort_values(); vg = grs.index[len(grs) // 2]
    val = [s for s in crc[crc.group == vg].id if s in S]; train = pool_ids + [s for s in crc[crc.group != vg].id if s in S]
    t = time.time(); m, norm, hist = run(train, val, None, 0)
    r = evaluate(m, list(EXT.values()), norm); r['fold'] = -1; r['n_train_slides'] = len(train); out.append(r)
    print(tag, 'external', f'{time.time()-t:.0f}s', r[['pearson', 'pearson_centered', 'pearson_depthres']].mean().round(3).to_dict(), flush=True)
else:
    for k in [int(f) for f in args.folds.split(',')]:
        te = [s for s in crc[crc.cv_fold == k].id if s in S]; tr_crc = crc[crc.cv_fold != k]
        grs = tr_crc.groupby('group').spots_under_tissue.sum().sort_values(); vg = grs.index[len(grs) // 2]
        val = [s for s in tr_crc[tr_crc.group == vg].id if s in S]; train = pool_ids + [s for s in tr_crc[tr_crc.group != vg].id if s in S]
        t = time.time(); m, norm, hist = run(train, val, te, k)
        r = evaluate(m, [S[s] for s in te], norm); r['fold'] = k; r['n_train_slides'] = len(train)
        r['best_epoch'] = int(np.nanargmax(hist)) if hist else -1; out.append(r)
        print(tag, 'fold', k, f'{time.time()-t:.0f}s', r[['pearson', 'pearson_centered', 'pearson_depthres']].mean().round(3).to_dict(),
              f'mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB', flush=True)
        del m; torch.cuda.empty_cache()
        pd.concat(out).assign(tag=tag, seed=args.seed).to_csv(f'{OUT}/{tag}.csv.tmp', index=False); os.replace(f'{OUT}/{tag}.csv.tmp', f'{OUT}/{tag}.csv')
pd.concat(out).assign(tag=tag, seed=args.seed).to_csv(f'{OUT}/{tag}.csv', index=False)
print('DONE', tag, f'{time.time()-t0:.0f}s')
