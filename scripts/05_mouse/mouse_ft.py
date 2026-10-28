"""Human-pretrain -> mouse fine-tune pilot. Conditions: human_only, mouse_only, human_ft (pretrained on human, fine-tuned on mouse train slides).
CV schemes: 'slide' (slides of each dataset spread over 5 folds; analogue of retraining on one's own atlas) and 'dataset' (leave-one-dataset-out)."""
import os, sys, time, copy, argparse, numpy as np, pandas as pd, torch
WS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, WS)
import sparcmap_lib as L
import encoders as ENC
ap = argparse.ArgumentParser(); ap.add_argument('--target', default='ulm'); ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--smoke', action='store_true')
args = ap.parse_args()
L.TGT_DIR = os.path.join(WS, 'targets_crc'); FD = ENC.feat_dir('crc', root=WS)
MT, MF = os.path.join(WS, 'targets_mouse'), ENC.feat_dir('mouse', root=WS)
OUT = os.path.join(WS, 'results_mouse_ft'); os.makedirs(OUT, exist_ok=True)
crc = pd.read_csv(os.path.join(WS, 'crc_folds.csv')); pool = pd.read_csv(os.path.join(WS, 'hest1k_train_pool.csv'))
mm = pd.read_csv(os.path.join(WS, 'mouse_test_slides_meta.csv'))
mm['dataset'] = mm.dataset_title.factorize()[0]
def bf16(model, c, n, r):
    with torch.autocast('cuda', dtype=torch.bfloat16): return model(c, n, r).float()
def metrics(pred, s):
    zs = lambda A: (A - np.nanmean(A, 0)) / (np.nanstd(A, 0) + 1e-9)
    Pz, Yz = zs(pred), zs(s['Y']); Pc = Pz - np.nanmean(Pz, 1, keepdims=True); Yc = Yz - np.nanmean(Yz, 1, keepdims=True)
    return pd.DataFrame(dict(sample=s['sid'], program=L.PROGRAMS, pearson=L._pearson_cols(pred, s['Y']), pearson_centered=L._pearson_cols(Pc, Yc),
        pearson_depthres=L._pearson_cols(L.residualize(pred, s['depth'][:, None]), L.residualize(s['Y'], s['depth'][:, None]))))
def evaluate(m, slides, norm): return pd.concat([metrics(L.predict(m, s, norm, forward=bf16), s) for s in slides])
t0 = time.time()
ok = lambda s: os.path.exists(f'{FD}/{s}.h5') and os.path.exists(f'{L.TGT_DIR}/{s}.parquet')
crc_ids = set(crc.id); pool_ids = [s for s in pool.id if ok(s) and s not in crc_ids]
if args.smoke: pool_ids = pool_ids[:6]
H = {s: L.load_slide(s, args.target, k=16, feat_dir=FD) for s in pool_ids + list(crc.id)}
old = L.TGT_DIR; L.TGT_DIR = MT
M = {s: L.load_slide(s, args.target, k=16, feat_dir=MF) for s in mm.id}; L.TGT_DIR = old
print('loaded human', len(H), 'mouse', len(M), f'{time.time()-t0:.0f}s', flush=True)
grs = crc.groupby('group').spots_under_tissue.sum().sort_values(); vg = grs.index[len(grs) // 2]
hval = [H[s] for s in crc[crc.group == vg].id]; htrain = [H[s] for s in pool_ids + list(crc[crc.group != vg].id)]
folds = {}
f_slide = np.zeros(len(mm), int)
for d, g in mm.groupby('dataset'):
    for j, i in enumerate(g.index): f_slide[i] = j % 5
folds['slide'] = f_slide; folds['dataset'] = mm.dataset.values
EP_H, EP_M, EP_FT = (1, 2, 2) if args.smoke else (30, 30, 10)
for seed in [int(x) for x in args.seeds.split(',')]:
    tag = f'{args.target}__s{seed}' + ('__smoke' if args.smoke else ''); rows = []
    t = time.time()
    pre, hnorm, _ = L.train(htrain, hval, epochs=EP_H, bs=1024, lr=2e-4, seed=1000 + seed, forward=bf16, steps_per_epoch=20 if args.smoke else None)
    print(tag, 'pretrained', f'{time.time()-t:.0f}s', flush=True)
    r = evaluate(pre, list(M.values()), hnorm); r['condition'] = 'human_only'; r['scheme'] = 'none'; r['fold'] = -1; rows.append(r)
    pre_state = copy.deepcopy(pre.state_dict()); del pre; torch.cuda.empty_cache()
    for scheme, fa in folds.items():
        for k in np.unique(fa):
            te = [M[s] for s in mm.id[fa == k]]; tr = [M[s] for s in mm.id[fa != k]]
            mo, mn, _ = L.train(tr, None, epochs=EP_M, bs=1024, lr=2e-4, seed=2000 + 10 * seed + int(k), forward=bf16, steps_per_epoch=20 if args.smoke else None)
            r = evaluate(mo, te, mn); r['condition'] = 'mouse_only'; r['scheme'] = scheme; r['fold'] = int(k); rows.append(r); del mo
            ft = L.build_model(len(L.PROGRAMS)).cuda(); ft.load_state_dict(pre_state)
            ft, fn, _ = L.train(tr, None, epochs=EP_FT, bs=1024, lr=1e-4, seed=3000 + 10 * seed + int(k), model=ft, forward=bf16, steps_per_epoch=20 if args.smoke else None)
            r = evaluate(ft, te, fn); r['condition'] = 'human_ft'; r['scheme'] = scheme; r['fold'] = int(k); rows.append(r); del ft; torch.cuda.empty_cache()
        print(tag, scheme, 'done', f'{time.time()-t:.0f}s', flush=True)
    R = pd.concat(rows).assign(seed=seed, target=args.target); R.to_csv(f'{OUT}/{tag}.csv', index=False)
    print(R.groupby(['condition', 'scheme'])[['pearson', 'pearson_depthres']].mean().round(3).to_string(), flush=True)
print('DONE', f'{time.time()-t0:.0f}s')
