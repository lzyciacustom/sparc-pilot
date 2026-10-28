"""CRC-only SPARC-Map (same 6-fold patient-group CV as HPRC runs); saves per-spot test predictions for nucleus-adjusted evaluation."""
import os, sys, argparse, time, numpy as np, pandas as pd, torch
WS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, WS)
import sparcmap_lib as L
import encoders as ENC
ap = argparse.ArgumentParser(); ap.add_argument('--target', default='ulm'); ap.add_argument('--seed', type=int, default=0); args = ap.parse_args()
L.TGT_DIR = os.path.join(WS, 'targets_crc'); FD = ENC.feat_dir('crc', root=WS)
OUT = os.path.join(WS, 'preds_crc', f'{args.target}__s{args.seed}'); os.makedirs(OUT, exist_ok=True)
crc = pd.read_csv(os.path.join(WS, 'crc_folds.csv'))
def bf16(model, c, n, r):
    with torch.autocast('cuda', dtype=torch.bfloat16): return model(c, n, r).float()
S = {s: L.load_slide(s, args.target, k=16, feat_dir=FD) for s in crc.id}
t0 = time.time()
for k in sorted(crc.cv_fold.unique()):
    te = crc[crc.cv_fold == k].id.tolist(); tr = crc[crc.cv_fold != k]
    grs = tr.groupby('group').spots_under_tissue.sum().sort_values(); vg = grs.index[len(grs) // 2]
    val = [S[s] for s in tr[tr.group == vg].id]; train = [S[s] for s in tr[tr.group != vg].id]
    m, norm, _ = L.train(train, val, epochs=30, bs=1024, lr=2e-4, seed=100 * args.seed + int(k), forward=bf16)
    for s in te:
        p = L.predict(m, S[s], norm, forward=bf16)
        np.savez_compressed(f'{OUT}/{s}.npz', barcode=S[s]['barcode'], pred=p.astype(np.float32), Y=S[s]['Y'], depth=S[s]['depth'])
    print(args.target, args.seed, 'fold', k, f'{time.time()-t0:.0f}s', flush=True); del m; torch.cuda.empty_cache()
print('DONE', f'{time.time()-t0:.0f}s')
