"""4-fold patient-group CV of baseline SPARC-Map on a chosen target type.
usage: python run_cv.py <target> <seed>
target in: ssgsea | ulm | mlm | ssgsea_depthres | rand{d}_ssgsea | rand{d}_mlm
Inner validation group chosen exactly as in run_baseline.py; training seed = 100*seed + fold
(seed 0 reproduces baseline seeds, which used seed=fold)."""
import os, sys; sys.path.append(os.getcwd())
import sparcmap_lib as L, pandas as pd, numpy as np, torch, time
tgt, seed = sys.argv[1], int(sys.argv[2])
folds = L.wt_folds(); out = []
def load(sid):
    s = L.load_slide(sid, 'ssgsea')
    if tgt in ('ssgsea', 'ulm', 'mlm'):
        if tgt != 'ssgsea': s['Y'] = L.load_slide(sid, tgt)['Y']
    elif tgt == 'ssgsea_depthres':
        s['Y'] = L.residualize(s['Y'].astype(np.float64), s['depth'][:, None]).astype(np.float32)
    elif tgt.startswith('rand'):
        d, m = tgt[4:].split('_')
        t = pd.read_parquet(f'targets_rand/{sid}.parquet').set_index('barcode').reindex(s['barcode'])
        s['Y'] = t[[f'rand{d}_{m}__{p}' for p in L.PROGRAMS]].values.astype(np.float32)
    else: raise ValueError(tgt)
    return s
todo = [k for k in range(4) if not os.path.exists(f'results/folds/cv_{tgt}_s{seed}_f{k}.csv') and not os.path.exists(f'results/folds/cv_{tgt}_s{seed}_f{k}.csv.lock')]
if not todo and not all(os.path.exists(f'results/folds/cv_{tgt}_s{seed}_f{k}.csv') for k in range(4)): print('all folds taken', tgt); sys.exit(0)
S = {s: load(s) for s in folds['sample']}
for k in sorted(folds.cv_fold.unique()):
    te = folds[folds.cv_fold == k]['sample'].tolist(); tr_all = folds[folds.cv_fold != k]
    vg = tr_all.groupby('group').n_spots.sum().sort_values().index[len(tr_all.group.unique()) // 2]
    va = tr_all[tr_all.group == vg]['sample'].tolist(); tr = tr_all[tr_all.group != vg]['sample'].tolist()
    ff = f'results/folds/cv_{tgt}_s{seed}_f{k}.csv'
    if os.path.exists(ff): out.append(pd.read_csv(ff)); continue
    try: os.close(os.open(ff + '.lock', os.O_CREAT | os.O_EXCL))
    except FileExistsError: print('fold locked by another stream', tgt, seed, k, flush=True); continue
    t = time.time()
    m, norm, h = L.train([S[s] for s in tr], [S[s] for s in va], epochs=30, seed=100 * seed + k, verbose=True)
    r = L.evaluate(m, [S[s] for s in te], norm); r['fold'] = k; out.append(r); r.to_csv(ff, index=False)
    print(tgt, seed, k, vg, f'{time.time()-t:.0f}s', f'maxres={torch.cuda.max_memory_reserved()/1e9:.2f}GB', r[['pearson', 'pearson_centered', 'pearson_depthres']].mean().round(3).to_dict(), flush=True)
    del m; torch.cuda.empty_cache()
if len(out) < 4: print('incomplete; folds done here', len(out)); sys.exit(0)
res = pd.concat(out).merge(folds[['sample', 'task', 'group']], on='sample'); res['target'] = tgt; res['seed'] = seed
res.to_csv(f'results/cv_{tgt}_s{seed}.csv', index=False)
print('ALL', tgt, seed, res[['pearson', 'pearson_centered', 'pearson_depthres']].mean().round(3).to_dict())
