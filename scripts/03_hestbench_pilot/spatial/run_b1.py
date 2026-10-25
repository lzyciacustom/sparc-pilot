"""B1 spatial-context ablation runner. Usage: python run_b1.py <worker_id> <n_workers> [bench]"""
import os, sys, time, pickle
os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.append(os.getcwd())
import numpy as np, torch
torch.set_num_threads(2)
import sparcmap_lib as L, spatial_models as SM

VARIANTS = {
    'mlp':          (16, 'mlp', {}),
    'base_k16':     (16, 'base', {}),
    'concert_k16':  (16, 'concert', dict(use_dist=True, use_cos=True, use_comp=True)),
    'noDist_k16':   (16, 'concert', dict(use_dist=False, use_cos=True, use_comp=True)),
    'noCos_k16':    (16, 'concert', dict(use_dist=True, use_cos=False, use_comp=True)),
    'noComp_k16':   (16, 'concert', dict(use_dist=True, use_cos=True, use_comp=False)),
    'base_k48':     (48, 'base', {}),
    'concert_k48':  (48, 'concert', dict(use_dist=True, use_cos=True, use_comp=True)),
}
METHODS = ['ssgsea', 'mlm']; SEEDS = [0, 1]; EPOCHS = 30; BS, LR = 1024, 2e-4  # bs1024/lr2e-4 instead of bs256/lr1e-4 (GPU contention)
OUT = 'b1_runs'; os.makedirs(OUT, exist_ok=True); os.makedirs('comp_cache', exist_ok=True)
folds = L.wt_folds()


def split(k):
    te = folds[folds.cv_fold == k]['sample'].tolist(); tr_all = folds[folds.cv_fold != k]
    vg = tr_all.groupby('group').n_spots.sum().sort_values().index[len(tr_all.group.unique()) // 2]
    va = tr_all[tr_all.group == vg]['sample'].tolist(); tr = tr_all[tr_all.group != vg]['sample'].tolist()
    return tr, va, te, vg


def get_comp(k, S):
    f = f'comp_cache/fold{k}.pkl'
    if os.path.exists(f): return pickle.load(open(f, 'rb'))
    tr, va, te, _ = split(k)
    comp = SM.fit_compartments([S[s] for s in tr + va], k=6, seed=k)
    pickle.dump(comp, open(f + '.tmp', 'wb')); os.replace(f + '.tmp', f)
    return comp


def make(kind, kw, P, comp):
    if kind == 'mlp': return SM.CentreMLP(1536, P), SM.mlp_forward
    if kind == 'base': return L.build_model(P), SM.bf16_forward
    return SM.ConcertTransformer(1536, P, comp=comp, **kw), SM.bf16_forward


def main():
    wid, _ = int(sys.argv[1]), int(sys.argv[2]); bench = len(sys.argv) > 3
    core = ['mlp', 'base_k16', 'concert_k16']; second = ['noDist_k16', 'noCos_k16', 'noComp_k16', 'base_k48', 'concert_k48']
    jobs = [(m, v, k, s) for m in METHODS for s in SEEDS for k in range(4) for v in core]
    jobs += [(m, v, k, 0) for m in METHODS for k in range(4) for v in second]
    jobs += [(m, v, k, 1) for m in METHODS for k in range(4) for v in second]
    if bench: jobs = jobs[:1]
    os.makedirs('b1_locks', exist_ok=True)
    cache = {}
    for (m, v, k, s) in jobs:
        out = f'{OUT}/{m}__{v}__f{k}__s{s}.csv'
        if os.path.exists(out) and not bench: continue
        if not bench:
            try: os.close(os.open(f'b1_locks/{m}__{v}__f{k}__s{s}', os.O_CREAT | os.O_EXCL))
            except FileExistsError: continue
        kk, kind, kw = VARIANTS[v]
        key = (m, kk)
        if key not in cache:
            cache.clear()
            cache[key] = {sid: L.load_slide(sid, m, k=kk) for sid in folds['sample']}
        S = cache[key]
        tr, va, te, vg = split(k)
        comp = get_comp(k, S) if kind == 'concert' else None
        torch.manual_seed(100 * s + k)
        model, fwd = make(kind, kw, len(L.PROGRAMS), comp)
        t = time.time()
        model, norm, hist = L.train([S[x] for x in tr], [S[x] for x in va], epochs=2 if bench else EPOCHS,
                                    seed=100 * s + k, model=model, forward=fwd, bs=BS, lr=LR)
        r = L.evaluate(model, [S[x] for x in te], norm, forward=fwd)
        r['method'], r['variant'], r['fold'], r['seed'] = m, v, k, s
        r['best_epoch'] = int(np.argmax(hist)); r['n_params'] = sum(p.numel() for p in model.parameters())
        dt = time.time() - t
        print(wid, m, v, k, s, f'{dt:.0f}s', r[['pearson', 'pearson_depthres']].mean().round(3).to_dict(),
              f'mem {torch.cuda.max_memory_allocated()/1e9:.2f}GB', flush=True)
        if not bench:
            os.makedirs('b1_preds', exist_ok=True)
            np.savez_compressed(f'b1_preds/{m}__{v}__f{k}__s{s}.npz', **{x: L.predict(model, S[x], norm, forward=fwd).astype(np.float32) for x in te},
                                __norm_med=norm[0], __norm_iqr=norm[1])
            r.to_csv(out + '.tmp', index=False); os.replace(out + '.tmp', out)
        del model; import gc; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
