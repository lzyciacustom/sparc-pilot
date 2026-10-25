"""Unified priority queue for B1 + B2 jobs (lock-file claiming, per-epoch logging). Usage: python run_queue.py <wid>"""
import os, sys, time, gc
os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.append(os.getcwd())
import numpy as np, pandas as pd, torch
torch.set_num_threads(2)
import sparcmap_lib as L, spatial_models as SM
import run_b1 as B1, run_b2 as B2

core = ['mlp', 'base_k16', 'concert_k16']; second = ['noDist_k16', 'noCos_k16', 'noComp_k16', 'base_k48', 'concert_k48']
abl = ['noDist_k16', 'noCos_k16', 'noComp_k16']; k48 = ['base_k48', 'concert_k48']
Q = []
Q += [('b1', 'ssgsea', v, k, s) for s in (0, 1) for k in range(4) for v in core]
Q += [('b1', 'mlm', v, k, 0) for k in range(4) for v in core]
Q += [('b2', 'ssgsea', h, k, 0) for k in range(4) for h in B2.HEADS]
Q += [('b2', 'mlm', h, k, 0) for k in range(4) for h in B2.HEADS]
Q += [('b1', 'mlm', v, k, 1) for k in range(4) for v in core]
Q += [('b1', m, v, k, 0) for m in ['ssgsea', 'mlm'] for k in range(4) for v in abl]
Q += [('b1', m, v, k, 0) for m in ['ssgsea', 'mlm'] for k in range(4) for v in k48]
Q += [('b2', m, h, k, 1) for m in ['ssgsea', 'mlm'] for k in range(4) for h in B2.HEADS]
Q += [('b1', m, v, k, 1) for m in ['ssgsea', 'mlm'] for k in range(4) for v in abl + k48]
os.makedirs('q_locks', exist_ok=True)
cache = {}


def slides(m, kk):
    if (m, kk) not in cache:
        cache.clear(); gc.collect()
        cache[(m, kk)] = {sid: L.load_slide(sid, m, k=kk) for sid in B1.folds['sample']}
    return cache[(m, kk)]


def run_b1(wid, m, v, k, s):
    out = f'{B1.OUT}/{m}__{v}__f{k}__s{s}.csv'
    kk, kind, kw = B1.VARIANTS[v]; S = slides(m, kk); tr, va, te, vg = B1.split(k)
    comp = B1.get_comp(k, S) if kind == 'concert' else None
    torch.manual_seed(100 * s + k); model, fwd = B1.make(kind, kw, len(L.PROGRAMS), comp)
    t = time.time()
    model, norm, hist = L.train([S[x] for x in tr], [S[x] for x in va], epochs=B1.EPOCHS, seed=100 * s + k, model=model,
                                forward=fwd, bs=B1.BS, lr=B1.LR, verbose=True)
    r = L.evaluate(model, [S[x] for x in te], norm, forward=fwd)
    r['method'], r['variant'], r['fold'], r['seed'] = m, v, k, s
    r['best_epoch'] = int(np.argmax(hist)); r['n_params'] = sum(p.numel() for p in model.parameters())
    os.makedirs('b1_preds', exist_ok=True)
    np.savez_compressed(f'b1_preds/{m}__{v}__f{k}__s{s}.npz', **{x: L.predict(model, S[x], norm, forward=fwd).astype(np.float32) for x in te},
                        __norm_med=norm[0], __norm_iqr=norm[1])
    r.to_csv(out + '.tmp', index=False); os.replace(out + '.tmp', out)
    print(wid, 'B1', m, v, k, s, f'{time.time()-t:.0f}s', r[['pearson', 'pearson_depthres']].mean().round(3).to_dict(), flush=True)
    del model


def main():
    wid = int(sys.argv[1])
    while True:
        claimed, pending_b2 = 0, False
        for job in Q:
            if run_job(wid, job): claimed = 1; break
        if claimed == 0:
            break


def run_job(wid, job):
    if job[0] != 'b1' or not job[2].endswith('k48'): return 0
    if True:
        typ, m, v, k, s = job
        if typ == 'b1':
            done = os.path.exists(f'{B1.OUT}/{m}__{v}__f{k}__s{s}.csv')
        else:
            bbv = open('b2_backbone.txt').read().strip() if os.path.exists('b2_backbone.txt') else None
            if bbv is None: return 0
            done = os.path.exists(f'{B2.OUT}/{m}__{bbv}__{v}__f{k}__s{s}.csv')
        if done: return 0
        lock = f'q_locks/{typ}__{m}__{v}__f{k}__s{s}' + ('' if typ == 'b1' else f'__{bbv}')
        if typ == 'b1' and v.endswith('k48'):
            try: os.close(os.open('q_locks/K48_SLOT2', os.O_CREAT | os.O_EXCL))
            except FileExistsError: return 0
        try: os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            if typ == 'b1' and v.endswith('k48'): os.remove('q_locks/K48_SLOT2')
            return 0
        print(wid, 'START', job, time.strftime('%H:%M'), flush=True)
        if typ == 'b1':
            try: run_b1(wid, m, v, k, s)
            finally:
                if v.endswith('k48') and os.path.exists('q_locks/K48_SLOT2'): os.remove('q_locks/K48_SLOT2')
        else: B2.run_one(wid, m, v, k, s, bbv, slides)
        gc.collect(); torch.cuda.empty_cache()
        return 1


if __name__ == '__main__':
    main()
