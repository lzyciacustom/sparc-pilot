import os, sys, time, json, numpy as np, pandas as pd, torch
sys.path.append(os.getcwd())
import al_core as A, sparcmap_lib as L

k = int(sys.argv[1]); method = sys.argv[2] if len(sys.argv) > 2 else 'ssgsea'
seeds = [int(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['0', '1', '2'])]
OUT = f'al_out/{method}'; os.makedirs(OUT, exist_ok=True)
N_INIT, B, ROUNDS = 2, 2, 6
STRATS = os.environ.get('AL_STRATS', 'random,maxunc,epig').split(',')
ALL_STRATS = ['random', 'maxunc', 'epig', 'featmatch']

f = L.wt_folds()
S = {s: L.load_slide(s, method) for s in f['sample']}
gs = A.GPUSlides(S)
grp = dict(zip(f['sample'], f['group'])); nsp = {s: len(S[s]['X']) for s in S}
te = f[f.cv_fold == k]['sample'].tolist(); tr = f[f.cv_fold != k]
vg = tr.groupby('group').n_spots.sum().sort_values().index[len(tr.group.unique()) // 2]
va = tr[tr.group == vg]['sample'].tolist(); pool = tr[tr.group != vg]['sample'].tolist()
print(f'fold {k} test {len(te)} val {vg} ({len(va)}) pool {len(pool)}', flush=True)


fmean = {s: S[s]['X'].mean(0) for s in S}
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def per_slide(r):
    return r.groupby('sample')[['pearson', 'pearson_depthres']].mean()


for seed in seeds:
    init_rng = np.random.default_rng(10_000 + 100 * k + seed)
    init = list(init_rng.choice(pool, N_INIT, replace=False))
    ens0 = None
    for st in STRATS:
        si = ALL_STRATS.index(st)
        fn = f'{OUT}/fold{k}_seed{seed}_{st}.csv'
        if os.path.exists(fn): print('skip', fn); continue
        sel_rng = np.random.default_rng(20_000 + 100 * k + 10 * seed + si)
        sc_rng = np.random.default_rng(30_000 + 100 * k + 10 * seed + si)
        lab = list(init); rows, prog_rows, sel_rows = [], [], []
        for rd in range(ROUNDS + 1):
            t0 = time.time()
            if rd == 0:
                if ens0 is None: ens0 = A.train_ensemble(gs, lab, va, base_seed=k * 1000 + seed * 100)
                ens = ens0
            else:
                ens = A.train_ensemble(gs, lab, va, base_seed=k * 1000 + seed * 100 + rd * 10 + si + 1)
            r = A.evaluate_ensemble(ens, gs, te); ps = per_slide(r)
            prog_rows.append(r.assign(round=rd))
            rows.append(dict(fold=k, seed=seed, strategy=st, round=rd, n_slides=len(lab),
                             n_spots=int(sum(nsp[s] for s in lab)), n_groups=len({grp[s] for s in lab}),
                             pearson=ps.pearson.mean(), pearson_depthres=ps.pearson_depthres.mean(),
                             val_best=float(np.mean([e[2] for e in ens])),
                             labelled=';'.join(lab), **{f'slide_pearson__{s}': ps.pearson[s] for s in ps.index},
                             **{f'slide_pdres__{s}': ps.pearson_depthres[s] for s in ps.index}))
            if rd < ROUNDS:
                cand = [p for p in pool if p not in lab]
                sc = A.score_candidates(ens, gs, cand, te, sc_rng)
                for c in cand: sc[c]['featsim'] = float(np.mean([cos(fmean[c], fmean[t]) for t in te]))
                chosen = A.select_batch(st, sc, cand, B, sel_rng)
                covered = {grp[s] for s in lab}
                for j, c in enumerate(chosen):
                    sel_rows.append(dict(fold=k, seed=seed, strategy=st, round=rd + 1, slide=c, group=grp[c],
                                         task=dict(zip(f['sample'], f['task']))[c], n_spots=nsp[c],
                                         group_already_covered=grp[c] in covered,
                                         same_group_as_batchmate=(j == 1 and grp[c] == grp[chosen[0]]),
                                         epig=sc[c]['epig'], maxunc=sc[c]['maxunc'],
                                         epig_rank=1 + sorted(cand, key=lambda x: -sc[x]['epig']).index(c),
                                         maxunc_rank=1 + sorted(cand, key=lambda x: -sc[x]['maxunc']).index(c),
                                         n_cand=len(cand)))
                # store all candidate scores (diagnostics: does the score predict realised gain?)
                pd.DataFrame([dict(fold=k, seed=seed, strategy=st, round=rd, slide=c, group=grp[c], epig=v['epig'],
                                   epig_meanr2=v['epig_meanr2'], maxunc=v['maxunc'], featsim=v.get('featsim', np.nan)) for c, v in sc.items()]
                             ).to_csv(f'{OUT}/scores_fold{k}_seed{seed}_{st}_r{rd}.csv', index=False)
                lab = lab + [str(c) for c in chosen]
            print(f'fold{k} seed{seed} {st} r{rd} n={rows[-1]["n_slides"]} r={rows[-1]["pearson"]:.3f} '
                  f'rd={rows[-1]["pearson_depthres"]:.3f} {time.time() - t0:.0f}s', flush=True)
            if rd > 0: del ens
            torch.cuda.empty_cache()
        pd.concat(prog_rows).assign(fold=k, seed=seed, strategy=st).to_parquet(f'{OUT}/prog_fold{k}_seed{seed}_{st}.parquet')
        pd.DataFrame(sel_rows).to_csv(f'{OUT}/sel_fold{k}_seed{seed}_{st}.csv', index=False)
        pd.DataFrame(rows).to_csv(fn, index=False)
    del ens0; torch.cuda.empty_cache()
print('DONE fold', k, flush=True)
