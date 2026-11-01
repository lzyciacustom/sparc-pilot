"""Follow-up to bm_controls.py.
(1) Realistic RNA-content conversion: cell-count fractions (what imaging measures) converted to transcript fractions using
    per-cell-type mean UMI per cell estimated from TRAINING donors only (rc_t), pi_tf = cf * rc / sum(cf * rc).
(2) Robustness: same tests A/B for three donor splits (seeds 2026 = original, 2027, 2028).
FPR unit: one test per (composition pair x program); statistic = Wilcoxon signed-rank over the 8 held-out donors of donor-level
differences (mean of 50 region-B spots minus mean of 50 region-A spots). 95% CI by bootstrap over the 40 composition pairs.
With 8 donors the exact two-sided Wilcoxon test rejects at p < 0.05 with null probability 0.039 (attainable p 0.0078, 0.0156,
0.0234, 0.0391, 0.0547, ...), the nominal reference for the FPRs.
Run from <workdir>/comp after bm_heldout_controls.py (it re-executes that script's data/training/test-A block per seed).
Outputs: comp/results/bm_heldout_testA_generalisation_s<seed>.csv, bm_heldout_testB2_splits_rnacontent.csv (per test), bm_heldout_testB2_summary.csv (FPR by arm and split,
also with |effect| > 0.10 / 0.25 SD), bm_heldout_rna_content_seed2026.csv (training vs held-out mean UMI per cell, seed 2026 split)."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f'{os.path.dirname(HERE)}/comp/results'
src = open(f'{HERE}/bm_heldout_controls.py').read(); head = src[:src.index('# ---------------- test B')]
out = []
for seed in [2026, 2027, 2028]:
    g = {}
    g['__file__'] = f'{HERE}/bm_heldout_controls.py'
    hs = head.replace('default_rng(2026)', f'default_rng({seed})').replace('bm_heldout_testA_generalisation.csv', f'bm_heldout_testA_generalisation_s{seed}.csv')
    exec(compile(hs, f'head{seed}', 'exec'), g)
    rng, K, HO, P, region, bM, SDY, wp = g['rng'], g['K'], g['HO'], g['P'], g['region'], g['bM'], g['SDY'], g['wp']
    lib_full, tix, tr = g['lib_full'], g['tix'], g['tr']
    rc = np.array([lib_full[tr & (tix == t)].mean() for t in range(K)])          # training-donor RNA content per type
    conv = lambda cf: (cf * rc) / (cf * rc).sum(1, keepdims=True)
    rows = []
    for rep in range(40):
        cA, cB = rng.dirichlet(np.ones(K)), rng.dirichlet(np.ones(K))
        acc = {k: [] for k in ['total', 'known', 'nnls', 'cellfrac', 'converted']}
        for dn in HO:
            ya, fa, ca, da, pa = region(dn, cA); yb, fb, cb, db, pb = region(dn, cB)
            m = lambda a, b: (np.nanmean(b, 0) - np.nanmean(a, 0)) / SDY
            acc['total'].append(m(ya, yb))
            for k, (qa, qb) in {'known': (fa, fb), 'nnls': (pa, pb), 'cellfrac': (ca, cb), 'converted': (conv(ca), conv(cb))}.items():
                acc[k].append(m(ya - bM(qa, da), yb - bM(qb, db)))
        for k in acc: acc[k] = np.array(acc[k])
        for j, p in enumerate(P):
            rows.append(dict(seed=seed, rep=rep, program=p, **{f'eff_{k}': acc[k][:, j].mean() for k in acc}, **{f'p_{k}': wp(acc[k][:, j]) for k in acc}))
    D = pd.DataFrame(rows); out.append(D)
    line = [f'seed {seed} held-out {list(map(str, HO))} r2_within {np.median(g["r2_within"]):.3f} r2_heldout {np.median(g["dfA"].r2_heldout_known_pi):.3f}']
    for k in ['total', 'known', 'nnls', 'cellfrac', 'converted']:
        per = D.groupby('rep')[f'p_{k}'].apply(lambda x: np.nanmean(x < 0.05)).values
        bs = [np.mean(np.random.default_rng(b).choice(per, len(per))) for b in range(2000)]
        line.append(f'  {k:9s} FPR {per.mean():.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]  median|eff| {np.nanmedian(np.abs(D[f"eff_{k}"])):.3f}')
    print('\n'.join(line), flush=True)
    if seed == 2026:                                                              # RNA-content transfer, training vs held-out donors
        rch = np.array([lib_full[~tr & (tix == t)].mean() for t in range(K)])
        pd.DataFrame({'type': g['TYPES'], 'train_mean_umi': rc, 'heldout_mean_umi': rch, 'ratio_heldout_train': rch / rc}
                     ).to_csv(f'{OUT}/bm_heldout_rna_content_seed2026.csv', index=False)
D = pd.concat(out); D.to_csv(f'{OUT}/bm_heldout_testB2_splits_rnacontent.csv', index=False)
arms, rows = ['total', 'known', 'nnls', 'cellfrac', 'converted'], []
for s, gg in list(D.groupby('seed')) + [('pooled', D)]:
    for a in arms:
        sig, e = gg[f'p_{a}'] < 0.05, gg[f'eff_{a}'].abs()
        rows.append(dict(seed=s, arm=a, FPR=sig.mean(), FPR_eff010=(sig & (e > 0.10)).mean(), FPR_eff025=(sig & (e > 0.25)).mean(),
                         med_abs_eff=e.median(), p95_abs_eff=e.quantile(.95)))
S = pd.DataFrame(rows); S.to_csv(f'{OUT}/bm_heldout_testB2_summary.csv', index=False); print(S.round(3).to_string(index=False))
