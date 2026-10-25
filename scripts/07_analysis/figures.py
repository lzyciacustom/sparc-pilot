"""Regenerate the HEST-1k-stage figures from results/raw and results/tables (run hest1k_stats.py, mouse_stats.py, nuclei_stats.py first).
Output: results/figures/regenerated/*.png. The figures shipped in results/figures/ are the versions produced during the pilot."""
import os, glob, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 300, 'pdf.fonttype': 42})
OUT = 'results/figures/regenerated'; os.makedirs(OUT, exist_ok=True)
REAL, RAND, ORANGE = '#1f5fa8', '#9a9a9a', '#b35806'; M = ['pearson', 'pearson_centered', 'pearson_depthres']
MLAB = {'pearson': 'Raw', 'pearson_centered': 'Centred', 'pearson_depthres': 'Depth-residualised'}
grp = pd.read_csv('metadata/crc_folds.csv').set_index('id')['group']
D = pd.read_parquet('results/raw/hest1k_wave1_long.parquet'); KEY = ['train', 'target', 'model', 'head']
SL = D.groupby(KEY + ['seed', 'sample'])[M].mean().reset_index().groupby(KEY + ['sample'])[M].mean().reset_index()
G = SL.assign(group=SL['sample'].map(grp)).groupby(KEY + ['group'])[M].mean(); cond = lambda tr, t, mo='base', h='det': G.loc[(tr, t, mo, h)]
R = pd.read_parquet('results/raw/hest1k_wave2_long.parquet'); R['cond'] = R.tag.str.split('__').str[:2].str.join('__')
def gt(c):
    x = R[R.cond == c]; s = x.groupby(['seed', 'sample'])[M].mean().reset_index().groupby('sample')[M].mean(); return s.groupby(s.index.map(grp)).mean()
def paired(ax, A, B, X, ca=REAL, cb=RAND):
    for i, m in enumerate(M):
        xa, xb = X[i] - .17, X[i] + .17
        for g in A.index: ax.plot([xb, xa], [B.loc[g, m], A.loc[g, m]], color='.85', lw=.6, zorder=1)
        ax.scatter(np.full(len(A), xa), A[m], s=10, color=ca, alpha=.6, lw=0); ax.scatter(np.full(len(B), xb), B[m], s=10, color=cb, alpha=.6, lw=0)
        ax.hlines(A[m].mean(), xa - .12, xa + .12, color=ca, lw=2.2, zorder=3); ax.hlines(B[m].mean(), xb - .12, xb + .12, color='.35', lw=2.2, zorder=3)
        ax.text(xa - .14, A[m].mean(), f'{A[m].mean():.2f}', ha='right', va='center', color=ca, fontsize=7)
        ax.text(xb + .14, B[m].mean(), f'{B[m].mean():.2f}', ha='left', va='center', color='.3', fontsize=7)
    ax.set_xticks(X); ax.set_xticklabels([MLAB[m] for m in M]); ax.axhline(0, color='.6', lw=.6)
# 1 real vs random
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True); X = [0, 1.45, 2.9]
for ax, (real, rands, nm) in zip(axs, [('ssgsea', ['rand0ssgsea', 'rand1ssgsea', 'rand2ssgsea'], 'ssGSEA targets (3 random draws)'), ('ulm', ['rand0ulm'], 'ULM targets (1 random draw)')]):
    paired(ax, cond('hest1k', real), sum(cond('hest1k', r) for r in rands) / len(rands), X); ax.set_title(nm, loc='left'); ax.set_xlim(-.7, 3.6)
axs[0].set_ylabel('Held-out Pearson r\n(mean per patient group)'); fig.suptitle('Curated programs vs size-matched random gene sets (HEST-1k training, 58 CRC test slides)', x=.08, ha='left', fontsize=9)
fig.savefig(f'{OUT}/fig_hest1k_real_vs_random.png', bbox_inches='tight')
# 2 targets
fams = [('ssgsea', 'ssGSEA'), ('mlm', 'MLM'), ('ssgsea_depthres', 'ssGSEA,\ndepth-regressed'), ('ulm', 'ULM')]
fig, ax = plt.subplots(figsize=(4.8, 3.1))
for tr, col, off in [('crc_only', '#8fb3dc', -.17), ('hest1k', REAL, .17)]:
    V = pd.concat([cond(tr, f).pearson_depthres.rename(f) for f, _ in fams], axis=1)
    for i, (f, _) in enumerate(fams):
        ax.scatter(np.full(len(V), i + off), V[f], s=9, color=col, alpha=.75, lw=0); ax.hlines(V[f].mean(), i + off - .12, i + off + .12, color=col, lw=2.2)
ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in fams]); ax.set_ylabel('Depth-residualised Pearson r\n(mean per patient group)')
ax.set_title('Removing the shared component makes programs more predictable', loc='left')
ax.text(.02, .97, '● HEST-1k training (~376 slides/fold)', color=REAL, transform=ax.transAxes, fontsize=7, va='top'); ax.text(.02, .9, '● CRC-only training (~48 slides/fold)', color='#6f9ccf', transform=ax.transAxes, fontsize=7, va='top')
fig.savefig(f'{OUT}/fig_hest1k_targets.png', bbox_inches='tight')
# 3 spatial + 4 upgrades (delta vs base with seed-noise band)
def noise(t):
    x = D[(D.train == 'hest1k') & (D.target == t) & (D.model == 'base') & (D['head'] == 'det')]
    b = x.groupby(['seed', 'sample']).pearson_depthres.mean().reset_index(); b['group'] = b['sample'].map(grp); b = b.groupby(['seed', 'group']).pearson_depthres.mean().unstack(0)
    return pd.concat([b[0] - b[1], b[0] - b[2], b[1] - b[2]]).std()
def deltas(ax, items, band, ylim=(-.1, .15)):
    ax.axhspan(-band, band, color='.92', zorder=0); ax.axhline(0, color='.5', lw=.7)
    for i, (A, B, lab) in enumerate(items):
        d = (A - B).dropna(); p = wilcoxon(d).pvalue; col = ORANGE if (d.mean() < 0 and p < .05) else REAL
        ax.scatter(i + np.random.default_rng(i).uniform(-.08, .08, len(d)), d, s=11, color=col, alpha=.75, lw=0); ax.hlines(d.mean(), i - .2, i + .2, color='.15', lw=2)
        ax.text(i, ylim[1] - .035, f'{d.mean():+.3f}\np={p:.2g}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([x[2] for x in items]); ax.set_ylim(*ylim)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
for ax, t, nm in zip(axs, ['ssgsea', 'ulm'], ['ssGSEA targets', 'ULM targets']):
    B = cond('hest1k', t).pearson_depthres
    deltas(ax, [(cond('hest1k', t, v).pearson_depthres, B, l) for v, l in [('mlp', 'Spot-only\nMLP'), ('concert', 'CONCERT-\nstyle bias'), ('gpk16', 'GP kernel\nk=16'), ('gpk48', 'GP kernel\nk=48')]], 2 * noise(t) / np.sqrt(3)); ax.set_title(nm, loc='left')
axs[0].set_ylabel('Δ depth-residualised r vs\nSPARC-Map (distance only)'); fig.savefig(f'{OUT}/fig_hest1k_spatial.png', bbox_inches='tight')
ups = [('ulm__tn-slide', 'ulm__tn-robust', 'Slide\nz-score'), ('ulm__tn-robust+tech', 'ulm__tn-robust', 'Technology\none-hot'), ('ulm__tn-robust+progw', 'ulm__tn-robust', 'Reliability\nweights'),
       ('ulm__tn-slide+tech+progw', 'ulm__tn-robust', 'All three'), ('ulm__tn-robust+gene_aux', 'ulm__tn-robust', 'Gene\naux. loss'), ('ulm__tn-robust+genes', 'ulm__tn-robust', 'Gene\ndecoder*'),
       ('ulm__tn-robust+ctx', 'ulm__tn-robust', '448 µm\ncontext'), ('ulm__tn-robust+ctx+crc_only', 'ulm__tn-robust+crc_only', '448 µm\ncontext†')]
fig, ax = plt.subplots(figsize=(7.8, 3.1)); deltas(ax, [(gt(a).pearson_depthres, gt(b).pearson_depthres, l) for a, b, l in ups], 2 * noise('ulm') / np.sqrt(3), (-.1, .12))
ax.set_ylabel('Δ depth-residualised r vs\nULM SPARC-Map baseline'); ax.set_title('None of the engineering upgrades beats the ULM baseline', loc='left')
fig.text(.125, -.08, '* 2 seeds. † CRC-only training vs CRC-only baseline. Orange: significantly worse (p < 0.05).', fontsize=7, color='.35'); fig.savefig(f'{OUT}/fig_upgrades.png', bbox_inches='tight')
# 5 decoupling
P = pd.read_csv('results/tables/decoupling_residual_per_program.csv', index_col=0)
r2f = 'results/tables/composition_r2_by_program.csv'
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=.32))
paired(axs[0], gt('ulm__tn-robust+decoup-res'), gt('ulm__tn-robust+decoup-randres'), X); axs[0].set_xlim(-.7, 3.6)
axs[0].set_ylabel('Held-out r on composition-residual targets'); axs[0].set_title('H&E predicts program activity beyond composition', loc='left')
if os.path.exists(r2f):
    r2 = pd.read_csv(r2f).set_index('program').median_R2; y = P['pearson_depthres_real_minus_random'].reindex(r2.index)
    axs[1].scatter(r2, y, s=18, color=REAL, lw=0); axs[1].axhline(0, color='.6', lw=.6)
    axs[1].set_xlabel('Variance explained by composition (median R²)'); axs[1].set_ylabel('Residual signal above random\n(Δ depth-residualised r)')
    axs[1].set_title(f'Spearman ρ = {spearmanr(r2, y).correlation:.2f}, n = {len(r2)} programs', loc='left')
fig.savefig(f'{OUT}/fig_decoup.png', bbox_inches='tight')
# 6 mouse retraining
S = pd.read_csv('results/tables/mouse_ft_per_section.csv'); U = S[S.target == 'ulm']; Nn = S[S.target == 'rand0ulm']
conds = [('none', 'human_only', 'Human only'), ('slide', 'mouse_only', 'Mouse only'), ('slide', 'human_ft', 'Human → mouse\nfine-tune'), ('dataset', 'mouse_only', 'Mouse only'), ('dataset', 'human_ft', 'Human → mouse\nfine-tune')]
xs = [0, 1.4, 2.4, 3.8, 4.8]; orgc = {'Bowel': REAL, 'Liver': ORANGE, 'Skin': '#5a9e6f'}
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for x, (sch, c, l) in zip(xs, conds):
    v = U[(U.scheme == sch) & (U.condition == c)]; ax.bar(x, Nn[(Nn.scheme == sch) & (Nn.condition == c)].pearson_depthres.mean(), width=.6, color='.88', zorder=0)
    for o, g in v.groupby('organ'): ax.scatter(x + np.random.default_rng(len(o)).uniform(-.14, .14, len(g)), g.pearson_depthres, s=13, color=orgc[o], alpha=.8, lw=0)
    ax.hlines(v.pearson_depthres.mean(), x - .28, x + .28, color='.1', lw=2.2); ax.text(x, .56, f'{v.pearson_depthres.mean():.2f}', ha='center', fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels([c[2] for c in conds]); ax.set_ylim(-.03, .6); ax.set_ylabel('Depth-residualised r (ULM, per mouse section)')
ax.set_title('Retraining on same-study mouse sections more than doubles signal (grey: random-gene-set null)', loc='left')
fig.savefig(f'{OUT}/fig_mouse_ft.png', bbox_inches='tight')
# 7 nuclei adjustment
E = pd.read_csv('results/raw/nuclei_adjusted_per_slide.csv')
fig, ax = plt.subplots(figsize=(4.2, 3.2))
for m, col, lab in [('ulm_s0', REAL, 'curated programs'), ('rand0ulm_s0', '.45', 'random gene sets')]:
    g = E[E.model == m].pivot(index='sample', columns='adjust', values='r'); g = g.groupby(g.index.map(grp)).mean()
    for gi in g.index: ax.plot([0, 1], g.loc[gi, ['depth', 'depth+nuclei']], color=col, alpha=.35, lw=.8)
    for x, a in [(0, 'depth'), (1, 'depth+nuclei')]: ax.scatter(np.full(len(g), x), g[a], s=12, color=col, lw=0); ax.hlines(g[a].mean(), x - .1, x + .1, color=col, lw=2.4)
    ax.text(1.14, g['depth+nuclei'].mean(), f"{lab}\n{g['depth'].mean():.2f} → {g['depth+nuclei'].mean():.2f}", va='center', fontsize=7, color=col)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Depth-adjusted', 'Depth + nuclear\nfeatures adjusted']); ax.set_xlim(-.3, 1.9); ax.set_ylabel('Held-out Pearson r (mean per patient group)')
ax.set_title('Program signal survives nuclear-feature adjustment', loc='left'); fig.savefig(f'{OUT}/fig_nuclei_adjust.png', bbox_inches='tight')
print('figures ->', OUT, sorted(os.listdir(OUT)))
