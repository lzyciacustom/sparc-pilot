"""Stage-3 figures from results/tables/{headroom,composition,mdacc} and results/raw/hest1k_h1_long.parquet:
fig_headroom_composition (noise ceiling, composition-aware model), fig_mdacc_micro_macro (MDACC), fig_proposal_numbers (the
numbers quoted in the proposal; README Figure A), fig_bm_heldout_controls (held-out-donor controls for b_M; README Figure B).
usage (repo root): python scripts/07_analysis/figures_stage3.py   -> results/figures/regenerated/*.png
The versions shipped in results/figures/stage3_composition/ were produced by this script."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 300, 'pdf.fonttype': 42, 'axes.titleweight': 'bold'})
T = 'results/tables'; OUT = 'results/figures/regenerated'; os.makedirs(OUT, exist_ok=True)
BLUE, GREY, RED, META = '#1f4e79', '#9a9a9a', '#c0504d', '#7f7f7f'
LEAK = {'COLONMAP_8270_AS', 'TENX_immune_P1'}


_LETTERS = []


def letter(ax, s): _LETTERS.append((ax, s))


def place_letters(fig):
    """Panel letters on one row, each at the left edge of its panel (axis label included), top-aligned with the titles."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    items = [(s, ax.get_tightbbox(r).transformed(inv)) for ax, s in _LETTERS if ax.figure is fig]
    top = max(b.y1 for _, b in items)
    for s, b in items: fig.text(b.x0, top, s, fontsize=9, fontweight='bold', va='bottom', ha='left')
    _LETTERS[:] = [(a, s) for a, s in _LETTERS if a.figure is not fig]


def strip(ax, data, labels, colors, seed=0):
    rng = np.random.default_rng(seed)
    for i, (v, c) in enumerate(zip(data, colors)):
        ax.scatter(i + rng.uniform(-0.14, 0.14, len(v)), v, s=9, color=c, alpha=0.75, lw=0, zorder=2)
        ax.hlines(np.median(v), i - 0.25, i + 0.25, color='black', lw=1.4, zorder=3)
    ax.set_xticks(range(len(data))); ax.set_xticklabels(labels); ax.set_xlim(-0.6, len(data) - 0.4)


# Figure 1: noise ceiling and composition-aware decomposition (CRC-only training, H-optimus-1, 16 patient groups, 3 seeds)
nc = pd.read_csv(f'{T}/headroom/noise_ceiling_by_program.csv', index_col=0)
cl = pd.read_csv(f'{T}/composition/comp_long.csv'); cl = cl[~cl.group.isin(LEAK)]
G = cl.groupby(['target', 'decon', 'seed', 'group']).mean(numeric_only=True).groupby(['target', 'decon', 'group']).mean()
U, Rn, Dj = G.loc[('ulm', 'out_parquet')], G.loc[('rand0ulm', 'out_parquet')], G.loc[('ulm', 'out_disjoint')]
fig, axs = plt.subplots(1, 3, figsize=(8.0, 2.8), gridspec_kw=dict(width_ratios=[1.1, 1, 1.2], wspace=0.5))
ax = axs[0]
ax.scatter(nc.ceil_count_ulm, nc.ach_ulm, s=14, color=BLUE, zorder=3)
ax.plot([0.4, 1], [0.4, 1], color=META, lw=0.8, ls='--'); ax.text(0.47, 0.49, 'achieved = ceiling', color=META, fontsize=6)
ax.set_xlim(0.4, 1.0); ax.set_ylim(0.1, 0.78); ax.set_xlabel('Reliability ceiling of measured target'); ax.set_ylabel('Achieved depth-adjusted r')
ax.set_title('About half the ceiling reached', loc='left')
for p, lab, xy in [('HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION', 'EMT', (0.86, 0.74)), ('REACTOME_NEUTROPHIL_DEGRANULATION', 'Neutrophil degran.', (0.66, 0.13)),
                   ('REACTOME_MISMATCH_REPAIR', 'Mismatch repair', (0.42, 0.29))]:
    ax.annotate(lab, (nc.loc[p, 'ceil_count_ulm'], nc.loc[p, 'ach_ulm']), xytext=xy, fontsize=6, arrowprops=dict(arrowstyle='-', lw=0.5, color=META))
letter(ax, 'a')
cols = [('total_direct', 'Direct\nULM'), ('total_componly', 'Comp.\nonly'), ('total_twostage', 'Two-\nstage'), ('total_joint', 'Joint')]
strip(axs[1], [U[c].values for c, _ in cols], [lab for _, lab in cols], [GREY, GREY, BLUE, BLUE])
axs[1].set_ylabel('Program accuracy (depth-adj. r)'); axs[1].set_title('Decomposition beats direct', loc='left'); letter(axs[1], 'b')
cols2 = [(U, 'resid_direct_implicit', 'Direct\n(implicit)'), (U, 'resid_twostage', 'Two-\nstage'), (U, 'resid_joint', 'Joint'),
         (Dj, 'resid_twostage', 'Gene-\ndisjoint'), (Rn, 'resid_twostage', 'Random\nsets')]
strip(axs[2], [d[c].values for d, c, _ in cols2], [lab for *_, lab in cols2], [GREY, BLUE, BLUE, BLUE, RED])
axs[2].set_ylabel('Residual accuracy (depth-adj. r)'); axs[2].set_title('Residual beyond random sets', loc='left'); letter(axs[2], 'c')
place_letters(fig); fig.savefig(f'{OUT}/fig_headroom_composition.png', bbox_inches='tight'); plt.close(fig)
print('fig1 medians', {c: round(U[c].median(), 3) for c, _ in cols}, {lab.replace('\n', ' '): round(np.median(d[c]), 3) for d, c, lab in cols2})

# Figure 2: MDACC liver metastasis stroma, micro- vs macrometastasis (patient level, 10 patients with both regions)
P = pd.read_csv(f'{T}/mdacc/mdacc_patient_level.csv'); con = 'micro_vs_macro_stroma'
cur = P[(P.contrast == con) & (P.set == 'curated') & (P.component == 'resid')].set_index('program')
rnd = P[(P.contrast == con) & (P.set == 'random_loo') & (P.component == 'resid')]
short = {'GSE26495_PD1HIGH_VS_PD1LOW_CD8_TCELL_UP': 'PD-1-high CD8 T cell', 'REACTOME_COLLAGEN_FORMATION': 'Collagen formation',
         'HALLMARK_TNFA_SIGNALING_VIA_NFKB': 'TNFα/NF-κB', 'HALLMARK_INTERFERON_GAMMA_RESPONSE': 'IFN-γ response', 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION': 'EMT'}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.55))
ax = axs[0]; rng = np.random.default_rng(0)
ax.scatter(rng.normal(0, 0.06, len(rnd)), rnd.z_mean, s=12, color=RED, alpha=0.7)
ax.scatter(1 + rng.normal(0, 0.06, len(cur)), cur.z_mean, s=12, color=BLUE, alpha=0.8)
mx = rnd.z_mean.abs().max(); ax.axhspan(-mx, mx, color=META, alpha=0.15, lw=0); ax.text(0.5, mx + 0.1, 'random range', ha='center', va='bottom', fontsize=6, color=META)
for p, lab, xy in [('GSE26495_PD1HIGH_VS_PD1LOW_CD8_TCELL_UP', 'PD-1-high CD8', (0.1, cur.z_mean.max() - 0.3)),
                   ('REACTOME_COLLAGEN_FORMATION', 'Collagen formation', (-0.4, cur.z_mean.min() - 0.6))]:
    ax.annotate(lab, (1, cur.loc[p, 'z_mean']), xytext=xy, fontsize=6, arrowprops=dict(arrowstyle='-', lw=0.5, color=META))
ax.set_xticks([0, 1]); ax.set_xticklabels(['Random\nsets', 'Curated\nprograms']); ax.set_xlim(-0.5, 1.5)
ax.set_ylim(min(-3, cur.z_mean.min() - 0.9), max(4.2, cur.z_mean.max() + 0.5))
ax.set_ylabel('Residual shift, micro − macro\n(z vs random sets, mean of 10 patients)'); ax.set_title('Residual shifts exceed random sets', loc='left'); letter(ax, 'a')
ax = axs[1]
progs = ['GSE26495_PD1HIGH_VS_PD1LOW_CD8_TCELL_UP', 'HALLMARK_INTERFERON_GAMMA_RESPONSE', 'HALLMARK_TNFA_SIGNALING_VIA_NFKB', 'REACTOME_COLLAGEN_FORMATION', 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION']
L = P[(P.contrast == con) & (P.set == 'curated')].pivot_table(index='program', columns='component', values='raw_diff').loc[progs]
y = np.arange(len(progs))[::-1]; h = 0.25
for k, (c, col, lab) in enumerate([('comp', GREY, 'Composition'), ('depth', '#d9d9d9', 'Depth'), ('resid', BLUE, 'Residual')]):
    ax.barh(y + (1 - k) * h, L[c].values, height=h, color=col, label=lab, edgecolor='none')
ax.scatter(L['y'].values, y, marker='|', s=60, color='black', zorder=3, label='Total')
ax.axvline(0, color='black', lw=0.6); ax.set_yticks(y); ax.set_yticklabels([short[p] for p in progs])
ax.set_xlabel('Micro − macro metastasis stroma (SD units)'); ax.set_title('Shifts are mostly residual', loc='left')
ax.legend(frameon=False, fontsize=6, loc='lower right'); letter(ax, 'b')
place_letters(fig); fig.savefig(f'{OUT}/fig_mdacc_micro_macro.png', bbox_inches='tight'); plt.close(fig)
print('fig2 components\n', L.round(2).to_string())


# Figure 3: the numbers quoted in the proposal (means over 16 patient groups; lines join the same group)
def paired(ax, pairs, labels, x0=0):
    """pairs: list of (curated, random) arrays aligned by group; draws group lines and mean bars with values"""
    for k, (a, b) in enumerate(pairs):
        xa, xb = x0 + 2 * k, x0 + 2 * k + 0.8
        for u, v in zip(a, b): ax.plot([xa, xb], [u, v], color='#d0d0d0', lw=0.6, zorder=1)
        ax.scatter(np.full(len(a), xa), a, s=9, color=BLUE, lw=0, alpha=0.8, zorder=2)
        ax.scatter(np.full(len(b), xb), b, s=9, color=GREY, lw=0, alpha=0.8, zorder=2)
        for x, v, c, ha, dx in [(xa, np.mean(a), BLUE, 'right', -0.22), (xb, np.mean(b), '#555555', 'left', 0.22)]:
            ax.hlines(v, x - 0.18, x + 0.18, color=c, lw=2, zorder=3); ax.text(x + dx, v, f'{v:.2f}', color=c, fontsize=6, ha=ha, va='center')
        n = int((np.asarray(a) > np.asarray(b)).sum()); ax.text(xa + 0.4, ax.get_ylim()[1], f'{n}/{len(a)}', fontsize=6, ha='center', va='top', color=META)
    ax.set_xticks([x0 + 2 * k + 0.4 for k in range(len(pairs))]); ax.set_xticklabels(labels); ax.axhline(0, color=META, lw=0.5)
    ax.set_xlim(min(ax.get_xlim()[0], x0 - 0.75), x0 + 2 * len(pairs) - 0.45)


LBLUE = '#8fb3d9'
grp = pd.read_csv('metadata/crc_folds.csv').set_index('id')['group']; grp = grp[~grp.isin(LEAK)]
H1 = pd.read_parquet('results/raw/hest1k_h1_long.parquet'); H1 = H1[(H1.train == 'hest1k') & (H1.model == 'base') & (H1['head'] == 'det')]


def per_group(t, m):
    s = H1[H1.target == t].groupby(['seed', 'sample'])[m].mean().groupby('sample').mean()
    return s.groupby(s.index.map(grp)).mean()


def rand_ss(m): return sum(per_group(f'rand{i}ssgsea', m) for i in range(3)) / 3


gi = per_group('ssgsea', 'pearson').index
raw = (per_group('ssgsea', 'pearson')[gi], rand_ss('pearson')[gi])
dss = (per_group('ssgsea', 'pearson_depthres')[gi], rand_ss('pearson_depthres')[gi])
dul = (per_group('ulm', 'pearson_depthres')[gi], per_group('rand0ulm', 'pearson_depthres')[gi])
fig, axs = plt.subplots(1, 3, figsize=(8.6, 2.9), gridspec_kw=dict(width_ratios=[0.8, 1.1, 1.1], wspace=0.5))
for ax, lim in zip(axs, [(-0.1, 0.85), (-0.1, 0.7), (-0.05, 0.6)]): ax.set_ylim(*lim)
paired(axs[0], [raw], ['ssGSEA'])
axs[0].set_ylabel('Held-out r (mean per patient group)'); axs[0].set_title('Raw: random sets close', loc='left'); letter(axs[0], 'a')
paired(axs[1], [dss, dul], ['ssGSEA', 'ULM'])
axs[1].set_ylabel('Depth-adjusted r'); axs[1].set_title('Depth-adjusted: programs separate', loc='left'); letter(axs[1], 'b')
d_ = U['total_direct'], U['total_twostage']; r_ = U['resid_twostage'], Rn['resid_twostage'].reindex(U.index)
ax = axs[2]
for u, v in zip(*d_): ax.plot([0, 0.8], [u, v], color='#d0d0d0', lw=0.6, zorder=1)
ax.scatter(np.zeros(len(U)), d_[0], s=9, color=LBLUE, lw=0, zorder=2); ax.scatter(np.full(len(U), 0.8), d_[1], s=9, color=BLUE, lw=0, zorder=2)
for x, v, c, ha, dx in [(0, d_[0].mean(), LBLUE, 'right', -0.22), (0.8, d_[1].mean(), BLUE, 'left', 0.22)]:
    ax.hlines(v, x - 0.18, x + 0.18, color=c, lw=2, zorder=3); ax.text(x + dx, v, f'{v:.2f}', color=c, fontsize=6, ha=ha, va='center')
ax.text(0.4, 0.6, f'{int((d_[1] > d_[0]).sum())}/{len(U)}', fontsize=6, ha='center', va='top', color=META)
paired(ax, [r_], ['Residual\n(curated vs random)'], x0=2)
ax.set_xlim(-0.75, 3.55); ax.set_xticks([0, 0.8, 2.4]); ax.set_xticklabels(['Direct', 'Two-\nstage', 'Residual\n(curated vs random)'])
ax.set_ylabel('Depth-adjusted r (CRC-only training)'); ax.set_title('Composition-aware model', loc='left'); letter(ax, 'c')
h = [plt.Line2D([], [], marker='o', ls='', color=BLUE, ms=4, label='curated programs'), plt.Line2D([], [], marker='o', ls='', color=GREY, ms=4, label='size-matched random sets')]
fig.legend(handles=h, loc='upper center', ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.07))
place_letters(fig); fig.savefig(f'{OUT}/fig_proposal_numbers.png', bbox_inches='tight'); plt.close(fig)
print('fig3 means', {k: (round(a.mean(), 3), round(b.mean(), 3)) for k, (a, b) in dict(raw=raw, depth_ssgsea=dss, depth_ulm=dul, direct_vs_twostage=d_, residual=r_).items()},
      'ULM - ssGSEA depth-adj', round((dul[0] - dss[0]).mean(), 3), f'{int((dul[0] > dss[0]).sum())}/16')

# Figure 4: held-out-donor controls for the reference baseline b_M (simulation, GSE132465; results/tables/composition/bm_heldout/)
B = f'{T}/composition/bm_heldout'
A = pd.concat([pd.read_csv(f'{B}/bm_heldout_testA_generalisation_s{s}.csv').assign(seed=s) for s in (2026, 2027, 2028)])
S = pd.read_csv(f'{B}/bm_heldout_testB2_summary.csv'); C = pd.read_csv(f'{B}/bm_heldout_testC_summary.csv')
fig, axs = plt.subplots(1, 3, figsize=(9.0, 2.9), gridspec_kw=dict(width_ratios=[1, 1.6, 1], wspace=0.55))
ax = axs[0]
for s, mk in zip((2026, 2027, 2028), 'o^s'):
    a = A[A.seed == s]; ax.scatter(a.r2_within_train_donors, a.r2_heldout_known_pi, s=10, marker=mk, color=BLUE, alpha=0.6, lw=0, label=f'split {s - 2025}')
ax.plot([0, 1], [0, 1], color=META, lw=0.8, ls='--'); ax.set_xlim(0.2, 1); ax.set_ylim(0.2, 1)
ax.set_xlabel('R² on training donors'); ax.set_ylabel('R² on 8 held-out donors'); ax.legend(frameon=False, fontsize=5.5, loc='upper left')
ax.set_title('b_M transfers to new donors', loc='left'); letter(ax, 'a')
ax = axs[1]; arms = [('total', 'none'), ('known', 'true\nfractions'), ('nnls', 'NNLS'), ('converted', 'counts,\nconverted'), ('cellfrac', 'counts,\nraw')]
P_ = S[S.seed == 'pooled'].set_index('arm'); sp_ = S[S.seed != 'pooled']
for i, (k, lab) in enumerate(arms):
    ax.bar(i, P_.loc[k, 'FPR'], color=GREY if k in ('total', 'cellfrac') else BLUE, width=0.65, alpha=0.85)
    ax.bar(i, P_.loc[k, 'FPR_eff025'], color='black', width=0.65, alpha=0.35)
    ax.scatter(np.full(3, i) + np.array([-0.15, 0, 0.15]), sp_[sp_.arm == k].FPR, s=7, color='black', zorder=3)
    ax.text(i, max(P_.loc[k, 'FPR'], sp_[sp_.arm == k].FPR.max()) + 0.025, f'{P_.loc[k, "FPR"]:.0%}', ha='center', fontsize=6.5, fontweight='bold')
ax.axhline(0.039, color=RED, lw=0.8, ls='--')
ax.set_xticks(range(len(arms))); ax.set_xticklabels([l for _, l in arms], fontsize=6); ax.set_ylim(0, 0.68); ax.set_xlabel('Composition used for the residual')
ax.set_ylabel('False-positive rate\n(composition-only contrasts)'); ax.set_title('Few false calls after decomposition', loc='left'); letter(ax, 'b')
ax.text(0.98, 0.80, 'bar: pooled; dots: 3 donor splits\ndark: |effect| > 0.25 SD\nred line: nominal 3.9%', transform=ax.transAxes, fontsize=5.5, ha='right', color=META)
ax = axs[2]
ax.scatter(C.total, C.resid_known, s=14, color=BLUE, label='true fractions', zorder=3); ax.scatter(C.total, C.resid_nnls, s=14, facecolor='none', edgecolor=BLUE, label='NNLS', zorder=3)
m = max(C.total.max(), C.resid_known.max()) * 1.08; ax.plot([0, m], [0, m], color=META, lw=0.8, ls='--')
ax.set_xlabel('Planted shift (SD)'); ax.set_ylabel('Recovered residual (SD)'); ax.legend(frameon=False, fontsize=5.5, loc='upper left')
ax.set_title('Planted state shifts recovered', loc='left'); letter(ax, 'c')
place_letters(fig); fig.savefig(f'{OUT}/fig_bm_heldout_controls.png', bbox_inches='tight'); plt.close(fig)
print('fig4 pooled FPR', P_.FPR.round(3).to_dict(), 'recovered/planted', (C.resid_known / C.total).round(2).agg(['min', 'max']).tolist())
