"""Stage-3 figures (headroom, composition-aware decomposition, MDACC micro- vs macrometastasis) from results/tables/{headroom,composition,mdacc}.
usage (repo root): python scripts/07_analysis/figures_stage3.py   -> results/figures/stage3_composition/*.png"""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 300, 'pdf.fonttype': 42, 'axes.titleweight': 'bold'})
T = 'results/tables'; OUT = 'results/figures/stage3_composition'; os.makedirs(OUT, exist_ok=True)
BLUE, GREY, RED, META = '#1f4e79', '#9a9a9a', '#c0504d', '#7f7f7f'
LEAK = {'COLONMAP_8270_AS', 'TENX_immune_P1'}


def letter(ax, s): ax.text(-0.18, 1.06, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')


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
fig.savefig(f'{OUT}/fig_headroom_composition.png', bbox_inches='tight'); plt.close(fig)
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
fig.savefig(f'{OUT}/fig_mdacc_micro_macro.png', bbox_inches='tight'); plt.close(fig)
print('fig2 components\n', L.round(2).to_string())
