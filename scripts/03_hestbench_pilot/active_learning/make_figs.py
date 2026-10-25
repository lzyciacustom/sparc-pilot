# called from python kernel after apply_figure_style(); defines make_fig(method)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
COL = {'epig': '#D55E00', 'maxunc': '#0072B2', 'random': '#7F7F7F', 'featmatch': '#009E73'}
LAB = {'epig': 'EPIG (batch, redundancy-penalised)', 'maxunc': 'Max. ensemble variance', 'random': 'Random',
       'featmatch': 'H&E-embedding match (no model)'}
ORDER = ['random', 'maxunc', 'featmatch', 'epig']

def make_fig(method, fn):
    D = f'al_results/{method}'
    runs = pd.read_csv(f'{D}/al_auc_per_run.csv'); lc = pd.read_csv(f'{D}/al_learning_curves.csv')
    lr = pd.read_csv(f'{D}/al_learning_curves_READ.csv')
    strats = [s for s in ORDER if s in set(lc.strategy)]
    n_units = runs.groupby('strategy').size().to_dict()
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.8))
    for j, (m, yl) in enumerate([('pearson', 'Pearson r (raw)'), ('pearson_depthres', 'Pearson r (depth-residualised)')]):
        for st in strats:
            g = lc[lc.strategy == st].sort_values('round'); lw = 1.8 if st == 'epig' else 1.1
            axs[0, j].errorbar(g['round'] * 2 + 2, g[f'{m}_mean'], g[f'{m}_sem'], color=COL[st], lw=lw, marker='o', ms=3, capsize=1.5, label=LAB[st])
            axs[1, j].errorbar(g['n_spots_mean'] / 1e3, g[f'{m}_mean'], g[f'{m}_sem'], color=COL[st], lw=lw, marker='o', ms=3, capsize=1.5)
        axs[0, j].set_xlabel('Labelled slides'); axs[1, j].set_xlabel('Labelled spots (k, mean per round)')
        axs[0, j].set_ylabel(yl); axs[1, j].set_ylabel(yl)
    axs[0, 0].set_title('Test-fold accuracy vs slide budget', loc='left')
    axs[1, 0].set_title('Same runs vs spot budget', loc='left')
    for st in strats:
        for m, ls in [('pearson', '-'), ('pearson_depthres', '--')]:
            g = lr[(lr.strategy == st) & (lr.metric == m)].sort_values('round')
            axs[0, 2].plot(g['round'] * 2 + 2, g['mean'], ls, color=COL[st], lw=1.6 if st == 'epig' else 1.0, marker='o', ms=2.5)
    axs[0, 2].set_title('READ test slides (n=4)', loc='left'); axs[0, 2].set_xlabel('Labelled slides'); axs[0, 2].set_ylabel('Pearson r')
    axs[0, 2].plot([], [], 'k-', lw=1, label='raw'); axs[0, 2].plot([], [], 'k--', lw=1, label='depth-res.'); axs[0, 2].legend(frameon=False, loc='lower right')
    # paired AUC differences per run
    W = runs.pivot_table(index=['fold', 'seed'], columns='strategy', values=['auc_pearson', 'auc_pearson_depthres'])
    comps = [c for c in ['random', 'maxunc', 'featmatch'] if c in strats]
    ax = axs[1, 2]; rng = np.random.default_rng(0)
    for i, c in enumerate(comps):
        for o, (m, mk) in enumerate([('auc_pearson', 'o'), ('auc_pearson_depthres', 's')]):
            d = (W[(m, 'epig')] - W[(m, c)]).dropna().values; x = i + (o - 0.5) * 0.35
            ax.scatter(x + rng.uniform(-.07, .07, len(d)), d, s=10, marker=mk, color=COL[c], alpha=.8, lw=0)
            ax.hlines(np.mean(d), x - .14, x + .14, color='k', lw=1.2)
    ax.axhline(0, color='k', lw=.6); ax.set_xticks(range(len(comps))); ax.set_xticklabels(['vs ' + {'random': 'random', 'maxunc': 'max-var', 'featmatch': 'H&E match'}[c] for c in comps])
    ax.set_ylabel('EPIG − comparator, AUC'); ax.set_title('Paired AUC gain per fold×seed', loc='left')
    ax.scatter([], [], marker='o', color='k', s=10, label='raw'); ax.scatter([], [], marker='s', color='k', s=10, label='depth-res.'); ax.legend(frameon=False, loc='lower left', fontsize=6)
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    for ax in axs.flat: ax.margins(0.05)
    for ax in [axs[0,0],axs[0,1],axs[0,2]]: ax.set_xticks([2,4,6,8,10,12,14])
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(fn, dpi=300, bbox_inches='tight'); return fig, n_units
