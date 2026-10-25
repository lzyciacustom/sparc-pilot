"""Build the CRC test set, patient groups, 6-fold split and HEST-1k training pool from the HEST metadata table.
Default mode reproduces the pilot files byte-for-byte (hest1k_crc_candidates.csv, crc_folds.csv, hest1k_train_pool.csv).
--strict additionally removes from the training pool every slide that shares (dataset, patient) or a COLON MAP subseries
block with a CRC test group (fixes the patient-level leakage found in the audit; see docs/AUDIT.md).
Run from the flat workspace after `download_hest.py meta`."""
import sys, numpy as np, pandas as pd
STRICT = '--strict' in sys.argv
m = pd.read_csv('hf_probe/MahmoodLab_hest/HEST_v1_2_0.csv')
bench = set(pd.read_csv('hest_bench_inventory.csv')['sample'])
# --- CRC test candidates (human bowel, cancer, whole-transcriptome Visium family) ---
b = m[(m.organ == 'Bowel') & (m.species == 'Homo sapiens')].copy()
crc = b[(b.disease_state.str.contains('Cancer', case=False, na=False)) & (b.oncotree_code.isin(['COAD', 'READ', 'COADREAD', 'CRC', 'MACR']) | b.oncotree_code.notna())]
wt = crc[crc.st_technology.isin(['Visium', 'Visium HD', "Visium HD 3'", 'Spatial Transcriptomics'])]
c = m.set_index('id').loc[wt.id].reset_index(); c = c[c.nb_genes > 10000]          # drops targeted panel TENX28
def grp(r):
    if r.dataset_title.startswith('COLON MAP'): return 'COLONMAP_' + '_'.join(str(r.subseries).split('_')[:2])
    if r.id.startswith('ZEN'): return 'ZEN_' + str(r.patient).replace(' ', '')
    if r.id in ('TENX152', 'TENX154', 'TENX155', 'TENX156'): return 'TENX_immune_P1'
    if r.id in ('TENX89', 'TENX90', 'TENX91', 'TENX92'): return 'TENX_cytassist_CRC'
    return 'TENX_' + r.id
c['group'] = c.apply(grp, axis=1); c['in_bench'] = c.id.isin(bench)
c[['id', 'group', 'dataset_title', 'st_technology', 'oncotree_code', 'preservation_method', 'spots_under_tissue', 'nb_genes', 'disease_comment', 'in_bench']].to_csv('hest1k_crc_candidates.csv', index=False)
# --- 6 folds: greedy balance of spots over patient groups (largest group first) ---
cand = pd.read_csv('hest1k_crc_candidates.csv')
g = cand.groupby('group').spots_under_tissue.sum().sort_values(ascending=False)
K = 6; load = np.zeros(K); fold = {}
for gr in g.index: k = int(np.argmin(load)); fold[gr] = k; load[k] += g[gr]
cand['cv_fold'] = cand.group.map(fold); cand.to_csv('crc_folds.csv', index=False)
# --- training pool: human whole-transcriptome Visium slides outside HEST-Bench (CRC test ids are dropped at run time) ---
h = m[(m.species == 'Homo sapiens') & (m.st_technology.isin(['Visium', 'Visium HD', "Visium HD 3'"])) & (m.nb_genes > 10000)]
h = h[~h.id.isin(bench)]
if STRICT:
    t = m.set_index('id').loc[cand.id]
    keys = set(zip(t.dataset_title, t.patient.fillna('').astype(str).str.strip())) - {(d, '') for d in t.dataset_title}
    blocks = {'_'.join(str(s).split('_')[:2]) for d, s in zip(t.dataset_title, t.subseries) if str(d).startswith('COLON MAP')}
    pk = list(zip(h.dataset_title, h.patient.fillna('').astype(str).str.strip()))
    hb = ['_'.join(str(s).split('_')[:2]) if str(d).startswith('COLON MAP') else None for d, s in zip(h.dataset_title, h.subseries)]
    drop = [(k in keys) or (b_ in blocks) for k, b_ in zip(pk, hb)]
    print('strict: removing', int(np.sum(drop) - h.id.isin(cand.id)[drop].sum()), 'non-test pool slides linked to test patients:',
          sorted(set(h.id[drop]) - set(cand.id)))
    h = h[~np.array(drop) | h.id.isin(cand.id)]
h[['id', 'dataset_title', 'organ', 'disease_state', 'oncotree_code', 'st_technology', 'patient', 'subseries', 'preservation_method', 'spots_under_tissue', 'nb_genes']].to_csv('hest1k_train_pool.csv', index=False)
print(len(cand), 'CRC slides,', cand.group.nunique(), 'groups;', len(h), 'pool slides (', int((~h.id.isin(cand.id)).sum()), 'after removing CRC ids )')
