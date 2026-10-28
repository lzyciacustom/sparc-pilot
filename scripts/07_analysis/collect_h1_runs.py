"""Concatenate the H-optimus-1 per-slide result CSVs (results/<target>__base__det__s<seed>[__crconly].csv from hest1k_run.py)
into results/raw/hest1k_h1_long.parquet, in the same layout as hest1k_wave1_long.parquet (H-optimus-0).
usage (repo root): python scripts/07_analysis/collect_h1_runs.py <workspace>/results"""
import glob, os, sys, pandas as pd
src = sys.argv[1]
fs = sorted(f for f in glob.glob(f'{src}/*__base__det__s*.csv') if '__sm' not in os.path.basename(f))  # __sm*: smoothing variants, not reported
grp = pd.read_csv('metadata/crc_folds.csv').set_index('id')['group']
D = pd.concat([pd.read_csv(f).assign(file=os.path.basename(f)) for f in fs], ignore_index=True)
D['train'] = D.crc_only.map({True: 'crc_only', False: 'hest1k'}); D['group'] = D['sample'].map(grp); D['encoder'] = 'hoptimus1'
D.to_parquet('results/raw/hest1k_h1_long.parquet', index=False)
print(len(fs), 'files', D.shape); print(D.groupby(['train', 'target']).seed.nunique().to_string())
