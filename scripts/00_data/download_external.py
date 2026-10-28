"""Download the external cohorts used by scripts/09_composition into $SPARC_EXT (default <workspace>/ext) and <workspace>/data.
usage: python download_external.py [liver] [mdacc] [colonmap]   (default: all)
  liver    GSE115469 (MacParland et al. 2018 human liver atlas)                       -> ext/liver/   (~0.1 GB)
  mdacc    GSE294385 (Liu et al., Cancer Cell 2026; 37 FFPE CytAssist Visium sections)  -> ext/mdacc/   (~6 GB)
           + authors' pathology layers (github.com/yliuup/CRC_micromets_ST, Meta_data/visium_meta_after_qc.rds),
             exported to CSV with Rscript (needs R; md5 of the RDS used here: 0acee439145cb7f7e0d9722ac10d2749)
  colonmap COLON MAP Visium master h5ad files (Heiser et al., Cell 2023; OSF hftq2, ST_master) -> data/colonmap_osf/ (~5 GB);
           only spot metadata (obs) is kept for analysis, in colonmap_obs.parquet
The MDACC sample -> patient map (metadata/mdacc_patient_map.csv) was transcribed from the paper's Table S1."""
import os, sys, subprocess, tarfile, glob
import requests
WS = os.path.dirname(os.path.abspath(__file__))
EXT = os.environ.get('SPARC_EXT', os.path.join(WS, 'ext'))
GEO = 'https://ftp.ncbi.nlm.nih.gov/geo/series'
MDACC_META = 'https://raw.githubusercontent.com/yliuup/CRC_micromets_ST/5750ff04a2870eeb141653ece23592f5aea503f4/Meta_data/visium_meta_after_qc.rds'


def fetch(url, path):
    if os.path.exists(path): return
    os.makedirs(os.path.dirname(path), exist_ok=True); tmp = path + '.part'
    subprocess.run(['curl', '-sSL', '--fail', '-C', '-', '-o', tmp, url], check=True); os.replace(tmp, path)


def liver():
    for f in ['GSE115469_CellClusterType.txt.gz', 'GSE115469_Data.csv.gz']:
        fetch(f'{GEO}/GSE115nnn/GSE115469/suppl/{f}', f'{EXT}/liver/{f}')


def mdacc():
    D = f'{EXT}/mdacc'; os.makedirs(f'{D}/raw', exist_ok=True)
    if not glob.glob(f'{D}/raw/*.tar.gz'):
        fetch(f'{GEO}/GSE294nnn/GSE294385/suppl/GSE294385_RAW.tar', f'{D}/GSE294385_RAW.tar')
        tarfile.open(f'{D}/GSE294385_RAW.tar').extractall(f'{D}/raw'); os.remove(f'{D}/GSE294385_RAW.tar')
    fetch(MDACC_META, f'{D}/visium_meta_after_qc.rds')
    if not os.path.exists(f'{D}/visium_meta_after_qc.csv'):
        r = (f"m <- readRDS('{D}/visium_meta_after_qc.rds'); "
             f"write.csv(m[, c('Barcode', 'Layer1', 'Layer2', 'Layer3')], '{D}/visium_meta_after_qc.csv', row.names = FALSE)")
        subprocess.run(['Rscript', '-e', r], check=True)
    print('mdacc sections:', len(glob.glob(f'{D}/raw/*.tar.gz')))


def colonmap():
    import anndata as ad, pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    D = f'{WS}/data/colonmap_osf'; os.makedirs(D, exist_ok=True)
    u = 'https://api.osf.io/v2/nodes/hftq2/files/osfstorage/'
    st = [x for x in requests.get(u, timeout=120).json()['data'] if x['attributes']['name'] == 'ST_master'][0]
    files, url = [], st['relationships']['files']['links']['related']['href']
    while url:
        d = requests.get(url, timeout=120).json(); files += [(y['attributes']['name'], y['links']['download']) for y in d['data']]; url = d['links'].get('next')

    def get(nl):
        n, link = nl; p = f'{D}/{n}'; fetch(link, p)
        a = ad.read_h5ad(p, backed='r'); o = a.obs.copy(); o['master_file'] = n; o['barcode'] = o.index.astype(str); a.file.close(); return o
    with ThreadPoolExecutor(6) as ex: obs = list(ex.map(get, files))
    cols = ['barcode', 'master_file', 'Sample', 'Patient', 'Block ID', 'Tumor Type', 'Tumor Location', 'Tumor Stage', 'Tumor Grade',
            'CIN Status', 'Evolution', 'pathology_annotation', 'MILWRM Domain', 'in_tissue']
    O = pd.concat([o[[c for c in cols if c in o]] for o in obs]); O.to_parquet(f'{D}/colonmap_obs.parquet'); print('colonmap files:', len(files), O.shape)


if __name__ == '__main__':
    for job in (sys.argv[1:] or ['liver', 'mdacc', 'colonmap']): {'liver': liver, 'mdacc': mdacc, 'colonmap': colonmap}[job]()
