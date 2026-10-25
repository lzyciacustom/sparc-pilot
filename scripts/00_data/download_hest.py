"""Download HEST data used by the pilot (Hugging Face; requires accepting the HEST terms and HF_TOKEN in the environment).
Run from the flat workspace created by scripts/setup_workspace.sh.

  python download_hest.py meta     # HEST_v1_2_0.csv metadata table -> hf_probe/MahmoodLab_hest/ ; human_gene_db.parquet -> data/hest1k_crc/
  python download_hest.py bench    # HEST-Bench (MahmoodLab/hest-bench) -> data/hest_bench   (pilot stage 1)
  python download_hest.py crc      # 58 CRC test slides (patches, st, metadata) -> data/hest1k_crc
  python download_hest.py pool     # HEST-1k human Visium training pool -> data/hest1k_crc   (~155 GB with crc)
  python download_hest.py mouse    # 27 mouse slides -> data/hest1k_mouse
"""
import os, sys, pandas as pd
from huggingface_hub import snapshot_download, hf_hub_download
HEST_REV = '7e8d5a0b0aace41d8c8ec0f6ecea80e4ad2a61ec'   # MahmoodLab/hest revision used in the pilot
TOK = os.environ.get('HF_TOKEN')
def slides(ids, out):
    pats = [f'{k}/{i}.{e}' for i in ids for k, e in [('patches', 'h5'), ('st', 'h5ad'), ('metadata', 'json')]]
    snapshot_download('MahmoodLab/hest', repo_type='dataset', revision=HEST_REV, local_dir=out, allow_patterns=pats, token=TOK, max_workers=8)
what = sys.argv[1]
if what == 'meta':
    hf_hub_download('MahmoodLab/hest', 'HEST_v1_2_0.csv', repo_type='dataset', revision=HEST_REV, local_dir='hf_probe/MahmoodLab_hest', token=TOK)
    hf_hub_download('MahmoodLab/hest', 'human_gene_db.parquet', repo_type='dataset', revision=HEST_REV, local_dir='data/hest1k_crc', token=TOK)
elif what == 'bench':
    snapshot_download('MahmoodLab/hest-bench', repo_type='dataset', local_dir='data/hest_bench', max_workers=8, ignore_patterns=['*.png', 'fm_v1/*'], token=TOK)
elif what == 'crc':
    slides(sorted(pd.read_csv('crc_folds.csv').id), 'data/hest1k_crc')
elif what == 'pool':
    slides(sorted(pd.read_csv('hest1k_train_pool.csv').id), 'data/hest1k_crc')
elif what == 'mouse':
    slides(sorted(pd.read_csv('mouse_test_slides_meta_all.csv').id), 'data/hest1k_mouse')
else:
    sys.exit(__doc__)
