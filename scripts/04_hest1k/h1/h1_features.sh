#!/bin/bash
# H-optimus-1 features for the 58 CRC test slides and the HEST-1k training pool (run from the flat workspace; needs HF_TOKEN).
# Pool patches are fetched in batches of 20, embedded, then deleted to keep disk use low (~84 patches/s on one RTX 3090).
set -euo pipefail
export SPARC_ENCODER=h1
python download_models.py h1
python - <<'PY'
import os, subprocess, pandas as pd
from huggingface_hub import snapshot_download
REV = '7e8d5a0b0aace41d8c8ec0f6ecea80e4ad2a61ec'; tok = os.environ['HF_TOKEN']; FD = 'features/hoptimus1_crc'
crc = sorted(pd.read_csv('crc_folds.csv').id)
snapshot_download('MahmoodLab/hest', repo_type='dataset', revision=REV, local_dir='data/hest1k_crc', allow_patterns=[f'patches/{i}.h5' for i in crc], token=tok, max_workers=8)
open('crc_ids.txt', 'w').write('\n'.join(crc))
subprocess.run(['python', 'extract_feats_dir.py', 'data/hest1k_crc/patches', FD, 'crc_ids.txt'], check=True)
pool = [i for i in sorted(pd.read_csv('hest1k_train_pool.csv').id) if i not in set(crc) and not os.path.exists(f'{FD}/{i}.h5')]
os.makedirs('data/pool_tmp/patches', exist_ok=True)
for b in range(0, len(pool), 20):
    batch = pool[b:b + 20]
    snapshot_download('MahmoodLab/hest', repo_type='dataset', revision=REV, local_dir='data/pool_tmp', allow_patterns=[f'patches/{i}.h5' for i in batch], token=tok, max_workers=8)
    open('batch_ids.txt', 'w').write('\n'.join(batch))
    subprocess.run(['python', 'extract_feats_dir.py', 'data/pool_tmp/patches', FD, 'batch_ids.txt'], check=True)
    for i in batch:
        p = f'data/pool_tmp/patches/{i}.h5'
        if os.path.exists(p) and os.path.exists(f'{FD}/{i}.h5'): os.remove(p)
    print(f'pool {min(b + 20, len(pool))}/{len(pool)}', flush=True)
PY
echo "features: $(ls features/hoptimus1_crc/*.h5 | wc -l) slides"
