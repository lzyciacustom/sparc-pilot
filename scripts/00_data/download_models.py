"""Patch-encoder weights (gated on Hugging Face; request access on the model pages, then set HF_TOKEN).
  python download_models.py        # H-optimus-1 (default encoder)
  python download_models.py h0     # H-optimus-0 (reproduces the earlier runs; SPARC_ENCODER=h0)
HoVer-Net (PanNuke) weights are fetched automatically by TIAToolbox on first use (TIACentre/TIAToolbox_pretrained_weights)."""
import os, sys
from huggingface_hub import snapshot_download
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoders import ENCODERS, name

for e in (sys.argv[1:] or ['h1']):
    m = ENCODERS[name(e)]
    snapshot_download(m['repo'], local_dir=m['dir'], allow_patterns=['config.json', m['weights']], token=os.environ.get('HF_TOKEN'))
    print('downloaded', m['repo'], '->', m['dir'])
