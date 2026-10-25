"""H-optimus-0 weights (gated on Hugging Face: request access at https://huggingface.co/bioptimus/H-optimus-0, then set HF_TOKEN).
HoVer-Net (PanNuke) weights are fetched automatically by TIAToolbox on first use (TIACentre/TIAToolbox_pretrained_weights)."""
import os
from huggingface_hub import snapshot_download
snapshot_download('bioptimus/H-optimus-0', local_dir='models/H-optimus-0', token=os.environ.get('HF_TOKEN'))
