"""Patch encoders used by the pilot. H-optimus-1 is the default (SPARC's encoder); H-optimus-0 reproduces the earlier runs.

Select with the environment variable SPARC_ENCODER=h1|h0 (default h1). Feature files are written to and read from
features/<tag>_<suffix>/, e.g. features/hoptimus1_crc/ (suffix '' gives features/<tag>/ for HEST-Bench).
Both models are ViT-g/14 (1536-d CLS output), run at 224 px with timm's interpolated position embedding.
"""
import json
import os

ENCODERS = {
    'h1': dict(repo='bioptimus/H-optimus-1', dir='models/H-optimus-1', weights='model.safetensors', tag='hoptimus1'),
    'h0': dict(repo='bioptimus/H-optimus-0', dir='models/H-optimus-0', weights='pytorch_model.bin', tag='hoptimus0'),
}
# normalisation constants published with both models (identical for H-optimus-0 and H-optimus-1)
MEAN = (0.707223, 0.578729, 0.703617)
STD = (0.211883, 0.230117, 0.177517)


def name(encoder=None):
    e = (encoder or os.environ.get('SPARC_ENCODER', 'h1')).lower()
    if e not in ENCODERS:
        raise ValueError(f'unknown encoder {e!r}; choose one of {sorted(ENCODERS)}')
    return e


def feat_dir(suffix='crc', encoder=None, root='.'):
    tag = ENCODERS[name(encoder)]['tag']
    return os.path.join(root, 'features', f'{tag}_{suffix}' if suffix else tag)


def load(encoder=None, device='cuda'):
    """Return (model, mean, std) with mean/std as (1, 3, 1, 1) tensors on `device`."""
    import timm
    import torch
    e = ENCODERS[name(encoder)]
    cfg = json.load(open(os.path.join(e['dir'], 'config.json')))
    model = timm.create_model(cfg['architecture'], pretrained=True, num_classes=0, init_values=1e-5, dynamic_img_size=False, img_size=224,
                              pretrained_cfg_overlay=dict(file=os.path.join(e['dir'], e['weights']))).to(device).eval()
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return model, mean, std
