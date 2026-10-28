"""Patch features (1536-d, fp16) for every HEST patch file in a directory, with the encoder selected by SPARC_ENCODER (default h1).
Used for the H-optimus-1 runs: python extract_feats_dir.py <patch_dir> <out_dir> [ids.txt]"""
import glob, os, sys, h5py, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import encoders as E

src, out_dir = sys.argv[1], sys.argv[2]; os.makedirs(out_dir, exist_ok=True)
ids = set(open(sys.argv[3]).read().split()) if len(sys.argv) > 3 else None
model, mean, std = E.load()
for f in sorted(glob.glob(f'{src}/*.h5')):
    sid = os.path.basename(f)[:-3]; out = f'{out_dir}/{sid}.h5'
    if (ids is not None and sid not in ids) or os.path.exists(out): continue
    t = time.time()
    with h5py.File(f) as h:
        n = h['img'].shape[0]; feats = np.zeros((n, 1536), np.float16)
        for s in range(0, n, 256):
            x = torch.from_numpy(h['img'][s:s + 256]).cuda().permute(0, 3, 1, 2).float() / 255.
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                feats[s:s + 256] = model((x - mean) / std).float().cpu().numpy()
        bc = h['barcode'][:]; xy = h['coords'][:]
    assert np.isfinite(feats).all(), sid
    with h5py.File(out + '.tmp', 'w') as o:
        o['features'] = feats; o['coords'] = xy; o['barcode'] = bc
    os.replace(out + '.tmp', out)
    print(sid, n, f'{n / (time.time() - t):.0f}/s', flush=True)
