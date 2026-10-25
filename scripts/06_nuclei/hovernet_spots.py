"""HoVer-Net (PanNuke, TIAToolbox) on the central 41 um of each sampled spot patch (224 px @0.5 um -> x2 upsample, centre 256 px).
Outputs nuclei/<sid>.parquet: barcode, n_nuc, nuc_density (per 1e4 um^2), type fractions (PanNuke: 1 neoplastic, 2 inflammatory, 3 connective, 4 dead, 5 epithelial), mean area."""
import os, sys, h5py, numpy as np, pandas as pd, torch, cv2, time
os.environ['TQDM_DISABLE'] = '1'
from multiprocessing import Pool
from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.architecture.hovernet import HoVerNet
FRAC = float(os.environ.get('FRAC', '0.3')); BS = 32; AREA_UM2 = (164 * 0.25) ** 2
_model = None
def post(args):
    np_, hv, tp = args
    r = _model.postproc([np_, hv, tp])[0]; inst = r['predictions']; info = r['info_dict']
    lab = np.unique(inst); lab = lab[lab > 0]
    a = np.bincount(inst.ravel())[lab] * 0.25 ** 2 if len(lab) else np.array([])
    t = np.asarray(info.get('type', [])).ravel() if isinstance(info, dict) else np.array([])
    d = lab
    out = dict(n_nuc=len(d), nuc_density=len(d) / AREA_UM2 * 1e4, nuc_mean_area=float(a.mean()) if len(a) else np.nan)
    for k, nm in enumerate(['neoplastic', 'inflammatory', 'connective', 'dead', 'epithelial'], 1): out[f'frac_{nm}'] = float((t == k).mean()) if len(t) else np.nan
    return out
def init():
    global _model; _model, _ = get_pretrained_model('hovernet_fast-pannuke')
if __name__ == '__main__':
    os.makedirs('nuclei', exist_ok=True)
    ids = sys.argv[1:]
    m, _ = get_pretrained_model('hovernet_fast-pannuke'); m = m.cuda().eval()
    pool = Pool(24, initializer=init)
    for sid in ids:
        out = f'nuclei/{sid}.parquet'
        if os.path.exists(out): continue
        t0 = time.time()
        with h5py.File(f'data/hest1k_crc/patches/{sid}.h5') as f:
            n = f['img'].shape[0]; idx = np.sort(np.random.default_rng(0).choice(n, max(1, int(FRAC * n)), replace=False))
            img = f['img'][idx]; bc = (f['barcode'] if 'barcode' in f else f['barcodes'])[idx]
        bc = [b[0].decode() if isinstance(b[0], bytes) else str(b[0]) for b in bc]
        rows = []
        for s in range(0, len(img), BS):
            up = np.stack([cv2.resize(x, (448, 448), interpolation=cv2.INTER_CUBIC)[96:352, 96:352] for x in img[s:s + BS]])
            with torch.no_grad(): o = HoVerNet.infer_batch(m, torch.from_numpy(up), device='cuda')
            rows += pool.map(post, [(o[0][i], o[1][i], o[2][i]) for i in range(len(up))])
        df = pd.DataFrame(rows); df.insert(0, 'barcode', bc); df.to_parquet(out + '.tmp', index=False); os.replace(out + '.tmp', out)
        print(sid, len(df), f'{len(df)/(time.time()-t0):.1f} spots/s', round(df.n_nuc.mean(), 1), flush=True)
