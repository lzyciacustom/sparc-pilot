"""Fast per-spot nuclear features from HEST 224px patches (0.5 um/px): haematoxylin colour deconvolution, per-slide Otsu,
distance-transform watershed; features inside the 55 um spot disk (radius 55 px)."""
import sys, os, h5py, numpy as np, pandas as pd
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import regionprops, label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from multiprocessing import Pool
MPP = 0.5; R = 55; yy, xx = np.mgrid[:224, :224]; DISK = (yy - 112) ** 2 + (xx - 112) ** 2 <= R ** 2
def hchan(img): return gaussian(rgb2hed(img)[..., 0], 1.0)
def feats(h, thr, img=None):
    tissue = img.mean(-1) < 220 if img is not None else np.ones_like(h, bool)
    m = (h > thr) & DISK & tissue
    m = ndi.binary_opening(m, iterations=1); m = ndi.binary_fill_holes(m)
    dist = ndi.distance_transform_edt(m)
    pk = peak_local_max(dist, min_distance=5, labels=label(m), exclude_border=False)
    mk = np.zeros_like(dist, int); mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = watershed(-dist, mk, mask=m)
    rp = [r for r in regionprops(lab) if 10 <= r.area * MPP ** 2 <= 150 and not (r.eccentricity < 0.3 and r.solidity > 0.97 and r.area * MPP ** 2 > 40)]
    a = np.array([r.area * MPP ** 2 for r in rp]); e = np.array([r.eccentricity for r in rp])
    n = len(rp); disk_area = DISK.sum() * MPP ** 2
    if n == 0: return dict(n_nuc=0, nuc_density=0., nuc_area_frac=float(m.sum() / DISK.sum()), nuc_mean_area=np.nan, frac_small_round=np.nan, frac_elongated=np.nan, frac_large=np.nan, h_mean=float(h[DISK].mean()))
    return dict(n_nuc=n, nuc_density=n / disk_area * 1e4, nuc_area_frac=float(m.sum() / DISK.sum()), nuc_mean_area=float(a.mean()),
                frac_small_round=float(((a < 30) & (e < 0.75)).mean()), frac_elongated=float((e > 0.9).mean()), frac_large=float((a > 60).mean()),
                h_mean=float(h[DISK].mean()))
def run(sid):
    out = f'nuclei/{sid}.parquet'
    if os.path.exists(out): return sid
    with h5py.File(f'data/hest1k_crc/patches/{sid}.h5') as f:
        img = f['img'][:]; bc = f['barcode'][:] if 'barcode' in f else f['barcodes'][:]
    bc = [b[0].decode() if isinstance(b[0], bytes) else str(b[0]) for b in bc]
    rng = np.random.default_rng(0); samp = rng.choice(len(img), min(200, len(img)), replace=False)
    thr = threshold_otsu(np.concatenate([hchan(img[i])[DISK & (img[i].mean(-1) < 220)] for i in samp]))
    rows = [feats(hchan(x), thr, x) for x in img]
    df = pd.DataFrame(rows); df.insert(0, 'barcode', bc); df['thr'] = thr
    df.to_parquet(out + '.tmp', index=False); os.replace(out + '.tmp', out); return sid
if __name__ == '__main__':
    os.makedirs('nuclei', exist_ok=True)
    ids = sys.argv[1:] or pd.read_csv('crc_folds.csv').id.tolist()
    with Pool(min(30, len(ids))) as p:
        for s in p.imap_unordered(run, ids): print(s, flush=True)
