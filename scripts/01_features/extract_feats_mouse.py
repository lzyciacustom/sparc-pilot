import glob, os, json, h5py, numpy as np, torch, timm, time
cfg=json.load(open('models/H-optimus-0/config.json'))
model=timm.create_model(cfg['architecture'], pretrained=True, num_classes=0, init_values=1e-5, dynamic_img_size=False, img_size=224,
        pretrained_cfg_overlay=dict(file='models/H-optimus-0/pytorch_model.bin')).cuda().eval()
mean=torch.tensor([0.707223,0.578729,0.703617],device='cuda').view(1,3,1,1); std=torch.tensor([0.211883,0.230117,0.177517],device='cuda').view(1,3,1,1)
import pandas as pd
crc=set(pd.read_csv('hest1k_crc_candidates.csv').id)
files=sorted(glob.glob('data/hest1k_mouse/patches/*.h5'), key=lambda f:(os.path.basename(f)[:-3] not in crc, os.path.getsize(f)))
files=list(files)
for f in files:
    sid=os.path.basename(f)[:-3]; out=f'features/hoptimus0_mouse/{sid}.h5'
    if os.path.exists(out): continue
    t=time.time()
    with h5py.File(f) as h:
        n=h['img'].shape[0]; feats=np.zeros((n,1536),np.float16)
        for s in range(0,n,256):
            x=torch.from_numpy(h['img'][s:s+256]).cuda().permute(0,3,1,2).float()/255.
            x=(x-mean)/std
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
                feats[s:s+256]=model(x).float().cpu().numpy()
        bc=h['barcode' if 'barcode' in h else 'barcodes'][:]; xy=h['coords'][:]
    with h5py.File(out+'.tmp','w') as o:
        o['features']=feats; o['coords']=xy; o['barcode']=bc
    os.replace(out+'.tmp',out)
    print(sid, n, f'{n/(time.time()-t):.0f}/s', flush=True)
