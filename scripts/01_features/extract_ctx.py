"""Multi-scale context features: paste each 224px@0.5um/px HEST patch (downsampled x4 -> 56px@2um/px) onto a slide
canvas at its level-0 coordinate, crop a 224x224 window (448 um) centred on each spot, embed with H-optimus-0.
Output features/hoptimus0_ctx/<sid>.h5 with features/coords/barcode in the SAME order as the patch file."""
import glob, os, json, h5py, numpy as np, torch, timm, time, pandas as pd, torch.nn.functional as F
cfg=json.load(open('models/H-optimus-0/config.json'))
model=timm.create_model(cfg['architecture'],pretrained=True,num_classes=0,init_values=1e-5,dynamic_img_size=False,img_size=224,
        pretrained_cfg_overlay=dict(file='models/H-optimus-0/pytorch_model.bin')).cuda().eval()
mean=torch.tensor([0.707223,0.578729,0.703617],device='cuda').view(1,3,1,1); std=torch.tensor([0.211883,0.230117,0.177517],device='cuda').view(1,3,1,1)
R='data/hest1k_crc'; crc=set(pd.read_csv('hest1k_crc_candidates.csv').id)
files=sorted(glob.glob(f'{R}/patches/*.h5'), key=lambda f:(os.path.basename(f)[:-3] not in crc, os.path.getsize(f)))
for f in files:
    sid=os.path.basename(f)[:-3]; out=f'features/hoptimus0_ctx/{sid}.h5'
    if os.path.exists(out): continue
    t=time.time(); mpp=json.load(open(f'{R}/metadata/{sid}.json'))['pixel_size_um_estimated']
    with h5py.File(f) as h:
        xy=h['coords'][:].astype(np.float64); bc=h['barcode'][:]; n=len(xy)
        um=xy*mpp
        o=um.min(0)-500; cx=((um-o)/2.0).astype(int)  # 2 um/px canvas coordinates of the patch top-left
        W,Hh=(cx.max(0)+56+300)
        canvas=torch.full((3,int(Hh),int(W)),240,dtype=torch.uint8)
        for s in range(0,n,2048):
            im=torch.from_numpy(h['img'][s:s+2048]).permute(0,3,1,2).float()
            sm=F.interpolate(im,size=(56,56),mode='area').round().clamp(0,255).to(torch.uint8)
            for j in range(sm.shape[0]):
                x0,y0=cx[s+j]; canvas[:,y0:y0+56,x0:x0+56]=sm[j]
    ctr=cx+28; feats=np.zeros((n,1536),np.float16)
    pad=F.pad(canvas.float().unsqueeze(0),(112,112,112,112),value=240.).squeeze(0).to(torch.uint8)
    for s in range(0,n,128):
        crops=torch.stack([pad[:,y:y+224,x:x+224] for x,y in ctr[s:s+128]])
        x=(crops.cuda().float()/255.-mean)/std
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
            feats[s:s+128]=model(x).float().cpu().numpy()
    with h5py.File(out+'.tmp','w') as o_:
        o_['features']=feats; o_['coords']=xy.astype(np.int64); o_['barcode']=bc
    os.replace(out+'.tmp',out); print(sid,n,f'{n/(time.time()-t):.0f}/s',flush=True)
