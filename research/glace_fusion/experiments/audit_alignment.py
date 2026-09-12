import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

root = Path('/root/rivermind-data/glace_nclt_corrected_20260912')
vendor = root / 'vendor_corrected'
sys.path.insert(0, '/root/rivermind-data/LEADER-v1-glace-independent')
sys.path.insert(0, str(vendor))
from research.glace_fusion.glace_adapter import deit_global_feature_fn, pixel_grid_uv
from research.glace_fusion.nclt_camera import preprocess_image
from ace_network import Regressor
from ace_util import get_pixel_grid
from dataset import CamLocDataset

torch.set_num_threads(2)
scene = root / 'scene/train'
files = sorted((scene / 'rgb').iterdir())
stems = [p.stem for p in files]
features = np.load(scene / 'features.npy', mmap_mode='r')
report = {'images': len(files), 'features_shape': list(features.shape),
          'pose_stems_match': stems == [p.stem for p in sorted((scene / 'poses').iterdir())],
          'calibration_stems_match': stems == [p.stem for p in sorted((scene / 'calibration').iterdir())],
          'all_features_finite': bool(np.isfinite(features).all()),
          'feature_norm_range': [float(x) for x in [np.linalg.norm(features,axis=1).min(), np.linalg.norm(features,axis=1).max()]]}
fn = deit_global_feature_fn(vendor, '/root/rivermind-data/LEADER-v1-visual-glace/research/visual_glace/CVPR23_DeitS_Rerank.pth')
batch = fn.batch
closure = dict(zip(batch.__code__.co_freevars, [c.cell_contents for c in batch.__closure__]))
model, transform = closure['model'], closure['transform']
ds = CamLocDataset(scene, mode=0, use_half=False, image_height=616, augment=False)
encoder = Regressor.create_from_encoder(torch.load(vendor / 'ace_encoder_pretrained.pt',map_location='cpu'),torch.zeros(3),1,True).cuda().eval()
rows = []
rgb_features = []
gray_features = []
positions = []
for ix in np.linspace(0,len(files)-1,64,dtype=int):
    path = files[ix]
    K0 = np.loadtxt(scene / 'calibration' / (path.stem+'.txt'))
    gray,K = preprocess_image(path,K0,616)
    item = ds[int(ix)]
    fresh = fn(gray)
    with Image.open(path) as im, torch.inference_mode():
        colored = model(transform(im.convert('RGB'))[None].cuda()).cpu().numpy()[0]
        local = encoder.get_features(item[0][None].cuda())
        local2 = encoder.get_features(torch.from_numpy((gray-.4)/.25)[None,None].cuda())
    h,w = local.shape[-2:]
    grid = get_pixel_grid(8)[:,:h,:w].numpy().reshape(2,-1).T
    rows.append({'index':int(ix),'image':path.stem,'feature_max_diff':float(np.max(np.abs(fresh-features[ix]))),
                 'rgb_gray_cosine':float(colored@fresh), 'pixel_max_diff':float(np.max(np.abs(item[0].numpy()[0]-(gray-.4)/.25))),
                 'K_max_diff':float(np.max(np.abs(item[4].numpy()-K))),
                 'local_feature_max_diff':float((local-local2).abs().max()),
                 'grid_max_diff':float(np.max(np.abs(grid-pixel_grid_uv(8,h,w))))})
    rgb_features.append(colored);gray_features.append(fresh);positions.append(item[2][:3,3].numpy())
    print(json.dumps(rows[-1]),flush=True)
report['rows']=rows
report['rgb_gray_cosine_median']=float(np.median([r['rgb_gray_cosine'] for r in rows]))
positions=np.asarray(positions)
dist=np.linalg.norm(positions[:,None]-positions[None],axis=2)
report['retrieval_probe_scope']='64 sparse training images; descriptive diagnostic, not localization accuracy'
for name, feats in [('rgb',rgb_features),('gray',gray_features)]:
    sim=np.asarray(feats)@np.asarray(feats).T
    np.fill_diagonal(sim,-np.inf)
    nearest=sim.argmax(axis=1)
    report[name+'_top1_pose_distance_median']=float(np.median(dist[np.arange(64),nearest]))
    report[name+'_top1_within_20m']=float(np.mean(dist[np.arange(64),nearest]<20))
(root/'alignment_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
