import json, os, sys
from pathlib import Path
import numpy as np

# 1) cv2 / torch availability in the leader env
try:
    import cv2
    print('cv2', cv2.__version__)
except Exception as e:
    print('cv2 MISSING', e)
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())

# 2) dinov2 assets
G = Path('/data2/zhangcy/v7-dinov2-recall-20261012-v1')
S = Path('/home/zhangcy/projects/v7-dinov2-recall-20261012-v1')
print('dinov2 data files:', sorted(os.listdir(G))[:12])
print('weights exists', (G/'dinov2_vitb14_pretrain.pth').exists(), (G/'dinov2_vitb14_pretrain.pth').stat().st_size)
print('source exists', (S/'dinov2-source').exists())
print('assets-freeze', json.loads((G/'assets-freeze.json').read_text()) if (G/'assets-freeze.json').exists() else 'MISSING')
ref_cls = np.load(G/'reference-cls.npy')
print('reference-cls', ref_cls.shape, ref_cls.dtype, float(np.linalg.norm(ref_cls[0])))

# 3) offline diag with ambiguity ratios
R = Path('/home/zhangcy/projects/e2e-moe-20261020-v1/results')
diags = [p for p in R.iterdir() if 'diag' in p.name]
print('diag files:', [p.name for p in diags][:10])
for p in diags[:3]:
    with p.open() as fh:
        first = fh.readline()
    print(p.name, first[:300])
