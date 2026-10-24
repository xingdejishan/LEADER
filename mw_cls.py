"""The pose-free channel: global DINOv2 CLS top-20 reference ids for each frame.

This is the multi-window study's own window-nomination channel, kept self-contained so
it does not reach into the visual expert: it carries its own copy of the frozen DINOv2
CLS embedding and of the frozen reference matrix, and nothing else.  Nothing here reads
a pose; the only input is the frame's camera image, the output is a ranked list of
library reference ids.

It is **not** the visual expert's reference selection any more.  That rule is SALAD
global cosine top-5 and lives in `salad_retrieval.py` (see `visual_branch.retrieve`).
The two-channel rule this file once shared (CLS top-5 union the 5 references nearest a
pose prior) was deleted with the pose prior itself.

    python mw_cls.py --gpu 3 --out /tmp/mw/cls/cls-ids.npz
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--gpu', default='3')
ap.add_argument('--subset', default=os.path.join(HERE, 'results/band5-nopose-subset.npz'))
ap.add_argument('--out', default='/tmp/mw/cls/cls-ids.npz')
ap.add_argument('--k', type=int, default=20)
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu

import torch                                                          # noqa: E402
from PIL import Image                                                # noqa: E402

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
CLS_ROOT = '/home/zhangcy/projects/v7-dinov2-recall-20261012-v1'
CLS_DATA = '/data2/zhangcy/v7-dinov2-recall-20261012-v1'
IMAGE_EDGE = 560


def model_image(path):
    """The frozen DINOv2 pipeline: upright, short side pinned, /14 crops."""
    with Image.open(path) as image:
        image = image.convert('RGB').transpose(Image.Transpose.ROTATE_270)
        w, h = image.size
        scale = min(1., IMAGE_EDGE / max(w, h))
        rw, rh = round(w * scale), round(h * scale)
        image = image.resize((rw, rh), Image.Resampling.BICUBIC)
        cw, ch = rw // 14 * 14, rh // 14 * 14
        left, top = (rw - cw) // 2, (rh - ch) // 2
        return np.asarray(image.crop((left, top, left + cw, top + ch))).copy()


def _unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


class ClsTable(object):
    """The frozen DINOv2 CLS embedding and the frozen reference matrix, nothing else."""

    def __init__(self):
        self.model = torch.hub.load(os.path.join(CLS_ROOT, 'dinov2-source'),
                                    'dinov2_vitb14', source='local', pretrained=False)
        self.model.load_state_dict(torch.load(
            os.path.join(CLS_DATA, 'dinov2_vitb14_pretrain.pth'), map_location='cpu'))
        self.model = self.model.eval().requires_grad_(False).cuda()
        self.reference = _unit(np.load(
            os.path.join(CLS_DATA, 'reference-cls.npy')).astype(np.float32))

    def ranking(self, image_path, k):
        x = torch.from_numpy(model_image(image_path)).permute(2, 0, 1).float() / 255
        x = ((x - torch.tensor([.485, .456, .406])[:, None, None])
             / torch.tensor([.229, .224, .225])[:, None, None])
        with torch.inference_mode():
            g = self.model.forward_features(x[None].cuda())['x_norm_clstoken'][0]
        query = _unit(g.float().cpu().numpy())
        similarity = self.reference @ query
        ids = [int(i) for i in np.argsort(-similarity)[:int(k)]]
        return ids, [float(similarity[i]) for i in ids]


subset = np.load(a.subset, allow_pickle=True)
keys = [str(k) for k in subset['keys']]
n = len(keys) if a.limit <= 0 else min(a.limit, len(keys))
os.makedirs(os.path.dirname(a.out), exist_ok=True)

table = ClsTable()
ids = np.zeros((n, a.k), dtype=np.int64)
cos = np.zeros((n, a.k), dtype=np.float64)
started = time.time()
for i in range(n):
    date, stamp = keys[i].split('/')
    image = os.path.join(DATA, 'NCLT', date, 'cam5', stamp + '.jpg')
    frame_ids, frame_cos = table.ranking(image, a.k)
    ids[i] = frame_ids
    cos[i] = frame_cos
    if (i + 1) % 20 == 0:
        print('%d/%d  %.0fs' % (i + 1, n, time.time() - started), flush=True)
np.savez_compressed(a.out, keys=np.array(keys[:n]), ids=ids, cosine=cos, k=a.k)
print('wrote %s  frames=%d  wall=%.0fs  median cosine top1=%.4f top20=%.4f'
      % (a.out, n, time.time() - started, float(np.median(cos[:, 0])),
         float(np.median(cos[:, -1]))))
