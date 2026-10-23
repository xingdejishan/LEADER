"""The pose-free channel: global DINOv2 CLS top-20 reference ids for each frame.

Same embedding and same reference matrix as the deployed visual expert's retrieval
(`visual_branch.VisualBranch.embed` / `.reference_cls`), and the same call the pose-free
study used (`band5_nopose.retrieve_cls_only`).  Nothing here reads a pose: the only input
is the frame's camera image, the output is a ranked list of library reference ids.

    python mw_cls.py --gpu 3 --out /tmp/mw/cls/cls-ids.npz
"""
import argparse
import json
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
ap.add_argument('--scratch', default='/tmp/mw/cls/vb')
ap.add_argument('--k', type=int, default=20)
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
import band5_nopose  # noqa: E402  (sets up the frozen branch)

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'

subset = np.load(a.subset, allow_pickle=True)
keys = [str(k) for k in subset['keys']]
n = len(keys) if a.limit <= 0 else min(a.limit, len(keys))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
os.makedirs(a.scratch, exist_ok=True)

branch = band5_nopose.build('B20', a.scratch, 'cuda')
ids = np.zeros((n, a.k), dtype=np.int64)
cos = np.zeros((n, a.k), dtype=np.float64)
started = time.time()
for i in range(n):
    date, stamp = keys[i].split('/')
    image = os.path.join(DATA, 'NCLT', date, 'cam5', stamp + '.jpg')
    frame_ids, detail = band5_nopose.retrieve_cls_only(branch, image, a.k)
    ids[i] = frame_ids
    cos[i] = detail['visual_cosine']
    if (i + 1) % 20 == 0:
        print('%d/%d  %.0fs' % (i + 1, n, time.time() - started), flush=True)
np.savez_compressed(a.out, keys=np.array(keys[:n]), ids=ids, cosine=cos, k=a.k)
print('wrote %s  frames=%d  wall=%.0fs  median cosine top1=%.4f top20=%.4f'
      % (a.out, n, time.time() - started, float(np.median(cos[:, 0])),
         float(np.median(cos[:, -1]))))
