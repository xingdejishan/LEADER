"""Diagnose the pole detector: scan z convention and cluster geometry."""
import json
import os
import sys

import numpy as np
from scipy import ndimage

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(MOE, 'stages/fast_search'))
sys.path.insert(0, MOE)
import fast_lib as F

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001/NCLT'
VOX = 0.30


def clusters(points):
    origin = points.min(0)
    idx = np.floor((points - origin) / VOX).astype(np.int32)
    grid = np.zeros(idx.max(0) + 1, bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3), bool))
    pt_lab = lab[idx[:, 0], idx[:, 1], idx[:, 2]]
    order = np.argsort(pt_lab)
    lab_sorted = pt_lab[order]
    pts_sorted = points[order]
    bounds = np.searchsorted(lab_sorted, np.arange(1, n + 2))
    out = []
    for k in range(n):
        a, b = bounds[k], bounds[k + 1]
        if b - a < 10:
            continue
        seg = pts_sorted[a:b]
        zr = float(seg[:, 2].max() - seg[:, 2].min())
        xy = seg[:, :2]
        w = float(np.linalg.norm(xy.max(0) - xy.min(0)))
        r = float(np.median(np.linalg.norm(xy, axis=1)))
        out.append((b - a, zr, w, r))
    return np.asarray(out)


def main():
    frames = json.load(open(os.path.join(MOE, 'inputs/frames.json')))
    for i in (0, 500, 1000):
        date = frames[i]['key'].split('/')[0]
        stamp = frames[i]['key'].split('/')[1]
        path = os.path.join(DATA, date, 'velodyne_sync', stamp + '.bin')
        if not os.path.isfile(path):
            print(i, 'missing', path); continue
        p = F.decode(path)
        print('frame %d  points %d' % (i, len(p)))
        print('   x p1/50/99: %.1f %.1f %.1f   y: %.1f %.1f %.1f   z: %.1f %.1f %.1f'
              % (*np.percentile(p[:, 0], [1, 50, 99]), *np.percentile(p[:, 1], [1, 50, 99]),
                 *np.percentile(p[:, 2], [1, 50, 99])))
        near = p[np.linalg.norm(p[:, :2], axis=1) <= 35.0]
        c = clusters(near)
        if len(c) == 0:
            print('   no clusters'); continue
        print('   clusters (>=10 pts): %d' % len(c))
        tall = c[(c[:, 1] >= 1.2) & (c[:, 2] <= 1.0) & (c[:, 0] >= 25)]
        print('   passing pole filter (h>=1.2, w<=1.0, n>=25): %d' % len(tall))
        print('   top 5 by points: pts %s  height %s  width %s  range %s'
              % (c[np.argsort(-c[:, 0])][:5, 0].astype(int),
                 np.round(c[np.argsort(-c[:, 0])][:5, 1], 1),
                 np.round(c[np.argsort(-c[:, 0])][:5, 2], 1),
                 np.round(c[np.argsort(-c[:, 0])][:5, 3], 1)))
        print('   tallest 5: pts %s  height %s  width %s  range %s'
              % (c[np.argsort(-c[:, 1])][:5, 0].astype(int),
                 np.round(c[np.argsort(-c[:, 1])][:5, 1], 1),
                 np.round(c[np.argsort(-c[:, 1])][:5, 2], 1),
                 np.round(c[np.argsort(-c[:, 1])][:5, 3], 1)))


if __name__ == '__main__':
    main()
