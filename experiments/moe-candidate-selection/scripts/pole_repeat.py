"""Landmark repeatability: the same pole seen in two nearby scans 鈥?how far apart?

This is the quantity that decides whether pole landmarks can be sharper than point
matching: if a pole's fitted centre moves by 0.2 m between scans, its bearing at 15 m
has ~0.8 deg of noise; if it moves by 0.8 m, the idea is dead regardless of the solver.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(MOE, 'stages/fast_search'))
sys.path.insert(0, MOE)
import fast_lib as F

DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001/NCLT'
CELL = 0.2
RANGE = 35.0
MIN_EXTENT = 1.5
MAX_FOOT = 0.8
MIN_CELLS = 4
GROUND_BAND = 0.25


def ground_z(points):
    key = (np.floor(points[:, 0]).astype(np.int64) * 100000
           + np.floor(points[:, 1]).astype(np.int64))
    order = np.argsort(key)
    ks = key[order]
    zs = points[order, 2]
    uniq, start = np.unique(ks, return_index=True)
    bounds = np.append(start, len(ks))
    g = np.empty(len(uniq))
    for k in range(len(uniq)):
        seg = zs[bounds[k]:bounds[k + 1]]
        g[k] = np.percentile(seg, 10) if len(seg) >= 4 else seg.min()
    lut = dict(zip(uniq.tolist(), g.tolist()))
    return np.array([lut[k] for k in key])


def poles_of(points):
    if len(points) < 200:
        return np.zeros((0, 3))
    pts = points[(points[:, 2] - ground_z(points)) > GROUND_BAND]
    if len(pts) < 100:
        return np.zeros((0, 3))
    ij = np.floor(pts[:, :2] / CELL).astype(np.int64)
    ij -= ij.min(0)
    key = ij[:, 0] * 100000 + ij[:, 1]
    uniq, inv = np.unique(key, return_inverse=True)
    zmin = np.full(len(uniq), np.inf); zmax = np.full(len(uniq), -np.inf)
    cnt = np.zeros(len(uniq), int)
    np.minimum.at(zmin, inv, pts[:, 2]); np.maximum.at(zmax, inv, pts[:, 2])
    np.add.at(cnt, inv, 1)
    ok = ((zmax - zmin) >= MIN_EXTENT) & (cnt >= 5)
    if ok.sum() == 0:
        return np.zeros((0, 3))
    gi = np.zeros((ij[:, 0].max() + 2, ij[:, 1].max() + 2), bool)
    ci, cj = np.divmod(uniq, 100000)
    gi[ci[ok], cj[ok]] = True
    lab, n = ndimage.label(gi, structure=np.ones((3, 3), bool))
    out = []
    for k in range(1, n + 1):
        m = lab[ci, cj] == k
        if m.sum() < MIN_CELLS:
            continue
        sub = pts[np.isin(inv, np.flatnonzero(ok & m))]
        if len(sub) < 15:
            continue
        if np.linalg.norm(sub[:, :2].max(0) - sub[:, :2].min(0)) > MAX_FOOT:
            continue
        out.append([sub[:, 0].mean(), sub[:, 1].mean(), float(sub[:, 2].max() - sub[:, 2].min())])
    return np.asarray(out) if out else np.zeros((0, 3))


def scan_path_of(entry, key):
    path = entry['scan_path']
    if path.endswith('/'):
        return path + key.split('/')[1] + '.bin'
    return path


def main():
    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    poses = [np.asarray(r['world_from_body'], float) for r in refs]
    pos = np.asarray([T[:3, 3] for T in poses])
    keys = [str(r['key']) for r in refs]
    dates = np.array([k.split('/')[0] for k in keys])
    stamps = np.array([int(k.split('/')[1]) for k in keys], dtype=np.int64)
    tree = cKDTree(pos)

    rng = np.random.default_rng(61)
    pairs = []
    for i in rng.choice(len(refs), 200, replace=False):
        _, nb = tree.query(pos[i], k=3)
        for j in np.atleast_1d(nb)[1:]:
            j = int(j)
            if dates[i] != dates[j]:
                continue
            if abs(stamps[i] - stamps[j]) / 1e6 > 6.0:
                continue
            d = np.linalg.norm(pos[i] - pos[j])
            if 1.5 < d < 6.0:
                pairs.append((i, j))
        if len(pairs) >= 25:
            break

    diffs, counts = [], []
    for i, j in pairs:
        Pi = poses[i]; Pj = poses[j]
        pi = poles_of(F.decode(scan_path_of(refs[i], keys[i])))
        pj = poles_of(F.decode(scan_path_of(refs[j], keys[j])))
        if len(pi) < 2 or len(pj) < 2:
            continue
        wi = pi[:, :2] @ Pi[:3, :3][:2, :2].T + Pi[:2, 3]
        wj = pj[:, :2] @ Pj[:3, :3][:2, :2].T + Pj[:2, 3]
        t = cKDTree(wj)
        d, idx = t.query(wi, k=1)
        keep = d < 1.0
        if keep.sum() == 0:
            continue
        diffs.extend(d[keep].tolist())
        counts.append((int(keep.sum()), len(wi), len(wj)))

    d = np.array(diffs)
    print('reference pairs: %d   matched poles total: %d' % (len(counts), len(d)))
    if len(d):
        print('same-pole world position difference:  median %.3f m   mean %.3f   p75 %.3f   p90 %.3f'
              % (np.median(d), d.mean(), np.percentile(d, 75), np.percentile(d, 90)))
        print('  share within 0.2 m: %.0f%%   within 0.5 m: %.0f%%'
              % (100 * (d <= 0.2).mean(), 100 * (d <= 0.5).mean()))
    if counts:
        c = np.array(counts)
        print('per-pair: matched %d of %d and %d detected' % (int(np.median(c[:, 0])),
                                                              int(np.median(c[:, 1])),
                                                              int(np.median(c[:, 2]))))


if __name__ == '__main__':
    main()
