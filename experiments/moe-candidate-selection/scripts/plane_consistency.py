"""Plane-level consistency: do the walls in a scan agree with the walls in the map?

Instead of matching points, aggregate: estimate a normal per point, keep the ones on
vertical surfaces, and build the histogram of their azimuth.  The yaw shift that best
aligns a scan's histogram with the map's is a plane-level estimate.  Its spread over
many references is the plane-level inconsistency, directly comparable with the 0.56 deg
point-level number measured earlier.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(MOE, 'stages/fast_search'))
sys.path.insert(0, MOE)

import fast_lib as F
import pipeline as P

BINS = 720                 # 0.25 deg over 180 deg
RADIUS = 30.0
K = 12
N_SCAN = 3000
N_MAP = 6000
N_REF = 30
N_NEIGHBOUR = 8
SHIFT = np.arange(-20.0, 20.0001, 0.25)


def scan_file(entry):
    path = entry['scan_path']
    if path.endswith('/'):
        return path + str(entry['key']).split('/')[1] + '.bin'
    return path


def wall_hist(points):
    tree = cKDTree(points)
    _, idx = tree.query(points, k=K, workers=1)
    nb = points[idx]
    c = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', c, c) / K
    _, vec = np.linalg.eigh(cov)
    n = vec[:, :, 0]
    keep = np.abs(n[:, 2]) < 0.25                      # near-vertical surface
    nv = n[keep]
    if len(nv) < 50:
        return None, int(keep.sum())
    az = np.degrees(np.arctan2(nv[:, 1], nv[:, 0])) % 180.0
    hist, _ = np.histogram(az, bins=BINS, range=(0.0, 180.0))
    return hist.astype(float), int(keep.sum())


def best_shift(h1, h2):
    """circular shift of h1 that best matches h2, in degrees"""
    if h1 is None or h2 is None:
        return None
    a = h1 - h1.mean()
    b = h2 - h2.mean()
    scores = []
    for s in SHIFT:
        k = int(round(s / 0.25))
        scores.append(float(np.dot(np.roll(a, k), b)))
    scores = np.array(scores)
    i = int(np.argmax(scores))
    if 0 < i < len(scores) - 1:                        # sub-bin parabola
        denom = scores[i - 1] - 2 * scores[i] + scores[i + 1]
        off = 0.5 * (scores[i - 1] - scores[i + 1]) / denom if denom != 0 else 0.0
        return float(SHIFT[i] + off * 0.25)
    return float(SHIFT[i])


def main():
    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    cfg = P.Config()
    poses = [np.asarray(r['world_from_body'], float) for r in refs]
    pos = np.asarray([T[:3, 3] for T in poses], float)
    tree = cKDTree(pos)

    rng = np.random.default_rng(31)
    sample = rng.choice(len(refs), N_REF * 5, replace=False)
    deltas, kept = [], []
    for i in sample:
        if len(deltas) >= N_REF:
            break
        T = poses[i]
        t, R = T[:3, 3], T[:3, :3]
        _, nb = tree.query(t, k=N_NEIGHBOUR + 1)
        nb = [int(j) for j in np.atleast_1d(nb) if int(j) != int(i)]
        cloud = []
        for j in nb:
            Tj = poses[j]
            pts = F.decode(scan_file(refs[j]))
            cloud.append(pts @ Tj[:3, :3].T + Tj[:3, 3])
        cloud = np.vstack(cloud)
        cloud = cloud[np.linalg.norm(cloud - t, axis=1) <= RADIUS]
        if len(cloud) < 3000:
            continue
        rng2 = np.random.default_rng(i)
        mpts = cloud[rng2.choice(len(cloud), min(N_MAP, len(cloud)), replace=False)]

        q = F.decode(scan_file(refs[i]))
        qw = q @ R.T + t
        qw = qw[np.linalg.norm(qw - t, axis=1) <= RADIUS]
        if len(qw) < 1000:
            continue
        qs = qw[rng2.choice(len(qw), min(N_SCAN, len(qw)), replace=False)]

        h_map, k_map = wall_hist(mpts)
        h_scan, k_scan = wall_hist(qs)
        d = best_shift(h_scan, h_map)
        if d is None:
            continue
        deltas.append(d)
        kept.append((k_scan, k_map))

    a = np.array(deltas)
    print('references tested: %d' % len(a))
    print('plane-level yaw offset:  mean %+.3f   median %+.3f   |offset| mean %.3f   p90 %.3f'
          % (a.mean(), np.median(a), np.abs(a).mean(), np.percentile(np.abs(a), 90)))
    print('share within 0.5 deg: %.0f%%   within 1 deg: %.0f%%'
          % (100 * (np.abs(a) <= 0.5).mean(), 100 * (np.abs(a) <= 1.0).mean()))
    k = np.array(kept)
    print('vertical-surface points used: scan median %d   map median %d'
          % (int(np.median(k[:, 0])), int(np.median(k[:, 1]))))


if __name__ == '__main__':
    main()
