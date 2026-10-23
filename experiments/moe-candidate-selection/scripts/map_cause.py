"""Where does the map's 0.7 deg inconsistency come from?

For each sampled reference: register it against a map of its neighbours, and record
the yaw offset together with covariates that separate the candidate causes.

  speed             metres/second during the sweep  -> motion distortion (undistortion fixes it)
  per-date constant -> trajectory/frame convention   -> per-date calibration fixes it
  uncorrelated      -> reference pose noise          -> joint registration fixes it
"""
import collections
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

GRID = np.arange(-6.0, 6.001, 0.5)
RADIUS = 30.0
TRUNC = 1.0
N_REF = 40
N_NEIGHBOUR = 8


def scan_file(entry):
    path = entry['scan_path']
    if path.endswith('/'):
        return path + str(entry['key']).split('/')[1] + '.bin'
    return path


def main():
    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    cfg = P.Config()
    poses = [np.asarray(r['world_from_body'], float) for r in refs]
    pos = np.asarray([T[:3, 3] for T in poses], float)
    keys = [str(r['key']) for r in refs]
    stamps = np.array([int(k.split('/')[1]) for k in keys], dtype=np.int64)
    dates = np.array([k.split('/')[0] for k in keys])
    tree = cKDTree(pos)

    # speed from the temporally preceding reference of the same date
    order = np.lexsort((stamps, dates))
    speed = np.full(len(refs), np.nan)
    dts = []
    for a, b in zip(order[:-1], order[1:]):
        if dates[a] != dates[b]:
            continue
        dt = (stamps[b] - stamps[a]) / 1e6
        dts.append(dt)
        if 0 < dt <= 6.0:
            speed[b] = float(np.linalg.norm(pos[b] - pos[a]) / dt)
    if dts:
        print('reference time spacing: median %.3f s  p10 %.3f  p90 %.3f'
              % (np.median(dts), np.percentile(dts, 10), np.percentile(dts, 90)))
    prev = np.full(len(refs), np.nan)
    prev[order[1:]] = order[:-1]

    rng = np.random.default_rng(23)
    sample = rng.choice(len(refs), N_REF * 4, replace=False)
    rows = []
    for i in sample:
        if len(rows) >= N_REF:
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
        if len(cloud) < 2000:
            continue
        mt = cKDTree(cloud)
        q = F.decode(scan_file(refs[i]))
        if len(q) > 4000:
            q = q[::max(1, len(q) // 4000)][:4000]
        costs = []
        for d in GRID:
            Rd = R @ Rotation.from_euler('z', d, degrees=True).as_matrix()
            costs.append(float(np.minimum(mt.query(q @ Rd.T + t, k=1, workers=1)[0], TRUNC).mean()))
        rows.append(dict(i=int(i), date=str(dates[i]), off=float(GRID[int(np.argmin(costs))]),
                         speed=float(speed[i]) if np.isfinite(speed[i]) else np.nan,
                         npts=int(len(cloud))))

    off = np.array([r['off'] for r in rows])
    sp = np.array([r['speed'] for r in rows])
    print('references %d   |offset| mean %.3f deg   p90 %.3f' % (len(rows), np.abs(off).mean(),
                                                                np.percentile(np.abs(off), 90)))
    ok = np.isfinite(sp)
    if ok.sum() > 5:
        print('correlation of offset with speed: r = %+.3f  (n=%d, speed %.1f-%.1f m/s)'
              % (np.corrcoef(off[ok], sp[ok])[0, 1], int(ok.sum()), np.nanmin(sp), np.nanmax(sp)))
    print()
    print('%-12s %5s %9s %9s %9s' % ('date', 'n', 'mean off', '|off| mean', 'mean speed'))
    for d in sorted(set(r['date'] for r in rows)):
        sub = [r for r in rows if r['date'] == d]
        o = np.array([r['off'] for r in sub])
        s = np.array([r['speed'] for r in sub])
        print('%-12s %5d %9.3f %9.3f %9.1f' % (d, len(sub), o.mean(), np.abs(o).mean(), np.nanmean(s)))


if __name__ == '__main__':
    main()
