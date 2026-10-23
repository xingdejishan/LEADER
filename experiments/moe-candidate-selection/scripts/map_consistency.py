"""How self-consistent is the reference map?

Take reference scans, register each one against a map built from the OTHER references
near it, and compare the fitted yaw with the yaw it is recorded at.  That residual is
the map's own registration error: a lower bound on what any single-frame estimator can
achieve against it, and a candidate explanation for the shared floor.
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

GRID = np.arange(-10.0, 10.001, 1.0)
RADIUS = 30.0
TRUNC = 1.0
N_REF = 30
N_NEIGHBOUR = 8


def scan_file(entry):
    path = entry['scan_path']
    if path.endswith('/'):
        stamp = str(entry['key']).split('/')[1]
        return path + stamp + '.bin'
    return path


def main():
    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    cfg = P.Config()
    pos = np.asarray([np.asarray(r['world_from_body'], float)[:3, 3] for r in refs], float)
    tree = cKDTree(pos)

    rng = np.random.default_rng(17)
    sample = rng.choice(len(refs), N_REF * 3, replace=False)
    used, deltas, n_nb = 0, [], []
    for i in sample:
        if used >= N_REF:
            break
        T = np.asarray(refs[i]['world_from_body'], float)
        t, R = T[:3, 3], T[:3, :3]
        _, nb = tree.query(t, k=N_NEIGHBOUR + 1)
        nb = [int(j) for j in np.atleast_1d(nb) if int(j) != int(i)]
        if not nb:
            continue
        cloud = []
        for j in nb:
            Tj = np.asarray(refs[j]['world_from_body'], float)
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
        deltas.append(float(GRID[int(np.argmin(costs))]))
        n_nb.append(len(cloud))
        used += 1

    d = np.array(deltas)
    print('references tested: %d   neighbours each: median %d points'
          % (len(d), int(np.median(n_nb))))
    print('fitted-vs-recorded yaw offset:  mean %+.3f deg   median %+.3f   |offset| mean %.3f   p90 |offset| %.3f'
          % (d.mean(), np.median(d), np.abs(d).mean(), np.percentile(np.abs(d), 90)))
    print('share with |offset| <= 1 deg: %.0f%%' % (100 * (np.abs(d) <= 1).mean()))


if __name__ == '__main__':
    main()
