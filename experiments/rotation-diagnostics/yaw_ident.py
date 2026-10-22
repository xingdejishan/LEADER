"""Does the map contain the yaw?  Sweep body yaw at the TRUE translation.

For each yaw-tail frame (|yaw error| > 8 deg) and a control set, park the pose at
the ground-truth translation and sweep body yaw over +-15 deg on the reference map.
Readings:

  identifiable  the map prefers the true yaw and the dip is sharp
  flat          the cost barely changes with yaw -> no rotational evidence here
  inverted      the map prefers a yaw far from the truth -> the evidence points wrong

This answers "is the yaw information present", NOT "can the pipeline reach it".
For the second question use yaw_reach.py, which sweeps at the pipeline's own
estimated translation.  The distinction matters: on this data the first says 15/16
and the second says 5/16.
"""
import importlib.util
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import config

sys.path.insert(0, str(config.MODEL / 'stages/fast_search'))
sys.path.insert(0, str(config.MODEL))

import pipeline as P          # noqa: E402
import fast_lib as F          # noqa: E402

GRID = np.arange(-15.0, 15.001, 1.0)
SHARP_DEG = 5.0
SHARP_MIN = 0.01


def main():
    spec = importlib.util.spec_from_file_location('score_unified', config.SCORER)
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    base = np.asarray(w['hard_base'], float)

    Rd = np.swapaxes(base[:, :3, :3], -1, -2) @ gt[:, :3, :3]
    rv = np.degrees(Rotation.from_matrix(Rd).as_rotvec())
    yaw_err = np.abs(rv[:, 2])
    full = np.linalg.norm(rv, axis=1)

    frames = json.load(open(config.FRAMES))
    refs = json.load(open(config.REFS))
    cfg = P.Config()
    maps = F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                       radius=cfg.crop_radius)

    tail = list(np.flatnonzero(yaw_err > 8))
    rng = np.random.default_rng(7)
    ctrl = list(rng.choice(np.flatnonzero(full < 0.7), min(15, int((full < 0.7).sum())),
                           replace=False))

    def sweep(i):
        t = gt[i][:3, 3]
        mp = maps.local(t, maps.nearby(t)[:cfg.reference_count])
        tree = cKDTree(mp)
        scan = P.prepare_scan(frames[i], cfg)
        q = scan if len(scan) <= 4000 else scan[::max(1, len(scan) // 4000)][:4000]
        costs = []
        for d in GRID:
            R = gt[i][:3, :3] @ Rotation.from_euler('z', d, degrees=True).as_matrix()
            costs.append(float(np.minimum(tree.query(q @ R.T + t, k=1, workers=1)[0], 1.0).mean()))
        costs = np.array(costs)
        at0 = float(costs[GRID == 0][0])
        sharp = float(np.mean([costs[GRID == s][0] for s in (-SHARP_DEG, SHARP_DEG)]) - at0)
        best = float(GRID[int(np.argmin(costs))])
        if abs(best) <= 2 and sharp > SHARP_MIN:
            reading = 'identifiable'
        elif sharp <= SHARP_MIN:
            reading = 'flat'
        else:
            reading = 'inverted'
        return best, at0, float(costs.min()), sharp, reading

    print('%-6s %8s %8s %9s %9s %9s  %s' %
          ('frame', 'argmin', 'c(truth)', 'c(best)', 'sharp5', 'yrelerr', 'reading'))
    tally = {}
    for tag, group in (('TAIL', tail), ('CTRL', ctrl)):
        print('--- %s ---' % tag)
        for i in group:
            best, at0, cmin, sharp, reading = sweep(i)
            print('%-6d %8.1f %9.4f %9.4f %9.4f %9.1f  %s'
                  % (i, best, at0, cmin, sharp, full[i], reading))
            tally.setdefault(tag, []).append(reading)
    print()
    for tag, readings in tally.items():
        print('%-5s n=%d  %s' % (tag, len(readings), {r: readings.count(r) for r in set(readings)}))


if __name__ == '__main__':
    main()
