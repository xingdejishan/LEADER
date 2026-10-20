"""Collect per-frame R4 JSON records into the deliverables.

    python r4_collect.py --tag main

Writes (under /data2/zhangcy/r4-full-20261019-v1/), always in the canonical
scored.npz key order:
  poses_<tag>.npz           keys + poses (N,4,4)          main output of the arm
  poses_<tag>_refined.npz   keys + poses (N,4,4)          arm + frozen refine stage
  diagnostics_<tag>.jsonl   one line per frame, everything except the raw cost grid
  grids_<tag>.npz           (N,41,81) float32 cost grids + S_GRID/D_GRID
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

OUT = Path('/data2/zhangcy/r4-full-20261019-v1')
SCORED = '/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/scored.npz'
S_GRID = np.round(np.arange(-10.0, 10.0001, 0.5), 3)
D_GRID = np.round(np.arange(-2.0, 2.0001, 0.05), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='main')
    ap.add_argument('--allow-partial', action='store_true')
    A = ap.parse_args()
    keys = [str(k) for k in np.load(SCORED, allow_pickle=True)['keys']]
    recs = {}
    for p in sorted(glob.glob(str(OUT / 'results' / A.tag / '*.json'))):
        r = json.load(open(p))
        recs[r['key']] = r
    n = len(keys)
    print('[collect] tag=%s records=%d expected=%d' % (A.tag, len(recs), n))
    miss = [k for k in keys if k not in recs]
    if miss and not A.allow_partial:
        print('[collect] MISSING %d frames, e.g. %s -- refusing (use --allow-partial)'
              % (len(miss), miss[:3]))
        return 1

    poses = np.zeros((n, 4, 4))
    ref = np.zeros((n, 4, 4))
    has_ref = np.zeros(n, bool)
    have = np.zeros(n, bool)
    grids = np.full((n, len(S_GRID), len(D_GRID)), np.nan, np.float32)
    for i, k in enumerate(keys):
        r = recs.get(k)
        if r is None:
            poses[i] = np.eye(4)
            continue
        have[i] = True
        poses[i] = np.asarray(r['pose'], float)
        if r.get('pose_refined'):
            ref[i] = np.asarray(r['pose_refined'], float)
            has_ref[i] = True
        else:
            ref[i] = poses[i]
        if 'grid' in r:
            grids[i] = np.asarray(r['grid'], np.float32)

    np.savez_compressed(OUT / ('poses_%s.npz' % A.tag),
                        keys=np.array(keys), poses=poses, have=have)
    np.savez_compressed(OUT / ('poses_%s_refined.npz' % A.tag),
                        keys=np.array(keys), poses=ref, has_refined=has_ref)
    np.savez_compressed(OUT / ('grids_%s.npz' % A.tag),
                        keys=np.array(keys), grids=grids, s_grid=S_GRID, d_grid=D_GRID)
    with open(OUT / ('diagnostics_%s.jsonl' % A.tag), 'w') as f:
        for i, k in enumerate(keys):
            r = recs.get(k)
            if r is None:
                continue
            out = {kk: vv for kk, vv in r.items() if kk != 'grid'}
            out['uses_truth_u'] = str(r.get('u_source', '')).startswith('truth')
            g = grids[i]
            if np.isfinite(g[:, 0]).any():
                a = int(np.argmin(np.abs(S_GRID - r['s'])))
                b = int(np.argmin(np.abs(D_GRID - r['d'])))
                out['s_curve'] = [round(float(x), 4) for x in g.min(1)]
                out['d_curve'] = [round(float(x), 4) for x in g[a]]
                out['d_curve_s0'] = [round(float(x), 4)
                                     for x in g[int(np.argmin(np.abs(S_GRID)))]]
                out['curve_argmin'] = [float(S_GRID[a]), float(D_GRID[b]),
                                       round(float(g[a, b]), 4)]
            f.write(json.dumps(out) + '\n')
    print('[collect] frames=%d refined=%d grids_nonempty=%d'
          % (int(have.sum()), int(has_ref.sum()), int(np.isfinite(grids[:, 0, 0]).sum())))
    print('[collect] wrote poses_%s.npz / poses_%s_refined.npz / grids_%s.npz / '
          'diagnostics_%s.jsonl' % (A.tag, A.tag, A.tag, A.tag))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
