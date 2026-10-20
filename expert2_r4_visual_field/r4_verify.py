"""R4 self-check: re-derive the cost of the selected candidate with a plain loop and
confirm the recorded grid, map size and selection are consistent.

    python r4_verify.py --tag smoke [--n 5]
"""
import argparse
import glob
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, '/home/zhangcy/projects/r4-full-20261019-v1')
import r4_lib as L  # noqa: E402

OUT = '/data2/zhangcy/r4-full-20261019-v1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='smoke')
    ap.add_argument('--n', type=int, default=5)
    A = ap.parse_args()
    w = L.World()
    files = sorted(glob.glob(OUT + '/results/%s/*.json' % A.tag))
    picked = files[::max(1, len(files) // A.n)][:A.n]
    print('checking %d frames of tag=%s' % (len(picked), A.tag))
    worst = 0.0
    for p in picked:
        r = json.load(open(p))
        i = w.keys.index(r['key'])
        q = w.query(i)
        mapp, ids, per = w.local_map(i)
        assert len(mapp) == r['map_points'], (len(mapp), r['map_points'])
        assert list(ids) == r['ref_ids']
        assert len(q) == r['query_points']
        tree = cKDTree(mapp)
        grid = np.asarray(r['grid'], float)
        a = int(np.argmin(np.abs(L.S_GRID - r['s'])))
        b = int(np.argmin(np.abs(L.D_GRID - r['d'])))
        assert abs(grid[a, b] - r['cost_selected']) < 1e-3
        # independent recomputation of three candidates, plain per-candidate loop
        u = np.asarray(r['u'], float)
        e = np.asarray(r['lateral'], float)
        T0 = w.deploy[i]
        for (aa, bb) in ((a, b), (int(np.argmin(np.abs(L.S_GRID))), int(np.argmin(np.abs(L.D_GRID)))),
                         (0, 0)):
            T = T0.copy()
            T[:3, 3] = T0[:3, 3] + L.S_GRID[aa] * u + L.D_GRID[bb] * e
            world = q @ T[:3, :3].T + T[:3, 3]
            dist = tree.query(world, k=1)[0]
            c = float(np.minimum(dist, L.TRUNC).mean())
            worst = max(worst, abs(c - grid[aa, bb]))
            print('  %-28s s=%+5.1f d=%+5.2f  grid=%.6f direct=%.6f  |delta|=%.2e'
                  % (r['key'], L.S_GRID[aa], L.D_GRID[bb], grid[aa, bb], c,
                     abs(c - grid[aa, bb])))
        # full grid argmin must be the recorded choice
        ai, bi = np.unravel_index(int(np.argmin(grid)), grid.shape)
        assert (ai, bi) == (a, b), ((ai, bi), (a, b))
    print("max |grid - direct| = %.2e  (grid is stored rounded to 4 dp, so the tolerance is 5e-5)" % worst)
    print("PASS" if worst < 5.0e-5 else "FAIL")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
