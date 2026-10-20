"""Probe 5: diagnose the remaining ~90-degree u disagreements (window members and paths)."""
import sys

import numpy as np

sys.path.insert(0, '/home/zhangcy/projects/r4-full-20261019-v1')
import r4_lib as L

w = L.World()
for i in (914, 1130, 1038, 880):
    d = w.dates[i]
    order = w._order[d]
    j = w._pos_in_date[d][i]
    print('=== idx %d %s j=%d/%d' % (i, w.keys[i], j, len(order)))
    lo, hi = max(0, j - 5), min(len(order), j + 6)
    for k in range(lo, hi):
        gi = order[k]
        g = w.gt[gi, :3, 3]
        p = w.deploy[gi, :3, 3]
        print('  k=%4d gt=[%9.2f %9.2f %6.2f] dep=[%9.2f %9.2f %6.2f] d=%.2f%s'
              % (k, g[0], g[1], g[2], p[0], p[1], p[2], np.linalg.norm(p - g),
                 '  <<<' if gi == i else ''))
    gd = w._quad_direction(w.gt, i)
    dd = w._quad_direction(w.deploy, i)
    print('  true-quad', None if gd is None else (np.round(gd[0], 3), gd[1]))
    print('  depl-quad', None if dd is None else (np.round(dd[0], 3), dd[1]))
