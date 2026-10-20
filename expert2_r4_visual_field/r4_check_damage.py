"""Check the two claims in REPORT section 6 about the damaged frames."""
import json
import sys

import numpy as np

OUT = '/data2/zhangcy/r4-full-20261019-v1'
sys.path.insert(0, '/home/zhangcy/projects/r-unified-eval-20261018-v1')
import score_unified as SU  # noqa: E402

w = SU.load_world()
keys = w['hard_keys']
keep = ~w['hard_replaced']
gt = w['hard_gt']
P = np.load(OUT + '/poses_main.npz', allow_pickle=True)['poses']
tb = SU.pos_err(w['hard_base'], gt)
tm = SU.pos_err(P, gt)
dm = keep & (tm > tb + 0.5)
print('damaged n=%d' % dm.sum())
print('base error of damaged frames:', np.round(tb[dm], 2).tolist())
for thr in (0.5, 1.0):
    print('  damaged with base error <= %.1f m: %d' % (thr, int((tb[dm] <= thr).sum())))
print('damaged with main error > 3 m: %d' % int((tm[dm] > 3).sum()))
print('worst 3 keys:', [keys[i] for i in np.flatnonzero(dm)[np.argsort(-(tm - tb)[dm])[:3]]])
