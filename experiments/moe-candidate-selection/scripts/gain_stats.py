"""Gain distribution and candidate thresholds after the sampling change."""
import json
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else '/home/zhangcy/projects/e2e-moe-20261020-v1/results/_g0_diag.jsonl'
d = [json.loads(l) for l in open(path)]
g = np.array([x['gain'] for x in d])
print('frames=%d  median=%.5f  p75=%.5f  p90=%.5f  frac>0=%.1f%%'
      % (len(g), np.median(g), np.percentile(g, 75), np.percentile(g, 90),
         100 * (g > 0).mean()))
for t in (0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01):
    print('  thr=%.4f -> pass %4d (%.1f%%)' % (t, int((g > t).sum()), 100 * (g > t).mean()))
fine = [x['ran_fine'] for x in d]
print('ran_fine in this run:', sum(fine))
