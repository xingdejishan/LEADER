import json, os, sys, traceback
import numpy as np
sys.path.insert(0, '/dev/shm')
import fs_compute as FS

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
FS.init()
i = 0
m = json.load(open(FS.MATCH % i))
v = json.load(open(FS.VERIFY % i))['poses']
ps = [np.asarray(h['pose'], float) for k, h in enumerate(m.get('hypotheses') or [])
      if v.get('candidate_%02d' % k) is not None]
v7 = np.asarray(np.load(os.path.join(MOE, 'inputs/initial.npz'), allow_pickle=True)['poses'], float)
try:
    out = FS.one((i, ps + [v7[i]]))
    print('ok', out[0], np.round(out[1], 3))
except Exception:
    traceback.print_exc()
