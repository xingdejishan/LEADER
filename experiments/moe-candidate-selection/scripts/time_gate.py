import json, sys, time
sys.path.insert(0, '/home/zhangcy/projects/e2e-three-route-gate-20261020-v1')
import numpy as np
import gate as G1

PR = '/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/'
frames = json.load(open(PR + 'frames.json'))
z = np.load(PR + 'initial.npz', allow_pickle=True)
poses = np.asarray(z['poses'], float)
lib = G1.ReferenceLibrary()
idx = [0, 120, 260, 400, 640, 880, 1010, 1180, 1330, 1470]
ts = []
ratios = []
for i in idx:
    t0 = time.perf_counter()
    r = G1.ambiguity_ratio(lib, frames[i]['scan_path'], poses[i])
    ts.append(time.perf_counter() - t0)
    ratios.append(round(float(r['ratio']), 3))
ts = np.array(ts) * 1000.0
print('frames %d  per-frame ms: median %.0f  min %.0f  max %.0f' % (len(idx), np.median(ts), ts.min(), ts.max()))
print('all calls (incl. warmup):', [round(float(x)) for x in ts])
print('ratios:', ratios)
