import sys, time, json
sys.path.insert(0, '/home/zhangcy/projects/e2e-moe-20261020-v1')
sys.path.insert(0, '/home/zhangcy/projects/e2e-three-route-gate-20261020-v1')
import numpy as np
import run_pipeline as RP
import gate as G1

RP._OPTS.update(gate2=0.01, rot=True)
t0 = time.perf_counter()
RP.init()
print('init %.1f s' % (time.perf_counter() - t0))

lib = G1.ReferenceLibrary()
idx = [0, 120, 260, 400, 640, 880]
rows = []
for i in idx:
    G1.ambiguity_ratio(lib, RP._G['frames'][i]['scan_path'], RP._G['poses'][i])  # warm
    t0 = time.perf_counter()
    G1.ambiguity_ratio(lib, RP._G['frames'][i]['scan_path'], RP._G['poses'][i])
    tg = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    RP.one(i)
    tr = (time.perf_counter() - t0) * 1000
    rows.append((tg, tr))
    print('frame %4d   gate %5.0f ms   route3 %5.0f ms' % (i, tg, tr))

a = np.array(rows)
print('median: gate %.0f ms   route3 %.0f ms   gate/route3 %.2f'
      % (np.median(a[:, 0]), np.median(a[:, 1]), np.median(a[:, 0]) / np.median(a[:, 1])))
