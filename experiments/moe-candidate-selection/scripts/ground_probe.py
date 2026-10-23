import json, sys, os
import numpy as np
sys.path.insert(0, '/home/zhangcy/projects/e2e-moe-20261020-v1/stages/fast_search')
sys.path.insert(0, '/home/zhangcy/projects/e2e-moe-20261020-v1')
import pipeline as P

frames = json.load(open('/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/frames.json'))
refs = json.load(open('/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/refs.json'))
cfg = P.Config()
z = np.load('/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/initial.npz', allow_pickle=True)
poses = np.asarray(z['poses'], float)

for i in (0, 500, 1000):
    scan = P.prepare_scan(frames[i], cfg)
    zz = scan[:, 2]
    print('frame %4d  points %5d   z percentiles 1/5/50/95/99: %.2f %.2f %.2f %.2f %.2f'
          % (i, len(scan), *np.percentile(zz, [1, 5, 50, 95, 99])))
    hist, edges = np.histogram(zz, bins=20, range=(-3, 3))
    top = np.argsort(hist)[-3:][::-1]
    print('     densest z bands:', [round(float((edges[k] + edges[k+1]) / 2), 2) for k in top],
          'counts', [int(hist[k]) for k in top])

# how many points fall into a ground band, and how planar is it
print()
for i in (0, 500, 1000):
    scan = P.prepare_scan(frames[i], cfg)
    lo, hi = np.percentile(scan[:, 2], [3, 20])
    m = (scan[:, 2] > lo) & (scan[:, 2] < hi)
    g = scan[m]
    if len(g) < 100:
        print(i, 'too few ground points', len(g)); continue
    c = g - g.mean(0)
    w, v = np.linalg.eigh(c.T @ c / max(1, len(g) - 1))
    n = v[:, 0]
    print('frame %4d ground band [%.2f,%.2f]  n=%5d  plane normal(bod) %s  thickness %.3f'
          % (i, lo, hi, len(g), np.round(n, 3), float(np.sqrt(w[0]))))
