import json, os, sys, time
import numpy as np
from scipy.spatial import cKDTree

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
sys.path.insert(0, os.path.join(ROOT, 'stages/fast_search'))
import fast_lib as F
import pipeline as P

for f in ('three_route_unified.py', 'results/three_route_merged.npz',
          'results/three_route_merged_diag.jsonl', 'stages/fast_search/fast_lib.py'):
    p = os.path.join(ROOT, f)
    print('%-42s %s' % (f, time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p)))))

frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
cfg = P.Config()
z = np.load(os.path.join(ROOT, 'results/three_route_merged.npz'), allow_pickle=True)
T3 = np.asarray(z['poses'], float)
diag = [json.loads(l) for l in open(os.path.join(ROOT, 'results/three_route_merged_diag.jsonl'))]
ratio = np.array([d['ratio'] for d in diag], float)
c3_ref = np.array([d.get('c_route3', np.nan) for d in diag], float)
cv_ref = np.array([d.get('c_visual', np.nan) for d in diag], float)
acc_ref = np.array([bool(d.get('visual')) for d in diag])

vis = [None] * len(frames)
for i in range(len(frames)):
    rec = json.load(open(VERIFY % i))['poses']
    best = None
    for name, row in rec.items():
        if not name.startswith('candidate_'):
            continue
        sc = (row.get('fair') or {}).get('geometry_score')
        if sc is None:
            continue
        if best is None or sc > best[0]:
            best = (float(sc), np.asarray(row['pose'], float))
    vis[i] = None if best is None else best[1]

maps = F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                   radius=cfg.crop_radius, max_refs=800, max_maps=4096)


def cost(tree, pts, trunc=1.0):
    d = tree.query(pts, k=1, workers=1, distance_upper_bound=trunc)[0]
    return float(np.where(np.isfinite(d), d, trunc).mean())


def score(pose, q):
    centre = pose[:3, 3]
    mp = maps.local(centre, maps.nearby(centre)[:cfg.reference_count])
    return cost(cKDTree(mp), q @ pose[:3, :3].T + centre)


gated = [i for i in range(len(frames)) if np.isfinite(ratio[i]) and ratio[i] < 0.21 and vis[i] is not None]
print('gated', len(gated), 'accepted in diag', int(acc_ref.sum()))

modes = ('prefix', 'spread', 'full')
diff = {m: np.zeros(len(gated)) for m in modes}
diffv = {m: np.zeros(len(gated)) for m in modes}
for k, i in enumerate(gated):
    if not len(maps.nearby(T3[i, :3, 3])):
        continue
    scan = P.prepare_scan(frames[i], cfg)
    qs = {'prefix': scan[:min(4000, len(scan))],
          'spread': F.spread_subset(scan, 4000),
          'full': scan}
    for m in modes:
        diff[m][k] = score(T3[i], qs[m]) - c3_ref[i]
        diffv[m][k] = score(vis[i], qs[m]) - cv_ref[i]

for m in modes:
    print('%-7s c3: max|d|=%.3e  n>1e-9=%d |  cv: max|d|=%.3e n>1e-9=%d'
          % (m, np.abs(diff[m]).max(), int((np.abs(diff[m]) > 1e-9).sum()),
             np.abs(diffv[m]).max(), int((np.abs(diffv[m]) > 1e-9).sum())))

print('\naccepted-in-diag frames, translation separation / rotation difference:')
for i in np.where(acc_ref)[0]:
    if vis[i] is None:
        print(i, 'no vis'); continue
    d = np.linalg.norm(vis[i][:3, 3] - T3[i, :3, 3])
    R = vis[i][:3, :3].T @ T3[i, :3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print('%5d sep=%9.4f rot=%7.4f deg  c3=%.4f cv=%.4f d=%.4f' % (i, d, ang, c3_ref[i], cv_ref[i], cv_ref[i] - c3_ref[i]))
