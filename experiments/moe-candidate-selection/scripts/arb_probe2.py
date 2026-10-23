import json, os, sys
import numpy as np

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
sys.path.insert(0, '/home/zhangcy/projects/r-unified-eval-20261018-v1')
import score_unified as SU

z = np.load(os.path.join(ROOT, 'results/three_route_merged.npz'), allow_pickle=True)
rows = [json.loads(l) for l in open(os.path.join(ROOT, 'results/three_route_merged_diag.jsonl'))]
print('keys match npz:', [str(k) for k in z['keys']][:3], rows[0]['key'])
n = len(rows)
ratio = np.array([r.get('ratio', np.nan) for r in rows], float)
print('gate frames ratio<0.21:', int((ratio < 0.21).sum()), ' finite ratio:', int(np.isfinite(ratio).sum()))
print('ratio quantiles', np.nanpercentile(ratio, [1, 5, 10, 25, 50]).round(3))
c3 = np.array([r.get('c_route3', np.nan) for r in rows], float)
cv = np.array([r.get('c_visual', np.nan) for r in rows], float)
print('scored frames:', int(np.isfinite(c3).sum()), ' visual accepted:', int(sum(1 for r in rows if r['visual'])))
print('c3 mean/med', np.nanmean(c3).round(4), np.nanmedian(c3).round(4))
print('cv mean/med', np.nanmean(cv).round(4), np.nanmedian(cv).round(4))

# visual candidates
vis = np.full((n, 4, 4), np.nan)
have = 0
for i in range(n):
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
    if best is not None:
        vis[i] = best[1]
        have += 1
print('frames with a visual candidate:', have)
d = np.linalg.norm(vis[:, :3, 3] - z['poses'][:, :3, 3], axis=1)
d = d[np.isfinite(d)]
print('pose separation quantiles (m)', np.percentile(d, [0, 10, 25, 50, 75, 90, 95, 99, 100]).round(2))

w = SU.load_world()
print('world keys == npz keys:', w['hard_keys'] == [str(k) for k in z['keys']])
pred = z['poses']
t = SU.pos_err(pred, w['hard_gt']); r = SU.rot_deg(pred[:, :3, :3], w['hard_gt'][:, :3, :3])
print('route3-only: MPE %.4f MOE %.4f ok %d >5m %d >10m %d' % (
    t.mean(), r.mean(), int(((t <= 1) & (r <= 2)).sum()), int((t > 5).sum()), int((t > 10).sum())))
# reproduce stored choice
chosen = np.stack([np.asarray(vis[i] if rows[i]['visual'] else z['poses'][i]) for i in range(n)])
t2 = SU.pos_err(chosen, w['hard_gt']); r2 = SU.rot_deg(chosen[:, :3, :3], w['hard_gt'][:, :3, :3])
print('diag-choice: MPE %.4f MOE %.4f ok %d >5m %d >10m %d' % (
    t2.mean(), r2.mean(), int(((t2 <= 1) & (r2 <= 2)).sum()), int((t2 > 5).sum()), int((t2 > 10).sum())))
