"""R4 extras: damage anatomy, error-bucket migration, along/lateral decomposition."""
import glob
import json
import sys

import numpy as np

OUT = '/data2/zhangcy/r4-full-20261019-v1'
sys.path.insert(0, '/home/zhangcy/projects/r-unified-eval-20261018-v1')
import score_unified as SU  # noqa: E402

w = SU.load_world()
keys = w['hard_keys']
gt = w['hard_gt']
base = w['hard_base']
keep = ~w['hard_replaced']
z = np.load(OUT + '/poses_main.npz', allow_pickle=True)
P = np.asarray(z['poses'], float)
diag = {}
for line in open(OUT + '/diagnostics_main.jsonl'):
    if line.strip():
        r = json.loads(line)
        diag[r['key']] = r
tb = SU.pos_err(base, gt)
tm = SU.pos_err(P, gt)
rb = SU.rot_deg(base[:, :3, :3], gt[:, :3, :3])
rm = SU.rot_deg(P[:, :3, :3], gt[:, :3, :3])
s = np.array([diag[k]['s'] for k in keys])
d = np.array([diag[k]['d'] for k in keys])
mp = np.array([diag[k]['map_points'] for k in keys])
src = np.array([diag[k]['u_source'] for k in keys])

print('=== A. error bucket migration (position error, kept frames) ===')
edges = [0, 0.5, 1, 3, 10, 1e9]
names = ['<=0.5', '0.5-1', '1-3', '3-10', '>10']
for a in range(len(names)):
    line = []
    for b in range(len(names)):
        m = keep & (tb >= edges[a]) & (tb < edges[a + 1]) & (tm >= edges[b]) & (tm < edges[b + 1])
        line.append(int(m.sum()))
    print('  base %-7s -> %s' % (names[a], ' '.join('%6d' % x for x in line)))

print('\n=== B. what the lateral search alone buys (s == 0 frames) ===')
for name, sel in (('s == 0 (n=%d)' % int((s == 0).sum()), (s == 0) & keep),
                  ('s != 0 (n=%d)' % int((s != 0).sum()), (s != 0) & keep),
                  ('|d| >= 0.5', (np.abs(d) >= 0.5) & keep),
                  ('|d| < 0.05', (np.abs(d) < 0.05) & keep)):
    print('  %-18s MPE %7.4f -> %7.4f   med %6.3f -> %6.3f   pos<=1m %4d -> %4d'
          % (name, tb[sel].mean(), tm[sel].mean(), np.median(tb[sel]), np.median(tm[sel]),
             int((tb[sel] <= 1).sum()), int((tm[sel] <= 1).sum())))

print('\n=== C. anatomy of the 18 damaged frames (>0.5 m worse, kept) ===')
dm = keep & (tm > tb + 0.5)
print('  map points of damaged: %s' % np.sort(mp[dm]))
print('  map points of all    : med=%d p10=%d' % (np.median(mp), np.percentile(mp, 10)))
print('  damaged with map <5000: %d/%d   with |s|>=2: %d   with |d|>=1: %d'
      % (int((mp[dm] < 5000).sum()), int(dm.sum()), int((np.abs(s[dm]) >= 2).sum()),
         int((np.abs(d[dm]) >= 1).sum())))
print('  damaged by u source: %s' % {u: int(((src == u) & dm).sum()) for u in set(src)})
print('  |s| distribution on damaged:', np.round(np.sort(np.abs(s[dm])), 1).tolist())

print('\n=== D. the 555 would-abstain frames ===')
would = np.array([bool(diag[k]['would_abstain']) for k in keys]) & keep
print('  n=%d  MPE %.4f -> %.4f ; pos<=1m %d -> %d ; improved>0.5m %d ; worse>0.5m %d'
      % (would.sum(), tb[would].mean(), tm[would].mean(),
         int((tb[would] <= 1).sum()), int((tm[would] <= 1).sum()),
         int(((tm - tb)[would] < -0.5).sum()), int(((tm - tb)[would] > 0.5).sum())))

print('\n=== E. rotation is untouched by the sweep (sanity) ===')
print('  max |MOE(main) - MOE(base)| = %.3e deg ; rotations identical: %s'
      % (abs(rm.mean() - rb.mean()),
         bool(np.allclose(P[:, :3, :3], base[:, :3, :3], atol=1e-12))))
print('  after the refine stage: MOE %.4f (base %.4f)' % (
    SU.rot_deg(np.load(OUT + '/poses_main_refined.npz', allow_pickle=True)['poses'][:, :3, :3],
               gt[:, :3, :3]).mean(), rb.mean()))
print('\n=== F. timing ===')
sec = np.array([diag[k]['seconds'] for k in keys])
tg = np.array([diag[k]['t_grid'] for k in keys])
print('  per-frame seconds: med=%.1f mean=%.1f max=%.1f ; grid fraction %.1f%%'
      % (np.median(sec), sec.mean(), sec.max(), 100.0 * (tg / np.maximum(sec, 1e-9)).mean()))
print('  single-core total: %.1f h for 1502 frames' % (sec.sum() / 3600.0))
