"""Final tally of the full-set online run (read-only, then writes one merged npz)."""
import collections
import json
import os

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
TRUTH = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
FILES = ['full-shard0.jsonl', 'fullb-shard1.jsonl', 'fullb-shard3.jsonl']
TOTAL = 104732

rows = {}
dups = 0
for name in FILES:
    try:
        with open(P + name) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                k = r.get('key')
                if k in rows and rows[k].get('seconds') and not r.get('seconds'):
                    continue
                if k in rows:
                    dups += 1
                rows[k] = r
    except IOError as exc:
        print('missing ledger', name, exc)

print('unique frames: %d / %d (%.2f%%)   duplicate rows seen: %d'
      % (len(rows), TOTAL, 100.0 * len(rows) / TOTAL, dups))
skipped = [r for r in rows.values() if r.get('skipped') or not r.get('seconds')]
print('skipped: %d  %s' % (len(skipped), dict(collections.Counter(r.get('skipped', '?') for r in skipped))))
ok = [r for r in rows.values() if r not in skipped]
print('produced poses: %d   missing keys: %d' % (len(ok), TOTAL - len(rows)))

truth = {}
for f in os.listdir(TRUTH):
    date = f.replace('.json', '')
    for ts, v in json.load(open(os.path.join(TRUTH, f))).items():
        truth[(date, str(ts))] = np.asarray(v['T_WB'], np.float64)


def rot_deg(a, b):
    r = np.swapaxes(a, -1, -2) @ b
    c = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


dates, e0, e1, r0, r1, keys, poses, t0s = [], [], [], [], [], [], [], []
for r in ok:
    key = r['key']
    d, ts = key.split('/')
    g = truth.get((d, ts))
    if g is None:
        continue
    A = np.asarray(r['t0'], np.float64).reshape(4, 4)
    B = np.asarray(r['pose'], np.float64).reshape(4, 4)
    dates.append(d)
    e0.append(np.linalg.norm(A[:3, 3] - g[:3, 3]))
    e1.append(np.linalg.norm(B[:3, 3] - g[:3, 3]))
    r0.append(rot_deg(A[:3, :3], g[:3, :3]))
    r1.append(rot_deg(B[:3, :3], g[:3, :3]))
    keys.append(key)
    poses.append(B)
    t0s.append(A)

dates = np.array(dates); e0 = np.array(e0); e1 = np.array(e1)
r0 = np.array(r0); r1 = np.array(r1)
print('\nframes with truth: %d' % len(e0))


def rep(tag, e, r):
    print('%-16s MPE=%.4f  MOE=%.4f  median=%.4f  ok(1m,2deg)=%.1f%%  >1m=%d  >5m=%d  >10m=%d'
          % (tag, e.mean(), r.mean(), np.median(e),
             100.0 * ((e <= 1) & (r <= 2)).mean(),
             int((e > 1).sum()), int((e > 5).sum()), int((e > 10).sum())))


rep('front end (V7)', e0, r0)
rep('MoE final', e1, r1)
for d in sorted(set(dates)):
    m = dates == d
    rep('  ' + d, e1[m], r1[m])

d = e1 - e0
print('\nchange vs front end: helped %d (%.1f%%), damaged %d (%.1f%%), flat %d'
      % (int((d < -0.1).sum()), 100.0 * (d < -0.1).mean(),
         int((d > 0.1).sum()), 100.0 * (d > 0.1).mean(), int((np.abs(d) <= 0.1).sum())))

np.savez(P + 'full-online.npz', keys=np.array(keys), poses=np.stack(poses),
         t0=np.stack(t0s))
print('\nwrote %sfull-online.npz  (%d poses)' % (P, len(keys)))
