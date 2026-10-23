import json
import os

import numpy as np

RES = '/home/zhangcy/projects/e2e-moe-20261020-v1/results'


def load(prefix, shard=0):
    path = os.path.join(RES, '%s-shard%d.jsonl' % (prefix, shard))
    return [json.loads(line) for line in open(path)]


def poses(rows):
    return {row['key']: (np.asarray(row['t0'], float).reshape(4, 4),
                         np.asarray(row['pose'], float).reshape(4, 4)) for row in rows}


base = poses(load('bench_p1'))
print('p1 rows', len(base))
for name in ('bench_p1r4', 'bench_p2', 'bench_p2r4', 'bench_p4'):
    other = poses(load(name))
    shared = [k for k in base if k in other]
    dt0 = [float(np.linalg.norm(base[k][0][:3, 3] - other[k][0][:3, 3])) for k in shared]
    dfin = [float(np.linalg.norm(base[k][1][:3, 3] - other[k][1][:3, 3])) for k in shared]
    dmat = max(float(np.abs(base[k][1] - other[k][1]).max()) for k in shared)
    print('%-12s shared=%3d  dT0 mean %.4f max %.4f m | dfinal mean %.4f max %.4f m | '
          'max|dT0mat| %.2e  max|dfinal_mat| %.2e'
          % (name, len(shared), sum(dt0) / len(dt0), max(dt0),
             sum(dfin) / len(dfin), max(dfin), dmat,
             max(float(np.abs(base[k][0] - other[k][0]).max()) for k in shared)))

print()
rows = load('stab200')
print('stab200 rows', len(rows))
rss = [row['rss_MiB'] for row in rows]
tot = [row['total'] for row in rows]
ends = [row['t_end'] for row in rows]
steady = rows[1:]
span = steady[-1]['t_end'] - steady[0]['t_end']
print('rss first %.0f  @50 %.0f  @100 %.0f  @150 %.0f  last %.0f  max %.0f MiB'
      % (rss[0], rss[49], rss[99], rss[149], rss[-1], max(rss)))
print('rss growth frame1->frame200: %+.1f MiB (%.2f%%); last 100 frames: %+.1f MiB'
      % (rss[-1] - rss[1], 100.0 * (rss[-1] - rss[1]) / rss[1], rss[-1] - rss[99]))
print('steady ms/frame %.1f  fps %.3f  span %.1f s' %
      (1000.0 * span / (len(steady) - 1), (len(steady) - 1) / span, span))
print('serial work mean %.3f s  -> overlap factor %.2f'
      % (sum(tot) / len(tot), (sum(tot) / len(tot)) / (span / (len(steady) - 1))))
for stage in ('decode', 'leader', 'refine', 'geometric_rot'):
    m = sum(row['seconds'][stage] for row in steady) / len(steady)
    print('  %-14s %.3f s/frame  util %.2f' % (stage, m, m / (span / (len(steady) - 1))))
# npz sanity
z = np.load(os.path.join(RES, 'stab200-shard0.npz'), allow_pickle=True)
print('npz keys', z['keys'].shape, 'poses', z['poses'].shape, 'finite',
      bool(np.isfinite(z['poses']).all()))
print('first/last key', z['keys'][0], z['keys'][-1])
