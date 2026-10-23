#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import statistics as st

p = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/online-shard0.jsonl'
rows = [json.loads(l) for l in open(p)]
print('rows', len(rows))
for name in ('decode_leader', 'refine', 'geometric_rot', 'arb', 'total'):
    if name == 'total':
        v = [r['total'] for r in rows]
    else:
        v = [r['seconds'][name] for r in rows]
    v = sorted(v)
    print('%-14s min %.3f  med %.3f  mean %.3f  p90 %.3f  max %.3f'
          % (name, v[0], st.median(v), sum(v) / len(v), v[int(0.9 * (len(v) - 1))], v[-1]))
print()
print('slowest 5 frames:')
for r in sorted(rows, key=lambda r: -r['total'])[:5]:
    print('  %.3f  %s  %s' % (r['total'], r['key'], {k: round(v, 3) for k, v in r['seconds'].items()}))
print()
print('sum of medians', round(sum(st.median([r['seconds'][k] for r in rows])
                                   for k in ('decode_leader', 'refine', 'geometric_rot', 'arb')), 4))
print('trimmed mean total (drop slowest 3)', round(sum(sorted(r['total'] for r in rows)[:-3]) / (len(rows) - 3), 4))
print('ratio sample', [round(r['ratio'], 3) for r in rows[:6]])
print('gain sample', [round(r['gain'], 4) for r in rows[:6]])
print('expanded set', sorted({r['expanded'] for r in rows}))
print('rotated', sum(1 for r in rows if r['rotated']), '/', len(rows))
print('solver', {r['solver'] for r in rows})
