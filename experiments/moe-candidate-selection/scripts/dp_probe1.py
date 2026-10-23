#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Probe 1: project layout, full-online.npz keys, stages dir, truth json shape."""
import json
import os
import sys

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
print('[tree]', P)
for root, dirs, files in os.walk(P):
    dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', 'node_modules')]
    rel = os.path.relpath(root, P)
    depth = 0 if rel == '.' else rel.count(os.sep) + 1
    if depth > 3:
        continue
    py = sorted([f for f in files if f.endswith('.py')])
    other = sorted([f for f in files if not f.endswith('.py') and not f.endswith('.pyc')])
    print('  ' * depth + rel + '/')
    for f in py:
        print('  ' * (depth + 1) + 'py  ' + f + '  %d' % (os.path.getsize(os.path.join(root, f))))
    for f in other[:20]:
        print('  ' * (depth + 1) + '    ' + f + '  %d' % (os.path.getsize(os.path.join(root, f))))

fo = P + '/results/full-online.npz'
print('\n[full-online]', os.path.exists(fo))
if os.path.exists(fo):
    with np.load(fo, allow_pickle=True) as z:
        for k in z.keys():
            a = np.asarray(z[k])
            print('   %-20s %-16s %s' % (k, str(a.shape), a.dtype))
        keys = [str(k) for k in z['keys']]
        print('   n_keys', len(keys), keys[0], keys[-1])
        from collections import Counter
        c = Counter(k.split('/')[0] for k in keys)
        print('   dates', len(c))
        for d, n in sorted(c.items()):
            print('     ', d, n)

print('\n[results dir]')
for f in sorted(os.listdir(P + '/results')):
    print('   ', f, os.path.getsize(os.path.join(P, 'results', f)))
