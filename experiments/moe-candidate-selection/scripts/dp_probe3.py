#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Probe 3: sweep_full.npz contents, cpu count, disk, truth files, score baselines."""
import json
import os
import shutil
import subprocess

import numpy as np

S = '/home/zhangcy/.dsh_scratch/'
P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
for name in ('sweep_full.npz', 'sweep-summary.npz', 'sweep_rot.npz'):
    for cand in (S + name, P + name):
        if os.path.exists(cand):
            print('==', cand, os.path.getsize(cand))
            z = np.load(cand, allow_pickle=True)
            for k in z.keys():
                a = np.asarray(z[k])
                print('   %-14s %-18s %s' % (k, str(a.shape), a.dtype))
            if 'gain' in z.keys():
                g = np.asarray(z['gain'], float)
                print('   gain: n=%d nan=%d >0.01=%d >0=%d median=%.4f'
                      % (len(g), int(np.isnan(g).sum()), int((g > 0.01).sum()),
                         int((g > 0).sum()), float(np.nanmedian(g))))
            break
    else:
        print('== MISSING', name)

print('\n[cpu] nproc', os.cpu_count())
print(subprocess.run(['nproc'], capture_output=True, text=True).stdout.strip())
print('\n[disk]')
print(subprocess.run(['df', '-h', '/home/zhangcy', '/data2'], capture_output=True, text=True).stdout)
print('\n[mem]')
print(subprocess.run(['free', '-g'], capture_output=True, text=True).stdout)

T = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
print('[truth]', T)
for f in sorted(os.listdir(T)):
    p = os.path.join(T, f)
    d = json.load(open(p))
    k0 = next(iter(d))
    print('  ', f, os.path.getsize(p), 'n=', len(d), 'first_key=', k0,
          'fields=', list(d[k0].keys()))
