#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only inspect #2: dataset root layout + manifest loader internals."""
import inspect
import json
import os
import sys
from pathlib import Path

DATA = Path('/data2/zhangcy/leader-v2-full-trajectory-20261001')
print('[data] exists=', DATA.exists())
for sub in ('NCLT', 'sam'):
    p = DATA / sub
    print('[data/%s] exists=%s' % (sub, p.exists()))
    if p.exists():
        try:
            print('   entries=', sorted(x.name for x in p.iterdir())[:12])
        except Exception as e:
            print('   err', e)

d = DATA / 'NCLT' / '2012-02-12'
if d.exists():
    print('[2012-02-12] entries=', sorted(x.name for x in d.iterdir())[:12])

s = DATA / 'sam' / '2012-02-12'
if s.exists():
    print('[sam/2012-02-12] entries=', sorted(x.name for x in s.iterdir())[:12])
    mp = s / 'cache_manifest.json'
    if mp.exists():
        blob = json.load(open(str(mp)))
        print('[manifest] type=', type(blob).__name__, 'len=', len(blob))
        if isinstance(blob, dict):
            ks = list(blob.keys())[:5]
            print('[manifest] sample keys=', ks)
            for k in ks[:2]:
                print('  ', k, '->', str(blob[k])[:200])
        else:
            print('[manifest] first=', str(blob[0])[:300])
        # find the file path root
        print('[manifest] raw head=', open(str(mp)).read()[:300])

EVAL = '/home/zhangcy/projects/leader-v5-eval-throughput-20261003'
sys.path.insert(0, EVAL)
import data.leader_nclt_manifest as lum
print('[lum] file=', lum.__file__)
src = inspect.getsource(lum)
print('[lum] len=', len(src))
for i, ln in enumerate(src.splitlines()):
    low = ln.lower()
    if any(k in low for k in ('def load', 'image', 'sam_', 'sam_features', 'cache_manifest', 'suffix', '.png', '.jpg', 'resized', 'rgb')):
        print('  %4d %s' % (i + 1, ln.rstrip()))
