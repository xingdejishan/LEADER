#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only inspect: where do images / SAM cache live for the v7 runtime?"""
import inspect
import json
import os
import sys

V7FULL = '/home/zhangcy/projects/v7-full-20261006'
sys.path.insert(0, V7FULL)

import v7_runtime as vr

print('[vr] file=', vr.__file__)
src = inspect.getsource(vr)
print('[vr] len(src)=', len(src))
for kw in ('sam', 'cache', 'image', 'root', 'velodyne', 'DATASET', 'data2'):
    hits = [ln for ln in src.splitlines() if kw.lower() in ln.lower()]
    print('=== kw=%s hits=%d' % (kw, len(hits)))
    for h in hits[:40]:
        print('   ', h.rstrip())
