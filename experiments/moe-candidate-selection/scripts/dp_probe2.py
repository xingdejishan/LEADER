#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Probe 2: dump the along-track correction code + gate + eval, for local reading."""
import os
import shutil
import sys

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
OUT = '/home/zhangcy/.dsh_scratch/_dp_src'
os.makedirs(OUT, exist_ok=True)

want = [
    'e2e_online.py',
    'stages/fast_search/fast_lib.py',
    'stages/fast_search/adaptive.py',
    'stages/fast_search/pipeline.py',
    'stages/fast_search/apply_gate.py',
    'stages/fast_search/refine_fast.py',
    'stages/fast_search/refine_rot.py',
    'stages/ambiguity.py',
    'run_pipeline.py',
    'README.md',
    'docs/fast-search.md',
]
for rel in want:
    src = os.path.join(P, rel)
    if os.path.exists(src):
        dst = os.path.join(OUT, rel.replace('/', '__'))
        shutil.copyfile(src, dst)
        print('copied', rel, os.path.getsize(dst))
    else:
        print('MISSING', rel)

# grep for the along-track / gate constants
import re
print('\n[grep along/yaw/gate in e2e_online.py]')
txt = open(os.path.join(P, 'e2e_online.py'), encoding='utf-8').read()
for i, ln in enumerate(txt.splitlines(), 1):
    if re.search(r'along|ALONG|yaw|YAW|gate|GATE|margin|MARGIN|depth|DEPTH', ln):
        print('%5d %s' % (i, ln))
