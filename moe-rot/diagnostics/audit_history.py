"""Audit the project history for prior attempts at the three proposed routes."""
import os
import re
from collections import Counter, defaultdict

ROOTS = ['/data2/zhangcy', '/home/zhangcy/projects']
KEYS = {
    'normal/法向': r'normal|法向',
    'gravity/重力': r'gravity|重力',
    'ground/地面': r'\bground\b|地面',
    'geodesic/测地': r'geodesic|测地',
    'procrustes': r'procrustes',
    'rotation loss': r'rotation[_ ]loss|pose[_ ]loss|L_rot',
    'observability': r'observab|可观测|弱方向|weak direction',
    'manhattan': r'manhattan',
}
EXT = ('.md', '.py')

hits = defaultdict(list)
counts = Counter()
files = 0
for root in ROOTS:
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count('/')
        if depth > 2:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in ('vendor', '__pycache__', 'data', 'NCLT', '.git')]
        for fn in filenames:
            if not fn.endswith(EXT):
                continue
            p = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(p) > 2_000_000:
                    continue
                text = open(p, encoding='utf-8', errors='ignore').read()
            except OSError:
                continue
            files += 1
            for name, pat in KEYS.items():
                if re.search(pat, text, re.I):
                    counts[name] += 1
                    hits[name].append(p)

print('scanned %d md/py files under %s' % (files, ' and '.join(ROOTS)))
for name in KEYS:
    print('\n== %-14s %4d files' % (name, counts[name]))
    seen = Counter()
    for p in hits[name]:
        rel = p.replace('/data2/zhangcy/', '').replace('/home/zhangcy/projects/', 'P:')
        top = rel.split('/')[0]
        seen[top] += 1
    for top, c in seen.most_common(10):
        print('   %-55s %d' % (top, c))
