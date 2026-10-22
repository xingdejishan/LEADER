"""Find (and optionally remove) truncated projection-cache files.

An earlier run hit ENOSPC while writing these caches, so some npz files are
truncated.  They are regenerable caches under the agent scratch directory.
"""
import os
import sys

import numpy as np

CACHE = '/home/zhangcy/.dsh_scratch/vb/ref-projection'
remove = '--remove' in sys.argv

bad = []
empty = []
for name in sorted(os.listdir(CACHE)):
    if not name.endswith('.npz'):
        continue
    path = os.path.join(CACHE, name)
    size = os.path.getsize(path)
    if size == 0:
        empty.append(path)
        continue
    try:
        with np.load(path, allow_pickle=False) as z:
            _ = z['model'].shape
    except Exception:
        bad.append((path, size))

print('cache %s : %d files' % (CACHE, len(os.listdir(CACHE))))
print('zero-byte: %d' % len(empty))
for p in empty:
    print('   %s' % p)
print('unreadable: %d' % len(bad))
for p, s in bad:
    print('   %s (%d bytes)' % (p, s))
if remove:
    for p in empty + [b[0] for b in bad]:
        os.remove(p)
    print('removed %d files' % (len(empty) + len(bad)))
