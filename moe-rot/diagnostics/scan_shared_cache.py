"""Scan the shared projection cache (read-only) for entries the loader cannot read.

The visual branch loads these with allow_pickle=False; an entry that needs pickle
(or is truncated) breaks the frame that touches it.
"""
import os

import numpy as np

STORE = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/ref-projection'
print('store %s exists=%s' % (STORE, os.path.isdir(STORE)))
if not os.path.isdir(STORE):
    raise SystemExit(0)
names = sorted(os.listdir(STORE))
print('entries %d' % len(names))
bad, empty, ok = [], [], 0
for n in names:
    if not n.endswith('.npz'):
        continue
    p = os.path.join(STORE, n)
    s = os.path.getsize(p)
    if s == 0:
        empty.append((n, s))
        continue
    try:
        with np.load(p, allow_pickle=False) as z:
            _ = z['model'].shape
        ok += 1
    except Exception as exc:                      # noqa: BLE001
        bad.append((n, s, str(exc)[:80]))
print('readable %d ; zero-byte %d ; failing %d' % (ok, len(empty), len(bad)))
for n, s, e in bad[:20]:
    print('   FAIL %s (%d bytes): %s' % (n, s, e))
for n, s in empty[:20]:
    print('   EMPTY %s (%d bytes)' % (n, s))
