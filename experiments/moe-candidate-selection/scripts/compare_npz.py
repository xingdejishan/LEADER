"""Compare two pose tables exactly. Read-only."""
import sys

import numpy as np

a, b = sys.argv[1], sys.argv[2]
za = np.load(a, allow_pickle=True)
zb = np.load(b, allow_pickle=True)
ka = [strk for strk in map(str, za['keys'])]
kb = [strk for strk in map(str, zb['keys'])]
pa = np.asarray(za['poses'], float)
pb = np.asarray(zb['poses'], float)
print('A %s  n=%d' % (a, len(ka)))
print('B %s  n=%d' % (b, len(kb)))
print('keys identical:', ka == kb)
if ka != kb:
    diff = [(i, x, y) for i, (x, y) in enumerate(zip(ka, kb)) if x != y]
    print('  first key mismatches:', diff[:3])
if pa.shape == pb.shape:
    d = np.abs(pa - pb)
    print('bitwise identical poses:', bool(np.array_equal(pa, pb)))
    print('max |diff| = %.3e' % d.max())
    per = d.reshape(len(pa), -1).max(1)
    print('frames with any diff: %d' % int((per > 0).sum()))
    print('frames with diff > 1e-9: %d' % int((per > 1e-9).sum()))
    print('frames with diff > 1e-3: %d' % int((per > 1e-3).sum()))
    if (per > 1e-9).any():
        w = np.argsort(per)[::-1][:5]
        for i in w:
            print('  frame %d %s diff=%.3e' % (i, ka[i], per[i]))
