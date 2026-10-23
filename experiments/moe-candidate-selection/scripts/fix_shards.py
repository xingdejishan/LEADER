"""Check shard integrity after the disk-full crash; drop shards that do not load."""
import os
import sys

import numpy as np

SH = '/home/zhangcy/.dsh_scratch/sweep_rot'
good = bad = 0
frames = 0
for name in sorted(f for f in os.listdir(SH) if f.endswith('.npz')):
    p = os.path.join(SH, name)
    try:
        z = np.load(p)
        rot = z['rot']
        idx = z['idx']
        assert rot.shape[1] == 7 and rot.shape[2:] == (4, 4), rot.shape
        assert len(rot) == len(idx) and int(idx[-1]) - int(idx[0]) == len(idx) - 1
        good += 1
        frames += len(idx)
    except Exception as exc:
        bad += 1
        print('BAD', name, type(exc).__name__, exc)
        os.remove(p)
print('valid shards %d (%d frames), removed %d' % (good, frames, bad))
