"""Write the fixed 100-frame upper-bound subset index list (registered in PREREGISTRATION)."""
import numpy as np

OUT = '/data2/zhangcy/r4-full-20261019-v1'
z = np.load('/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/scored.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
dates = np.asarray(z['dates']).astype(str)
order = sorted(range(len(keys)), key=lambda i: (dates[i], keys[i]))
pos = np.unique(np.linspace(0, len(keys) - 1, 100).astype(int))
idx = [order[p] for p in pos]
open(OUT + '/subset100.txt', 'w').write(','.join(map(str, idx)))
np.save(OUT + '/subset100.npy', np.array(idx))
from collections import Counter
print('n=', len(idx), Counter(dates[i] for i in idx))
print(idx)
