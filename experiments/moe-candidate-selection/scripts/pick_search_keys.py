"""Pick frames where the search actually runs, for the defaults verification."""
import json

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/results/'
z = np.load(P + 'full-online-final.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
gain = np.asarray(z['gain'], float)
run = gain > 0.01
sel = []
for date in sorted(set(k.split('/')[0] for k in keys)):
    idx = [i for i in range(len(keys)) if run[i] and keys[i].split('/')[0] == date]
    step = max(1, len(idx) // 15)
    sel.extend([keys[i] for i in idx[::step][:15]])
with open('/home/zhangcy/.dsh_scratch/keys_search.json', 'w') as fh:
    json.dump(sel, fh)
print('selected %d frames that trigger the search (%s)' % (len(sel), sorted(set(k.split('/')[0] for k in sel))))
