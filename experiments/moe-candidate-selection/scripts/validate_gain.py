"""Second fidelity check: my gain vs the online ledger's gain, same frames."""
import json
import os

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
R = P + '/results/'
led = {}
for name in ('full-shard0.jsonl', 'fullb-shard1.jsonl', 'fullb-shard3.jsonl'):
    p = R + name
    if not os.path.exists(p):
        continue
    for line in open(p):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        led[r['key']] = r

z = np.load('/home/zhangcy/.dsh_scratch/sweep_full_partial.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
gain = np.asarray(z['gain'], np.float64)
m = np.isfinite(gain)
print('computed %d frames' % int(m.sum()))
mine, their = [], []
for i in np.where(m)[0]:
    r = led.get(keys[i])
    if r is None or r.get('gain') is None:
        continue
    mine.append(gain[i])
    their.append(float(r['gain']))
mine = np.array(mine)
their = np.array(their)
d = mine - their
print('n=%d  max|diff| %.3e  mean|diff| %.3e  exact %d' % (len(d), np.abs(d).max(),
      np.abs(d).mean(), int((np.abs(d) < 1e-12).sum())))
print('mine  median %.5f  share>0.005 %.1f%%  share>0.01 %.1f%%  share>0.02 %.1f%%  share>0.05 %.1f%%'
      % (np.median(mine), 100 * (mine > 0.005).mean(), 100 * (mine > 0.01).mean(),
         100 * (mine > 0.02).mean(), 100 * (mine > 0.05).mean()))
print('theirs full-set: ', end='')
allg = np.array([float(r['gain']) for r in led.values() if r.get('gain') is not None])
print('n=%d median %.5f  >0.005 %.1f%%  >0.01 %.1f%%  >0.02 %.1f%%  >0.03 %.1f%%  >0.05 %.1f%%'
      % (len(allg), np.median(allg), 100 * (allg > 0.005).mean(), 100 * (allg > 0.01).mean(),
         100 * (allg > 0.02).mean(), 100 * (allg > 0.03).mean(), 100 * (allg > 0.05).mean()))
# my gain on the same t0-error strata as the whole set would need the full run; report here
