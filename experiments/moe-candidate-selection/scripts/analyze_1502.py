import json, sys
import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/initial.npz'
z = np.load(P, allow_pickle=True)
print('KEYS', list(z.keys()))
for k in z.keys():
    a = z[k]
    print('  %-18s %-10s %s' % (k, a.dtype, a.shape))
    if a.dtype == object or a.ndim <= 2 and a.dtype.kind in 'OUS':
        try:
            flat = a.reshape(-1)
            print('    sample', repr(flat[0])[:160])
        except Exception as e:
            print('    sample-err', e)

# try to find the key list
cand = None
for k in z.keys():
    a = z[k]
    if a.dtype == object and a.reshape(-1)[0] is not None and isinstance(a.reshape(-1)[0], (tuple, list, str)):
        cand = (k, a)
        break
if cand:
    k, a = cand
    flat = list(a.reshape(-1))
    print('KEYLIST field', k, 'n', len(flat))
    print('  first3', [repr(x) for x in flat[:3]])
    print('  last1', repr(flat[-1]))
    dates = {}
    for x in flat:
        d = x[0] if isinstance(x, (tuple, list)) else str(x).split('/')[0]
        dates[d] = dates.get(d, 0) + 1
    print('  dates', dates)
print('FRAMES_JSON_head')
try:
    j = json.load(open('/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/frames.json'))
    print(' type', type(j), 'len', len(j))
    s = j if isinstance(j, list) else list(j.items())[:2]
    print(' sample', repr(s[:2])[:400])
except Exception as e:
    print(' frames.json err', e)
