import json, os, collections
import numpy as np

z = np.load('/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/initial.npz', allow_pickle=True)
keys = [str(x) for x in z['keys']]
dates = collections.Counter(k.split('/')[0] for k in keys)
print('N', len(keys), 'DATES', dict(dates))

# DEV_DATES
src = '/home/zhangcy/projects/leader-far-covolume-standard-20261001/source-v1/data/leader_nclt_manifest.py'
txt = open(src).read()
import re
for line in txt.splitlines():
    if 'DEV_DATES' in line or 'TEST_DATES' in line or 'TRAIN_DATES' in line:
        print('SRC', line.strip()[:200])

mp = '/data2/zhangcy/leader-v2-full-trajectory-20261001/NCLT/2012-02-12/manifest.json'
print('manifest size', os.path.getsize(mp))
m = json.load(open(mp))
print('manifest type', type(m))
if isinstance(m, dict):
    print('keys', list(m.keys())[:20])
