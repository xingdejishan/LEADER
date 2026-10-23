import json
import os
import sys
import numpy as np

ROOT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, '/home/zhangcy/projects/r-unified-eval-20261018-v1')
import score_unified as S

z = np.load(S.SCORED, allow_pickle=True)
canon = [str(k) for k in z['keys']]
print('canon', len(canon), canon[:3], type(z['keys'][0]))
p = os.path.join(ROOT, 'results/three_route_merged.npz')
zz = np.load(p, allow_pickle=True)
pk = [str(k) for k in zz['keys']]
print('pred', len(pk), pk[:3], type(zz['keys'][0]))
inter = set(canon) & set(pk)
print('overlap', len(inter))
print('canon-only', [k for k in canon if k not in set(pk)][:3])
print('pred-only', [k for k in pk if k not in set(canon)][:3])
