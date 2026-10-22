import json
import os

import numpy as np

P = '/home/zhangcy/projects/e2e-moe-20261020-v1'
T = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
truth = {}
for f in os.listdir(T):
    d = f[:-5]
    for ts, v in json.load(open(os.path.join(T, f))).items():
        truth[(d, str(ts))] = np.asarray(v['T_WB'], float)
Z = np.load(P + '/results/full-online-final.npz', allow_pickle=True)
keys = [str(k) for k in Z['keys']]
Pz = np.asarray(Z['poses'], float)
gt = np.stack([truth[tuple(k.split('/'))] for k in keys])
et = np.linalg.norm(Pz[:, :3, 3] - gt[:, :3, 3], axis=1)
R = np.swapaxes(Pz[:, :3, :3], -1, -2) @ gt[:, :3, :3]
er = np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
print('MPE>1 m : %.2f%%' % (100 * (et > 1).mean()))
print('MOE>2 deg: %.2f%%' % (100 * (er > 2).mean()))
print('both    : %.2f%%' % (100 * ((et > 1) & (er > 2)).mean()))
print('ok(1m,2d): %.2f%%' % (100 * ((et <= 1) & (er <= 2)).mean()))
