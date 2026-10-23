import json, os
import numpy as np

S = '/home/zhangcy/.dsh_scratch/rot-coupling-subset.npz'
z = np.load(S, allow_pickle=True)
keys = [str(k) for k in z['keys']]
gt, ep0 = z['gt'], z['ep0']
pose0, poseA, poseB = z['pose_baseline'], z['pose_A'], z['pose_B']
J = json.load(open(S.replace('.npz', '.json')))
rows = J['rows']


def ang(A, G):
    R = A[:3, :3] @ G[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


m = np.where((ep0 > 5) & (ep0 <= 8))[0]
sv = np.array([rows[i]['diag']['B']['s'] for i in m])
sb = np.array([rows[i]['diag']['baseline']['s'] for i in m])
stuck = m[np.abs(sv) < 1e-9]
moved = m[np.abs(sv) >= 1e-9]
print('baseline s==0 on stuck:', int((np.abs(sb[np.abs(sv) < 1e-9]) < 1e-9).sum()), '/', len(stuck))
for name, idx in (('stuck', stuck), ('moved', moved)):
    for v, p in (('base', pose0), ('A', poseA), ('B', poseB)):
        e = np.array([np.linalg.norm(p[i][:3, 3] - gt[i][:3, 3]) for i in idx])
        ok = np.array([(e[k] <= 1) and ang(p[i], gt[i]) <= 2 for k, i in enumerate(idx)])
        print('  %-5s %-4s n=%d MPE %.2f ok %.2f >1m %d >5m %d'
              % (name, v, len(idx), e.mean(), ok.mean(), (e > 1).sum(), (e > 5).sum()))
