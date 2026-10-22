"""Is the ground-truth attitude itself noisy?

Everything about MOE assumes the truth is exact.  The truth attains its up-vector
move 1.19 deg between consecutive frames, which is a lot for a car, so this checks
whether that motion is a smooth physical signal or white noise: for white noise
the lag-1 autocorrelation of the first differences is about -0.5, for a smooth
signal it is close to 0 (and physical oscillation shows up as a periodic pattern).
"""
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
t0 = np.asarray(Z['t0'], float)
gt = np.stack([truth[tuple(k.split('/'))] for k in keys])

order = np.array(sorted(range(len(keys)),
                        key=lambda i: (keys[i].split('/')[0], int(keys[i].split('/')[1]))))
ts = np.array([int(keys[i].split('/')[1]) for i in order], dtype=np.int64)
dt = np.diff(ts) / 1e6
cont = dt < 0.6
print('frame spacing median %.3f s ; %d gaps' % (np.median(dt), int((~cont).sum())))


def upvec(R):
    return R[:, 2]


def heading(R):
    return np.unwrap(np.arctan2(R[1, 0], R[0, 0]))


def autocorr_diff(x, lags=8):
    d = np.diff(x)
    d = d - d.mean()
    out = []
    for L in range(1, lags + 1):
        a, b = d[:-L], d[L:]
        out.append(float((a * b).mean() / (d.var() + 1e-18)))
    return out


for tag, R in (('truth', gt[order, :3, :3]), ('estimate', t0[order, :3, :3])):
    U = np.array([upvec(r) for r in R])
    # body-frame-ish components of the up vector (small angles)
    ux, uy = U[:, 0], U[:, 1]
    H = heading(R)
    print('\n## %s' % tag)
    print('   consecutive change: |du| %.3f deg ; |dheading| %.3f deg'
          % (np.degrees(np.arccos(np.clip((U[:-1] * U[1:]).sum(1), -1, 1))).mean(),
             np.degrees(np.abs(np.diff(H))).mean()))
    for name, x in (('up x', ux), ('up y', uy), ('heading', H), ('up z', U[:, 2])):
        ac = autocorr_diff(x, 8)
        print('   %-8s first-diff autocorr lags1..4: %s   (white ~ -0.50)'
              % (name, ' '.join('%+.2f' % v for v in ac[:4])))

# a cleaner white-noise test on the truth up-vector: how much of its variance is
# removed by a 5-frame local quadratic fit (a white component is removed at once)
U = np.array([upvec(r) for r in gt[order, :3, :3]])
for k in (3, 5, 9):
    half = k // 2
    res = []
    for i in range(len(U)):
        lo, hi = max(0, i - half), min(len(U), i + half + 1)
        idx = [j for j in range(lo, hi) if abs(j - i) < 20 and cont[min(j, i)]]
        if len(idx) < k:
            continue
        p = np.polyfit(np.array(idx, float) - i, U[idx], 2)
        res.append(np.linalg.norm(p[2] - U[i]))
    print('\n   truth up-vector: residual of a local quadratic fit over %d frames: '
          'mean %.4f (a smooth signal leaves ~0)' % (k, np.mean(res)))

# and: is the estimate's error correlated with the truth's own acceleration?
Ue = np.array([upvec(r) for r in t0[order, :3, :3]])
E = np.degrees(np.arccos(np.clip((Ue * U).sum(1), -1, 1)))
acc = np.zeros(len(U))
acc[1:-1] = np.linalg.norm(U[2:] - 2 * U[1:-1] + U[:-2], axis=1)
print('\n   corr(|tilt error|, |truth tilt second difference|) = %+.3f'
      % np.corrcoef(E, np.degrees(acc))[0, 1])
