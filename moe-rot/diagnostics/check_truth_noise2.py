"""Truth-attitude smoothness, done right (fixed the heading indexing bug)."""
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
seg = np.concatenate([[0], np.cumsum(~cont)])   # segment id per frame
print('segments %d ; frame spacing median %.3f s' % (seg[-1] + 1, np.median(dt)))


def head(R):
    return np.arctan2(R[:, 1, 0], R[:, 0, 0])


def unwrap_seg(h):
    out = h.copy()
    for s in np.unique(seg):
        m = seg == s
        out[m] = np.unwrap(h[m])
    return out


def ac_diff(x, lags=6):
    d = np.diff(x)
    d = d[np.isfinite(d)]
    d = d - d.mean()
    return [float((d[:-L] * d[L:]).mean() / (d.var() + 1e-18)) for L in range(1, lags + 1)]


for tag, R in (('truth', gt[order, :3, :3]), ('estimate', t0[order, :3, :3])):
    U = R[:, :, 2]
    H = unwrap_seg(head(R))
    print('\n## %s' % tag)
    print('   consecutive |du| %.3f deg ; |dheading| %.3f deg'
          % (np.degrees(np.arccos(np.clip((U[:-1] * U[1:]).sum(1), -1, 1))).mean(),
             np.degrees(np.abs(np.diff(H))).mean()))
    for name, x in (('up x', U[:, 0]), ('up y', U[:, 1]), ('heading', H)):
        print('   %-8s first-diff autocorr: %s  (white ~ -0.50, smooth ~ +)'
              % (name, ' '.join('%+.2f' % v for v in ac_diff(x))))

print('\n## smoothness: residual of a local quadratic fit to the truth up-vector')
U = gt[order, :3, 3 - 1]
for k in (5, 9, 15):
    half = k // 2
    res = []
    for i in range(half, len(U) - half):
        if seg[i] != seg[i - half] or seg[i] != seg[i + half]:
            continue
        x = np.arange(-half, half + 1, dtype=float)
        pts = U[i - half:i + half + 1]
        for c in range(3):
            p = np.polyfit(x, pts[:, c], 2)
            res.append((p[2] - pts[half, c]) ** 2)
    rms = np.degrees(np.sqrt(np.mean(res))) if res else float('nan')
    print('   k=%2d  quadratic-fit residual rms %.4f deg  (smooth => small)' % (k, rms))

Ue = t0[order, :3, :3][:, :, 2]
for k in (5, 9):
    half = k // 2
    res = []
    for i in range(half, len(Ue) - half):
        if seg[i] != seg[i - half] or seg[i] != seg[i + half]:
            continue
        x = np.arange(-half, half + 1, dtype=float)
        pts = Ue[i - half:i + half + 1]
        for c in range(3):
            p = np.polyfit(x, pts[:, c], 2)
            res.append((p[2] - pts[half, c]) ** 2)
    print('   estimate k=%2d residual rms %.4f deg' % (k, np.degrees(np.sqrt(np.mean(res)))))

E = np.degrees(np.arccos(np.clip((U * Ue).sum(1), -1, 1)))
print('\n   tilt-error level %.3f deg' % E.mean())
