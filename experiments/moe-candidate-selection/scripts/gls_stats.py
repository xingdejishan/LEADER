import numpy as np
z = np.load('/data2/zhangcy/epipolar-gls-comparison-20261014-v1/errors.npz', allow_pickle=True)
arms = [k for k in z.keys() if k.endswith('_rotation')]
base = z['baseline_rotation']
print('frames', len(base))
print('%-38s %8s %8s %8s %8s' % ('arm', 'mean', 'median', 'p90', 'worse'))
print('%-38s %8.4f %8.4f %8.4f %8s' % ('baseline', base.mean(), np.median(base), np.percentile(base, 90), '-'))
for k in arms:
    if k == 'baseline_rotation':
        continue
    a = z[k]
    d = a - base
    worse = int((d > 1e-9).sum())
    print('%-38s %8.4f %8.4f %8.4f %8d   (delta %+0.4f, better on %d)' %
          (k.replace('_rotation', ''), a.mean(), np.median(a), np.percentile(a, 90), worse, d.mean(), int((d < -1e-9).sum())))
# paired bootstrap on the most interesting arm
rng = np.random.default_rng(0)
for k in ('epipolar_rotation', 'rematched_gls_rotation', 'frozen_gls_rotation'):
    d = z[k] - base
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print('%-30s paired delta %+0.4f  95%% CI [%+0.4f, %+0.4f]  %s' %
          (k.replace('_rotation',''), d.mean(), lo, hi, 'significant' if (lo < 0 and hi < 0) or (lo > 0 and hi > 0) else 'crosses zero'))
