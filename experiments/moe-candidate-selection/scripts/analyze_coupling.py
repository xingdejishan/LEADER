import json, os
import numpy as np

S = '/home/zhangcy/.dsh_scratch/rot-coupling-subset.npz'
z = np.load(S, allow_pickle=True)
keys = [str(k) for k in z['keys']]
gt = z['gt']; t0 = z['t0']; ep0 = z['ep0']; hard = z['hard']
P = {n: z['pose_' + n] for n in ('baseline', 'A', 'B')}
sec = {n: z['sec_' + n] for n in ('baseline', 'A', 'B')}
J = json.load(open(S.replace('.npz', '.json')))
rows = J['rows']
assert len(rows) == len(keys)


def ang(A, G):
    R = A[:3, :3] @ G[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def ep(p, i):
    return float(np.linalg.norm(p[i, :3, 3] - gt[i, :3, 3]))


E = {n: np.array([ep(P[n], i) for i in range(len(keys))]) for n in P}
ER = {n: np.array([ang(P[n][i], gt[i]) for i in range(len(keys))]) for n in P}
ER0 = np.array([ang(t0[i], gt[i]) for i in range(len(keys))])
E['t0'] = ep0

B = {'>0.5m': ep0 > 0.5, 'control': ep0 <= 0.5,
     '0.5-2m': (ep0 > 0.5) & (ep0 <= 2), '2-5m': (ep0 > 2) & (ep0 <= 5),
     '5-8m': (ep0 > 5) & (ep0 <= 8), '>8m': ep0 > 8}

print('== front-end rotation error by band (deg): mean / median / p90 ==')
for k, m in B.items():
    if m.sum():
        print('  %-8s n=%4d  %.2f / %.2f / %.2f' % (
            k, m.sum(), ER0[m].mean(), np.median(ER0[m]), np.percentile(ER0[m], 90)))

print('\n== ok(1m,2deg) and error distribution by band ==')
for k, m in B.items():
    if not m.sum():
        continue
    for n in ('baseline', 'A', 'B'):
        ok = ((E[n] <= 1) & (ER[n] <= 2))[m].mean()
        print('  %-8s %-8s ok=%.3f  MPE=%.2f  P50=%.2f  P90=%.2f  >1m=%d >5m=%d'
              % (k, n, ok, E[n][m].mean(), np.median(E[n][m]),
                 np.percentile(E[n][m], 90), (E[n][m] > 1).sum(), (E[n][m] > 5).sum()))

print('\n== transitions in 5-8m band (43 frames): baseline ok vs variant ok ==')
m = B['5-8m']
bok = ((E['baseline'] <= 1) & (ER['baseline'] <= 2))
for n in ('A', 'B'):
    vok = ((E[n] <= 1) & (ER[n] <= 2))
    print('  %-2s  base_ok&%s_ok=%d  base_ok&%s_fail=%d  base_fail&%s_ok=%d  both_fail=%d'
          % (n, n, int((bok & vok)[m].sum()), n, int((bok & ~vok)[m].sum()),
             n, int((~bok & vok)[m].sum()), int((~bok & ~vok)[m].sum())))
print('  baseline: ok=%d fail=%d' % (int(bok[m].sum()), int((~bok[m]).sum())))

print('\n== control regressions: frames ok under baseline but not under variant ==')
c = B['control']
for n in ('A', 'B'):
    vok = ((E[n] <= 1) & (ER[n] <= 2))
    print('  %-2s  lost=%d  gained=%d  net=%+d  dER mean=%+.2f deg'
          % (n, int((bok & ~vok)[c].sum()), int((~bok & vok)[c].sum()),
             int((vok.astype(int) - bok.astype(int))[c].sum()),
             float((ER[n] - ER['baseline'])[c].mean())))

print('\n== 5-8m band detail: yaw chosen by B, per-frame errors ==')
order = np.where(m)[0]
print('  yaw histogram:', {d: int(sum(1 for i in order if abs(rows[i]['diag']['B']['yaw'] - d) < 1e-6))
                           for d in (-3, -2, -1, 0, 1, 2, 3)})
print('   idx  ep0   base   A      B     yaw   er0   s_B   d_B')
for i in order:
    print('  %4d %5.2f %5.2f %5.2f %5.2f %6.1f %5.2f %6.2f %5.2f'
          % (i, ep0[i], E['baseline'][i], E['A'][i], E['B'][i],
             rows[i]['diag']['B']['yaw'], ER0[i], rows[i]['diag']['B']['s'],
             rows[i]['diag']['B']['d']))

print('\n== cost (mean truncated residual) by band, mean ==')
for k in ('>0.5m', 'control', '5-8m'):
    mm = B[k]
    vals = {n: np.array([rows[i]['diag'][n]['cost'] for i in range(len(keys))]) for n in ('baseline', 'A')}
    vals['B'] = np.array([rows[i]['diag']['B']['cost'] for i in range(len(keys))])
    print('  %-8s base=%.4f A=%.4f B=%.4f' % (k, vals['baseline'][mm].mean(),
                                              vals['A'][mm].mean(), vals['B'][mm].mean()))

print('\n== 2-5m band: yaw / alternating effect ==')
m2 = B['2-5m']
print('  n=%d  base MPE %.2f A %.2f B %.2f' % (m2.sum(), E['baseline'][m2].mean(),
                                               E['A'][m2].mean(), E['B'][m2].mean()))
print('  base>1m %d  A>1m %d  B>1m %d' % ((E['baseline'][m2] > 1).sum(),
                                          (E['A'][m2] > 1).sum(), (E['B'][m2] > 1).sum()))

print('\n== seconds/frame (mean over all) ==')
print('  baseline %.3f  A %.3f  B %.3f  (prep %.3f)'
      % (sec['baseline'].mean(), sec['A'].mean(), sec['B'].mean(),
         np.mean([r['prep'] for r in rows])))
