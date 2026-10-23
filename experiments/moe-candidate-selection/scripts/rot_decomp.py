"""Decompose the hard-frame rotation error into body-frame roll / pitch / yaw."""
import importlib.util
import json

import numpy as np
from scipy.spatial.transform import Rotation

SCORER = '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py'
spec = importlib.util.spec_from_file_location('su', SCORER)
su = importlib.util.module_from_spec(spec)
spec.loader.exec_module(su)

w = su.load_world()
keys = w['hard_keys']
gt = np.asarray(w['hard_gt'], float)
base = np.asarray(w['hard_base'], float)
print('frames %d' % len(keys))


def decompose(pred, name):
    Rp, Rg = pred[:, :3, :3], gt[:, :3, :3]
    Rd = np.swapaxes(Rp, -1, -2) @ Rg
    rv = Rotation.from_matrix(Rd).as_rotvec()
    ang = np.degrees(np.linalg.norm(rv, axis=1))
    deg = np.degrees(rv)
    print('\n== %s ==' % name)
    print('mean angle      : %.4f deg   (median %.4f, p90 %.4f)'
          % (ang.mean(), np.median(ang), np.percentile(ang, 90)))
    print('mean |x| |y| |z|: %.4f  %.4f  %.4f  (body roll / pitch / yaw)'
          % (np.abs(deg[:, 0]).mean(), np.abs(deg[:, 1]).mean(), np.abs(deg[:, 2]).mean()))
    print('median|axis|    : %.4f  %.4f  %.4f'
          % (np.median(np.abs(deg[:, 0])), np.median(np.abs(deg[:, 1])), np.median(np.abs(deg[:, 2]))))
    tot = (deg ** 2).sum(1).mean()
    print('share of squared error: x %.1f%%  y %.1f%%  z %.1f%%'
          % (100 * (deg[:, 0] ** 2).mean() / tot, 100 * (deg[:, 1] ** 2).mean() / tot,
             100 * (deg[:, 2] ** 2).mean() / tot))
    for lo, hi, tag in ((0, 2, '<=2 deg'), (2, 8, '2-8 deg'), (8, 1e9, '>8 deg')):
        m = (ang > lo) & (ang <= hi)
        if m.sum() < 5:
            continue
        print('  band %-9s n=%4d  mean angle %6.3f  |roll| %5.3f  |pitch| %5.3f  |yaw| %5.3f'
              % (tag, int(m.sum()), ang[m].mean(), np.abs(deg[m, 0]).mean(),
                 np.abs(deg[m, 1]).mean(), np.abs(deg[m, 2]).mean()))
    return ang, deg


ang_b, deg_b = decompose(base, 'refined_V7 baseline (T0)')

# the shipped three-route output
z = np.load('/home/zhangcy/projects/e2e-moe-20261020-v1/results/three_route.npz', allow_pickle=True)
k2p = {str(k): p for k, p in zip(z['keys'], z['poses'])}
tr = np.stack([k2p[k] for k in keys])
ang_t, deg_t = decompose(tr, 'three-route (current MoE output)')

# is the yaw error tied to the along-track ambiguity ratio?
ratio = np.array([json.loads(l)['ratio'] for l in
                  open('/home/zhangcy/projects/e2e-moe-20261020-v1/results/three_route_diag.jsonl')])
ok = np.isfinite(ratio)
print('\ncorrelation of |yaw error| with along-track ambiguity ratio: r = %.3f (n=%d)'
      % (np.corrcoef(np.abs(deg_b[ok, 2]), ratio[ok])[0, 1], int(ok.sum())))
print('mean |yaw| when ratio<0.21: %.3f deg (n=%d)   ratio>=0.21: %.3f deg (n=%d)'
      % (np.abs(deg_b[ok & (ratio < 0.21), 2]).mean(), int((ok & (ratio < 0.21)).sum()),
         np.abs(deg_b[ok & (ratio >= 0.21), 2]).mean(), int((ok & (ratio >= 0.21)).sum())))
