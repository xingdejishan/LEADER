"""Final integrity check of the delivered artifacts."""
import json
import sys

import numpy as np

OUT = '/data2/zhangcy/r4-full-20261019-v1'
SC = '/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/scored.npz'
z = np.load(SC, allow_pickle=True)
keys = [str(k) for k in z['keys']]
for tag in ('main', 'main_refined', 'truth', 'old'):
    d = np.load('%s/poses_%s.npz' % (OUT, tag), allow_pickle=True)
    ks = [str(k) for k in d['keys']]
    P = d['poses']
    print('%-12s keys=%d unique=%d aligned=%s finite=%s shape=%s'
          % (tag, len(ks), len(set(ks)), ks == keys, bool(np.isfinite(P).all()), P.shape))
d = np.load(OUT + '/poses_main.npz', allow_pickle=True)
P = d['poses']
B = np.asarray(z['refined_V7_poses'], float)
diff = np.linalg.norm(P[:, :3, 3] - B[:, :3, 3], axis=1)
print('main vs deployment translation move: med=%.3f p90=%.3f max=%.3f ; identical frames=%d'
      % (np.median(diff), np.percentile(diff, 90), diff.max(), int((diff < 1e-12).sum())))
print('main rotation identical to deployment:',
      bool(np.allclose(P[:, :3, :3], B[:, :3, :3], atol=1e-12)))
PR = np.load(OUT + '/poses_main_refined.npz', allow_pickle=True)['poses']
print('refined rotation differs from deployment on %d frames'
      % int((~np.isclose(PR[:, :3, :3], B[:, :3, :3], atol=1e-9).all(axis=(1, 2))).sum()))
g = np.load(OUT + '/grids_main.npz', allow_pickle=True)
print('grids shape=%s finite rows=%d' % (g['grids'].shape, int(np.isfinite(g['grids'][:, 0, 0]).sum())))
s = json.load(open(OUT + '/summary_main.json'))
print('summary keys:', sorted(s.keys()))
print('gate:', s['gate'], '| damage:', s['damage'], '| capture %.1f%%' % s['capture_pct'])
print('rotation ceiling:', s['rotation_ceiling'])
