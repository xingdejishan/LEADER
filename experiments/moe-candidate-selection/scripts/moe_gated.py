"""Does the assembled MoE actually fix the 84 gated frames?"""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
HIST = '/data2/zhangcy/native1502-eval-20261015-v1/historical-predictions.npz'


def errs(pred, gt):
    t = np.linalg.norm(pred[:, :3, 3] - gt[:, :3, 3], axis=1)
    Rd = np.swapaxes(pred[:, :3, :3], -1, -2) @ gt[:, :3, :3]
    r = np.degrees(np.linalg.norm(Rotation.from_matrix(Rd).as_rotvec(), axis=1))
    return t, r


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)
    v7 = np.asarray(w['hard_base'], float)

    z = np.load(HIST, allow_pickle=True)
    hk = [str(k) for k in z['keys']]
    pos = {k: i for i, k in enumerate(hk)}
    lead = np.stack([np.asarray(z['historical_LEADER'][pos[str(k)]], float) for k in w['hard_keys']])

    ratio = np.array([json.loads(l).get('ratio', np.nan) for l in
                      open(os.path.join(MOE, 'results', 'three_route_diag.jsonl'))])
    gated = np.isfinite(ratio) & (ratio < 0.21)

    tl, rl = errs(lead, gt)
    tv, rv = errs(v7, gt)
    rows = [('LEADER', tl, rl), ('V7 (deployed)', tv, rv)]
    for f, tag in (('results/three_route.npz', 'three-route (MoE)'),
                   ('results/yaw_sweep.npz', 'search + yaw sweep'),
                   ('results/hybrid_rotation.npz', 'search+sweep+transplant')):
        p = os.path.join(MOE, f)
        if not os.path.isfile(p):
            continue
        zz = np.load(p, allow_pickle=True)
        m = {str(k): np.asarray(q, float) for k, q in zip(zz['keys'], zz['poses'])}
        arr = np.stack([m[str(k)] for k in w['hard_keys']])
        rows.append((tag, *errs(arr, gt)))

    print('%-22s %8s %8s %8s | %8s %8s' % ('', 'all MPE', 'all MOE', 'gated MPE', 'gated MOE', 'gated n'))
    for tag, t, r in rows:
        print('%-22s %8.3f %8.3f %10.2f %10.2f %8d'
              % (tag, t.mean(), r.mean(), t[gated].mean(), r[gated].mean(), int(gated.sum())))
    print()
    print('worst-case on the gated set (max position error):')
    for tag, t, r in rows:
        print('   %-22s max %.1f m   frames >10 m: %d   >100 m: %d'
              % (tag, t[gated].max(), int((t[gated] > 10).sum()), int((t[gated] > 100).sum())))


if __name__ == '__main__':
    main()
