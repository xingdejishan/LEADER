"""Same rotation transplant, but on the raw search pose (yaw sweep undone).

Answers whether the sweep and the transplant are additive or overlapping.
"""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
MARGIN = 0.10


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    keys = w['hard_keys']

    z = np.load(os.path.join(MOE, 'results', 'yaw_sweep.npz'), allow_pickle=True)
    smap = {str(k): np.array(p, float) for k, p in zip(z['keys'], z['poses'])}
    diag = [json.loads(l) for l in open(os.path.join(MOE, 'results', 'yaw_sweep_diag.jsonl'))]

    raw, hybrid = [], []
    used = 0
    for i, k in enumerate(keys):
        P = smap[str(k)].copy()
        d = diag[i]
        if d.get('take'):
            applied = Rotation.from_euler('z', d['dyaw'], degrees=True).as_matrix()
            if d.get('dpitch') or d.get('droll'):
                applied = applied \
                    @ Rotation.from_euler('y', d['dpitch'], degrees=True).as_matrix() \
                    @ Rotation.from_euler('x', d['droll'], degrees=True).as_matrix()
            P[:3, :3] = P[:3, :3] @ applied.T          # undo the sweep -> raw search pose
        raw.append(P.copy())

        H = P.copy()
        rec = json.load(open(VERIFY % i))['poses']
        base = ((rec.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
        best = None
        for name, row in rec.items():
            if not name.startswith('candidate_'):
                continue
            sc = (row.get('fair') or {}).get('geometry_score')
            if sc is None:
                continue
            if best is None or sc > best[0]:
                best = (float(sc), np.asarray(row['pose'], float))
        if best is not None and base is not None and best[0] > float(base) + MARGIN:
            H[:3, :3] = best[1][:3, :3]
            used += 1
        hybrid.append(H)

    print('frames transplanted: %d' % used)
    su.evaluate(np.stack(raw), None, w, 'raw-search')
    su.evaluate(np.stack(hybrid), None, w, 'transplant-only')


if __name__ == '__main__':
    main()
