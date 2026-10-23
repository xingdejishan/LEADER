"""Build the final hybrid pose table and score it with the frozen scorer.

translation : the along-track search output (with the yaw sweep applied)
rotation    : the visual candidate's rotation, on frames where its geometry score
              beats the baseline's by more than MARGIN

Both halves are single-frame decisions.  MPE cannot move because the translation is
untouched; only MOE is at stake.
"""
import json
import os
import sys

import numpy as np

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

    search = np.load(os.path.join(MOE, 'results', 'yaw_sweep.npz'), allow_pickle=True)
    smap = {str(k): np.array(p, float) for k, p in zip(search['keys'], search['poses'])}

    poses, n_used = [], 0
    for i, k in enumerate(keys):
        P = smap[str(k)].copy()
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
            P[:3, :3] = best[1][:3, :3]
            n_used += 1
        poses.append(P)
    poses = np.stack(poses)
    out = os.path.join(MOE, 'results', 'hybrid_rotation.npz')
    np.savez(out, keys=np.array(keys), poses=poses)
    print('rotation transplanted on %d frames -> %s' % (n_used, out))

    su.evaluate(poses, None, w, 'hybrid-rot')


if __name__ == '__main__':
    main()
