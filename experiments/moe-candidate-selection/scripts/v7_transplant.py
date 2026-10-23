"""V7 translation + transplanted visual rotation. No LiDAR search at all."""
import json
import os

import numpy as np

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    keys = w['hard_keys']
    v7 = np.asarray(w['hard_base'], float)

    out = []
    used = 0
    for i, k in enumerate(keys):
        P = v7[i].copy()
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
        if best is not None and base is not None and best[0] > float(base) + 0.10:
            P[:3, :3] = best[1][:3, :3]
            used += 1
        out.append(P)
    print('rotation transplanted on %d frames' % used)
    su.evaluate(v7, None, w, 'v7-only')
    su.evaluate(np.stack(out), None, w, 'v7+rot-transplant')


if __name__ == '__main__':
    main()
