"""Best of both: the search pose everywhere, the whole visual pose where the margin rule fires."""
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

    z = np.load(os.path.join(MOE, 'results', 'yaw_sweep.npz'), allow_pickle=True)
    smap = {str(k): np.array(p, float) for k, p in zip(z['keys'], z['poses'])}

    for margin in (0.05, 0.10, 0.15):
        out = []
        used = 0
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
            if best is not None and base is not None and best[0] > float(base) + margin:
                P = best[1].copy()
                used += 1
            out.append(P)
        rows = su.evaluate(np.stack(out), None, w, 'search+whole m=%.2f n=%d' % (margin, used),
                           verbose=False)
        r = rows['hard:all1502']
        print('margin %.2f  frames %3d   MPE %.4f   MOE %.4f   ok1/2 %d   ok3/8 %d   >10m %d'
              % (margin, used, r['MPE'], r['MOE'], r['success_1m_2deg'], r['success_3m_8deg'],
                 r['over_10m']))


if __name__ == '__main__':
    main()
