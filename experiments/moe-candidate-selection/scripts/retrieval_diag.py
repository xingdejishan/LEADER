"""Front-end diagnosis: do the frames that fail verification retrieve worse references?

For every frame, compare the retrieved reference set against where the vehicle really is.
If the failing frames systematically retrieve references far from the truth, the bottleneck
is retrieval, not matching or PnP, and no downstream change can fix it.
"""
import json
import os

import numpy as np

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
MATCH = ('/data2/zhangcy/sp-lightglue-hard1502-20261013-v1/prov/'
         'match-frozen_nopruneopt/%06d.json')


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = np.asarray(w['hard_gt'], float)

    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    rpos = np.asarray([np.asarray(r['world_from_body'], float)[:3, 3] for r in refs])

    groups = {'verified': [], 'no_verified': []}
    for i in range(len(gt)):
        m = json.load(open(MATCH % i))
        v = json.load(open(('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
                            'verify-frozen_nopruneoptREF/%06d.json') % i))['poses']
        any_ver = any(bool(row.get('geometry_verified'))
                      for k, row in v.items() if k.startswith('candidate_'))
        pr = [int(x) for x in (m.get('pair_references') or [])]
        if not pr:
            continue
        d = np.linalg.norm(rpos[pr] - gt[i][:3, 3], axis=1)
        rec = dict(n=len(pr), dmin=float(d.min()), dmed=float(np.median(d)),
                   n5=int((d <= 5).sum()), n10=int((d <= 10).sum()))
        groups['verified' if any_ver else 'no_verified'].append(rec)

    print('%-12s %5s %8s %8s %8s %8s %8s' %
          ('group', 'n', 'refs', 'min d', 'median d', 'refs<5m', 'refs<10m'))
    for k, rows in groups.items():
        if not rows:
            continue
        a = {key: np.array([r[key] for r in rows]) for key in rows[0]}
        print('%-12s %5d %8.1f %8.2f %8.2f %8.2f %8.2f'
              % (k, len(rows), a['n'].mean(), a['dmin'].mean(), a['dmed'].mean(),
                 a['n5'].mean(), a['n10'].mean()))

    print()
    for k, rows in groups.items():
        if not rows:
            continue
        a = {key: np.array([r[key] for r in rows]) for key in rows[0]}
        print('%-12s share with a reference within 2 m: %.0f%%   3 m: %.0f%%   none within 10 m: %.0f%%'
              % (k, 100 * (a['dmin'] <= 2).mean(), 100 * (a['dmin'] <= 3).mean(),
                 100 * (a['n10'] == 0).mean()))


if __name__ == '__main__':
    main()
