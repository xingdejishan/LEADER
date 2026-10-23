"""Does retrieval inherit the pose error?  Nearest retrieved reference vs the pose error."""
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
    v7 = np.asarray(w['hard_base'], float)

    refs = json.load(open(os.path.join(MOE, 'inputs/refs.json')))
    rpos = np.asarray([np.asarray(r['world_with_body'], float)[:3, 3]
                       if 'world_with_body' in r else np.asarray(r['world_from_body'], float)[:3, 3]
                       for r in refs])

    perr, dmin = [], []
    for i in range(len(gt)):
        m = json.load(open(MATCH % i))
        pr = [int(x) for x in (m.get('pair_references') or [])]
        if not pr:
            continue
        d = np.linalg.norm(rpos[pr] - gt[i][:3, 3], axis=1)
        perr.append(float(np.linalg.norm(v7[i][:3, 3] - gt[i][:3, 3])))
        dmin.append(float(d.min()))

    perr = np.array(perr); dmin = np.array(dmin)
    print('frames %d   correlation(pose error, nearest ref distance) = %+.3f'
          % (len(perr), np.corrcoef(perr, dmin)[0, 1]))
    print()
    print('%-18s %5s %10s %12s %12s' % ('V7 position error', 'n', 'median dmin', 'share no ref<10m', 'share dmin<=2m'))
    for lo, hi in ((0, 1), (1, 5), (5, 20), (20, 100), (100, 1e9)):
        m = (perr > lo) & (perr <= hi) if lo else (perr <= hi)
        if m.sum() == 0:
            continue
        print('%-18s %5d %10.2f %12.0f%% %12.0f%%'
              % ('%g-%g m' % (lo, hi) if hi < 1e8 else '>100 m', int(m.sum()),
                 np.median(dmin[m]), 100 * (dmin[m] > 10).mean(), 100 * (dmin[m] <= 2).mean()))


if __name__ == '__main__':
    main()
