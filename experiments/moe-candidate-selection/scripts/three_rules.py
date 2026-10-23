"""Three selection rules, same whole-pose replacement:
   margin        : candidate score > V7 score + 0.10          (153 frames on hard)
   gate+margin   : the same AND along-track ambiguity < 0.21   (what ships today)
   margin-noGood : margin, but never on frames where V7 is already <= 1 m
"""
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
    gt = np.asarray(w['hard_gt'], float)
    v7 = np.asarray(w['hard_base'], float)
    pos = np.linalg.norm(v7[:, :3, 3] - gt[:, :3, 3], axis=1)
    ratio = np.array([json.loads(l).get('ratio', np.nan) for l in
                      open(os.path.join(MOE, 'results', 'three_route_diag.jsonl'))])

    margins, cand = np.zeros(len(gt), bool), {}
    for i in range(len(gt)):
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
        cand[i] = best
        margins[i] = best is not None and base is not None and best[0] > float(base) + 0.10

    gate = np.isfinite(ratio) & (ratio < 0.21)
    rules = {
        'margin': margins,
        'gate+margin': margins & gate,
        'margin-noGood': margins & (pos > 1.0),
    }
    for name, sel in rules.items():
        out = []
        for i in range(len(gt)):
            out.append(cand[i][1].copy() if (sel[i] and cand[i]) else v7[i].copy())
        rows = su.evaluate(np.stack(out), None, w, name, verbose=False)
        r = rows['hard:all1502']
        print('%-14s frames %3d   MPE %.4f   MOE %.4f   ok1/2 %d   ok3/8 %d'
              % (name, int(sel.sum()), r['MPE'], r['MOE'], r['success_1m_2deg'], r['success_3m_8deg']))


if __name__ == '__main__':
    main()
