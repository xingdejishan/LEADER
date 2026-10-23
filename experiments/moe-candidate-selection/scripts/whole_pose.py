"""If the candidate's position is also better on the selected frames, take the whole pose."""
import json

import numpy as np

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

    cand = {}
    base_score = {}
    for i in range(len(gt)):
        rec = json.load(open(VERIFY % i))['poses']
        base = ((rec.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
        base_score[i] = None if base is None else float(base)
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

    def build(mode, margin):
        out = []
        used = 0
        for i in range(len(gt)):
            P = v7[i].copy()
            b = cand[i]
            if b is not None and base_score[i] is not None and b[0] > base_score[i] + margin:
                used += 1
                if mode == 'rot':
                    P[:3, :3] = b[1][:3, :3]
                elif mode == 'pos':
                    P[:3, 3] = b[1][:3, 3]
                else:
                    P = b[1].copy()
            out.append(P)
        return np.stack(out), used

    for mode, name in (('rot', 'rotation only'), ('pos', 'position only'), ('both', 'whole pose')):
        for m in (0.10,):
            poses, used = build(mode, m)
            rows = su.evaluate(poses, None, w, '%s m=%.2f n=%d' % (name, m, used), verbose=False)
            r = rows['hard:all1502']
            print('%-14s margin %.2f  frames %3d   MPE %.4f   MOE %.4f   ok1/2 %d   ok3/8 %d'
                  % (name, m, used, r['MPE'], r['MOE'], r['success_1m_2deg'], r['success_3m_8deg']))


if __name__ == '__main__':
    main()
