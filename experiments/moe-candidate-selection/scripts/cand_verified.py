"""Is the visual rotation bad because it is vision, or because the matches are wrong?

Splits the candidates by the LiDAR verification flag that the verify table already
carries (geometry_verified) and reports the rotation error in each group.  If the
verified ones are accurate, the problem is match quality and selection, not the
modality.
"""
import collections
import json

import numpy as np
from scipy.spatial.transform import Rotation

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

    def rot_err(a, b):
        Rd = np.swapaxes(a, -1, -2) @ b
        return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Rd).as_rotvec())))

    groups = collections.defaultdict(list)
    per_frame_best_verified = []
    ch_stats = collections.defaultdict(list)
    frames_with_verified = 0

    for i in range(len(gt)):
        rec = json.load(open(VERIFY % i))['poses']
        any_verified = False
        best_verified = None
        for name, row in rec.items():
            if not name.startswith('candidate_'):
                continue
            P = np.asarray(row['pose'], float)
            e = rot_err(P[:3, :3], gt[i][:3, :3])
            verified = bool(row.get('geometry_verified'))
            groups['verified' if verified else 'not verified'].append(e)
            ch_stats[str(row.get('channel'))].append(e)
            if verified:
                any_verified = True
                sc = (row.get('fair') or {}).get('geometry_score')
                if best_verified is None or (sc is not None and sc > best_verified[0]):
                    best_verified = (sc if sc is not None else -9, e, P)
        if any_verified:
            frames_with_verified += 1
            per_frame_best_verified.append(best_verified[1])

    for k, v in groups.items():
        v = np.array(v)
        print('%-14s candidates %5d   rotation error  mean %6.2f  median %6.2f  p25 %5.2f  p75 %6.2f'
              % (k, len(v), v.mean(), np.median(v), np.percentile(v, 25), np.percentile(v, 75)))
    for k, v in ch_stats.items():
        v = np.array(v)
        print('  channel %-8s n=%5d  mean %6.2f  median %6.2f' % (k, len(v), v.mean(), np.median(v)))

    pv = np.array(per_frame_best_verified)
    print()
    print('frames with at least one verified candidate: %d / %d' % (frames_with_verified, len(gt)))
    if len(pv):
        print('using the best verified candidate per frame: rotation error mean %.2f  median %.2f  p75 %.2f'
              % (pv.mean(), np.median(pv), np.percentile(pv, 75)))


if __name__ == '__main__':
    main()
