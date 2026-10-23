"""Budget-constrained hybrid candidate generator, measured on the 40 dumped frames.

Candidate pool = per-reference candidates from the matcher (fast PnP, as shipped)
               + one global RANSAC+LM solution on the aggregated correspondences.
Dedupe by SE(3), cap at the same slot count as the baseline, then Oracle Recall@13.
All inputs come from the same run (arm v8_noprune, --fast-pnp 1), so matches, binding and
intrinsics are identical for both halves.
"""
import json
import os
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

STORE = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1'
OUT = '/dev/shm/match_out'
sys.path.insert(0, STORE)
sys.path.insert(0, '/home/zhangcy/projects/sp-lightglue-hard1502-20261013-v1')
import common as C

T_BC = np.asarray(C.load(C.STORE / 'library.json')['T_BC'], float)
CAP = 13
DEDUP_T, DEDUP_R = 1.5, 3.0


def ang(a, b):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(a.T @ b).as_rotvec())))


def hit(P, g):
    return (np.linalg.norm(P[:3, 3] - g[:3, 3]) <= 1.0 and ang(P[:3, :3], g[:3, :3]) <= 2.0)


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    keys = [str(k) for k in w['hard_keys']]
    gt = {k: np.asarray(p, float) for k, p in zip(keys, w['hard_gt'])}
    dump = np.load('/dev/shm/corr_dump.npy', allow_pickle=True).item()

    base_hit = hyb_hit = 0
    n = 0
    added = 0
    t_ransac = []
    for idx, d in sorted(dump.items()):
        path = os.path.join(OUT, '%06d.json' % idx)
        if not os.path.isfile(path):
            continue
        rec = json.load(open(path))
        base = [np.asarray(h['pose'], float) for h in (rec.get('hypotheses') or [])]
        g = gt[keys[idx]]
        n += 1
        base_hit += any(hit(P, g) for P in base[:CAP])

        world = np.asarray(d['world'], float)
        pixels = np.asarray(d['pixels'], float)
        ids = np.asarray(d['ids'])
        t0 = time.perf_counter()
        sol = C.solve_pnp(world, pixels, ids, T_BC) if len(world) >= 8 else None
        t_ransac.append(time.perf_counter() - t0)
        pool = list(base)
        if sol is not None and sol.get('qualified'):
            P = np.asarray(sol['pose'], float)
            if all(np.linalg.norm(P[:3, 3] - q[:3, 3]) > DEDUP_T or ang(P[:3, :3], q[:3, :3]) > DEDUP_R
                   for q in pool):
                pool = [P] + pool
                added += 1
        pool = pool[:CAP]
        hyb_hit += any(hit(P, g) for P in pool)

    print('frames %d   median baseline candidates %.0f' % (n, np.median([1])))
    print('baseline (matcher pool, capped %d)      Oracle Recall@13: %d/%d = %.0f%%'
          % (CAP, base_hit, n, 100 * base_hit / n))
    print('hybrid (+ global RANSAC, deduped, cap %d) Oracle Recall@13: %d/%d = %.0f%%'
          % (CAP, hyb_hit, n, 100 * hyb_hit / n))
    print('frames where the RANSAC pose survived dedupe and entered the pool: %d' % added)
    print('global RANSAC per-frame: median %.1f ms  mean %.1f ms'
          % (1000 * np.median(t_ransac), 1000 * np.mean(t_ransac)))


if __name__ == '__main__':
    main()
