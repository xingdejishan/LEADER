"""Subset PnP: several pose hypotheses from coherent groups of the aggregated correspondences.

Baseline: the candidates the frozen pipeline produced for the same frames.
New: cluster the aggregated 2D-3D correspondences by 3D location, run one PnP per
cluster, deduplicate, and measure whether the resulting small set contains a pose within
1 m / 2 deg more often (Oracle Recall@13).
"""
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

STORE = '/data2/zhangcy/sp-lightglue-hard1502-20261013-v1'
MATCH = os.path.join(STORE, 'prov/match-frozen_nopruneopt/%06d.json')
sys.path.insert(0, STORE)
sys.path.insert(0, '/home/zhangcy/projects/sp-lightglue-hard1502-20261013-v1')
import common as C

T_BC = np.asarray(C.load(C.STORE / 'library.json')['T_BC'], float)
EPS, MIN_SAMPLES, MAX_CAND = 2.0, 8, 13


def ang(a, b):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(a.T @ b).as_rotvec())))


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    gt = {str(k): np.asarray(p, float) for k, p in zip(w['hard_keys'], w['hard_gt'])}

    dump = np.load('/dev/shm/corr_dump.npy', allow_pickle=True).item()
    print('frames in dump: %d' % len(dump))

    def recall(cands, g):
        if not cands:
            return False
        for P in cands:
            if (np.linalg.norm(P[:3, 3] - g[:3, 3]) <= 1.0
                    and ang(P[:3, :3], g[:3, :3]) <= 2.0):
                return True
        return False

    base_hit = new_hit = 0
    n_base, n_new = [], []
    for idx, d in sorted(dump.items()):
        key = [str(k) for k in w['hard_keys']][idx]
        g = gt[key]
        world = np.asarray(d['world'], float)
        pixels = np.asarray(d['pixels'], float)
        ids = np.asarray(d['ids'])
        if len(world) < 8:
            continue

        # baseline: the frozen candidates of this frame
        m = json.load(open(MATCH % idx))
        base = [np.asarray(h['pose'], float) for h in (m.get('hypotheses') or [])][:MAX_CAND]
        base_hit += recall(base, g)
        n_base.append(len(base))

        # new: one PnP per coherent 3D cluster
        lab = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(world)
        cands = []
        for c in sorted(set(lab)):
            if c < 0:
                continue
            sel = lab == c
            if sel.sum() < 8:
                continue
            sol = C.solve_pnp(world[sel], pixels[sel], ids[sel], T_BC)
            if sol is not None and sol.get('qualified'):
                cands.append(np.asarray(sol['pose'], float))
        # plus the single global solution, as the aggregated channel does today
        sol = C.solve_pnp(world, pixels, ids, T_BC)
        if sol is not None and sol.get('qualified'):
            cands.append(np.asarray(sol['pose'], float))
        # deduplicate
        kept = []
        for P in cands:
            if all(np.linalg.norm(P[:3, 3] - q[:3, 3]) > 1.5 or ang(P[:3, :3], q[:3, :3]) > 3.0
                   for q in kept):
                kept.append(P)
        kept = kept[:MAX_CAND]
        n_new.append(len(kept))
        new_hit += recall(kept, g)

    n = len(n_base)
    print('frames scored: %d' % n)
    print('baseline (frozen candidates)  Oracle Recall@13: %d/%d = %.0f%%   median candidates %.0f'
          % (base_hit, n, 100 * base_hit / n, np.median(n_base)))
    print('subset PnP (coherent clusters) Oracle Recall@13: %d/%d = %.0f%%   median candidates %.0f'
          % (new_hit, n, 100 * new_hit / n, np.median(n_new)))


if __name__ == '__main__':
    main()
