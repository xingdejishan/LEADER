"""Attribution: which source produced the correct candidates - the clustered PnP or the
global one?  Same data, same solver; each candidate is tagged with its origin."""
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
EPS, MIN_SAMPLES = 2.0, 8


def ang(a, b):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(a.T @ b).as_rotvec())))


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
    stats = dict(cluster_only=0, global_only=0, both=0, neither=0,
                 n_cluster_cands=0, n_global_ok=0, frames=0, cluster_qualified=0)

    for idx, d in sorted(dump.items()):
        g = gt[keys[idx]]
        world = np.asarray(d['world'], float); pixels = np.asarray(d['pixels'], float)
        ids = np.asarray(d['ids'])
        if len(world) < 8:
            continue
        stats['frames'] += 1

        lab = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(world)
        cl = []
        for c in sorted(set(lab)):
            if c < 0:
                continue
            sel = lab == c
            if sel.sum() < 8:
                continue
            sol = C.solve_pnp(world[sel], pixels[sel], ids[sel], T_BC)
            if sol is not None and sol.get('qualified'):
                cl.append(np.asarray(sol['pose'], float))
        stats['n_cluster_cands'] += len(cl)
        if cl:
            stats['cluster_qualified'] += 1

        gl = None
        sol = C.solve_pnp(world, pixels, ids, T_BC)
        if sol is not None and sol.get('qualified'):
            gl = np.asarray(sol['pose'], float)
            stats['n_global_ok'] += 1

        def hit(P):
            return (np.linalg.norm(P[:3, 3] - g[:3, 3]) <= 1.0
                    and ang(P[:3, :3], g[:3, :3]) <= 2.0)

        hc = any(hit(P) for P in cl)
        hg = gl is not None and hit(gl)
        if hc and hg:
            stats['both'] += 1
        elif hc:
            stats['cluster_only'] += 1
        elif hg:
            stats['global_only'] += 1
        else:
            stats['neither'] += 1

    print('frames %d' % stats['frames'])
    print('cluster PnP produced a qualified solution on %d frames (%d solutions total)'
          % (stats['cluster_qualified'], stats['n_cluster_cands']))
    print('global PnP qualified on %d frames' % stats['n_global_ok'])
    print()
    print('correct-pose hits:  cluster only %d   global only %d   both %d   neither %d'
          % (stats['cluster_only'], stats['global_only'], stats['both'], stats['neither']))


if __name__ == '__main__':
    main()
