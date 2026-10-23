"""Add a yaw search to the along-track branch, then score it.

The search only moves rotation, so translation -- and therefore MPE -- is exactly
unchanged by construction.  The point is MOE.

    stage 1  the existing pipeline (gate2 -> along-track x lateral search) -> T3
    stage 2  coarse yaw sweep +-20 deg at 2 deg on 1500 points
    stage 3  fine yaw sweep +-2 deg at 0.5 deg on 4000 points around the coarse best
    stage 4  accept only if the fine optimum beats the current cost by MARGIN

Writes results/yaw_sweep.npz and scores both the swept and the un-swept table with
the frozen scorer, so the comparison is on one ruler.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

MOE = '/home/zhangcy/projects/e2e-moe-20261020-v1'
sys.path.insert(0, os.path.join(MOE, 'stages/fast_search'))
sys.path.insert(0, MOE)

import run_pipeline as RP
import pipeline as P

COARSE = np.arange(-20.0, 20.001, 2.0)
FINE = np.arange(-2.0, 2.001, 0.5)
RP_GRID = np.arange(-2.0, 2.001, 1.0)
CP, FP = 1500, 4000
MARGIN = 0.005
_G = {}


def init(margin, rp=0):
    RP._OPTS.update(gate2=0.01)
    RP.init()
    _G['margin'] = margin
    _G['rp'] = rp


def cost(tree, q, R, t):
    return float(np.minimum(tree.query(q @ R.T + t, k=1, workers=1)[0], 1.0).mean())


def one(i):
    _, T3, diag = RP.one(i)
    t, R0 = T3[:3, 3].copy(), T3[:3, :3].copy()
    cfg = RP._G['cfg']
    scan = P.prepare_scan(RP._G['frames'][i], cfg)
    qc = scan if len(scan) <= CP else scan[::max(1, len(scan) // CP)][:CP]
    qf = scan if len(scan) <= FP else scan[::max(1, len(scan) // FP)][:FP]
    mp = RP._G['maps'].local(t, RP._G['maps'].nearby(t)[:cfg.reference_count])
    tree = cKDTree(mp)

    cc = np.array([cost(tree, qc, R0 @ Rotation.from_euler('z', d, degrees=True).as_matrix(), t)
                   for d in COARSE])
    d0 = float(COARSE[int(np.argmin(cc))])
    cf = np.array([cost(tree, qf, R0 @ Rotation.from_euler('z', d, degrees=True).as_matrix(), t)
                   for d in (d0 + FINE)])
    k = int(np.argmin(cf))
    d_best = float(d0 + FINE[k])
    c_best = float(cf[k])
    dp = dr = 0.0
    if _G.get('rp'):
        Rz = Rotation.from_euler('z', d_best, degrees=True).as_matrix()
        for p in RP_GRID:
            for r in RP_GRID:
                R = R0 @ Rz @ Rotation.from_euler('y', p, degrees=True).as_matrix() \
                        @ Rotation.from_euler('x', r, degrees=True).as_matrix()
                c = cost(tree, qf, R, t)
                if c < c_best:
                    c_best, dp, dr = c, float(p), float(r)
    c_now = cost(tree, qf, R0, t)
    take = bool(c_best < c_now - _G['margin'])

    T = T3.copy()
    if take:
        R = R0 @ Rotation.from_euler('z', d_best, degrees=True).as_matrix()
        if dp or dr:
            R = R @ Rotation.from_euler('y', dp, degrees=True).as_matrix() \
                  @ Rotation.from_euler('x', dr, degrees=True).as_matrix()
        T[:3, :3] = R
    return i, T, dict(key=diag['key'], gain=diag['gain'], ran_fine=diag['ran_fine'],
                      dyaw=float(d_best), dpitch=dp, droll=dr,
                      take=take, c_now=float(c_now), c_best=float(c_best))


def _w(i):
    return one(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--margin', type=float, default=MARGIN)
    ap.add_argument('--rp', type=int, default=0, help='1: also search roll/pitch (+-2 deg)')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    init(a.margin, a.rp)
    n = len(RP._G['frames'])
    idx = list(range(n if a.limit <= 0 else min(a.limit, n)))
    if a.workers > 1:
        with Pool(a.workers, initializer=init, initargs=(a.margin, a.rp)) as p:
            res = p.map(_w, idx, chunksize=8)
    else:
        res = [_w(i) for i in idx]

    out = os.path.join(MOE, 'results', 'yaw_sweep.npz')
    keys = [str(k) for k in np.load(os.path.join(MOE, 'inputs/initial.npz'),
                                    allow_pickle=True)['keys']][:len(idx)]
    poses = np.stack([r[1] for r in res])
    np.savez(out, keys=np.array(keys), poses=poses)
    with open(os.path.join(MOE, 'results', 'yaw_sweep_diag.jsonl'), 'w') as fh:
        for _, _, d in res:
            fh.write(json.dumps(d) + '\n')

    take = np.array([r[2]['take'] for r in res])
    dy = np.array([abs(r[2]['dyaw']) for r in res])
    print('frames %d   accepted %d (%.1f%%)   median |dyaw| on accepted %.2f deg'
          % (len(idx), int(take.sum()), 100.0 * take.mean(), float(np.median(dy[take])) if take.any() else float('nan')))

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'su', '/home/zhangcy/projects/r-unified-eval-20261018-v1/score_unified.py')
    su = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(su)
    w = su.load_world()
    # un-swept table: recover T3 by undoing exactly the rotation the sweep applied
    un = []
    for _, T, d in res:
        U = np.array(T, float)
        if d['take']:
            applied = Rotation.from_euler('z', d['dyaw'], degrees=True).as_matrix()
            if d.get('dpitch') or d.get('droll'):
                applied = applied \
                    @ Rotation.from_euler('y', d['dpitch'], degrees=True).as_matrix() \
                    @ Rotation.from_euler('x', d['droll'], degrees=True).as_matrix()
            U[:3, :3] = U[:3, :3] @ applied.T
        un.append(U)
    su.evaluate(np.stack(un), None, w, 'no-sweep')
    su.evaluate(poses, None, w, 'sweep')


if __name__ == '__main__':
    main()
