"""EXP W2: normals again, but estimated with enough support to mean something.

The first attempt compared |normal angle| between the scan and the merged map with a
12-neighbour PCA, and got 39 deg at both poses - that is the estimator failing, not a
negative result.  Here the scan uses its raw returns (no voxel) and the map uses a
25-neighbour PCA, and only planar neighbourhoods are scored.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

PROJECT = '/home/zhangcy/projects/e2e-moe-20261020-v1'
DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'
TRUTH_DIR = os.path.join(DATA, 'evaluator-only-truth')
sys.path.insert(0, os.path.join(PROJECT, 'stages/fast_search'))
sys.path.insert(0, os.path.join(PROJECT, 'stages'))
import fast_lib as F      # noqa: E402
import pipeline as P      # noqa: E402

OUT = '/home/zhangcy/.dsh_scratch/moe_rot_20261022/expW2_normals.npz'
N = int(os.environ.get('N', '80'))


def normals(pts, k, planar_max):
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)), workers=4)
    nb = pts[idx]
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', nb, nb)
    w, v = np.linalg.eigh(cov)
    planarity = w[:, 0] / np.maximum(w.sum(1), 1e-12)
    return v[:, :, 0], planarity < planar_max


def main():
    refs = json.load(open(os.path.join(PROJECT, 'inputs/refs.json')))
    pos = np.asarray([r['world_from_body'] for r in refs], np.float64)[:, :3, 3]
    maps = F.MapSource(refs, pos, radius=60.0)
    cfg = P.Config()
    blob = np.load(PROJECT + '/results/full-online-final.npz', allow_pickle=True)
    keys = [str(k) for k in blob['keys']]
    t0 = np.asarray(blob['t0'], np.float64)
    truth = {}
    for f in os.listdir(TRUTH_DIR):
        d = f[:-5]
        for ts, v in json.load(open(os.path.join(TRUTH_DIR, f))).items():
            truth[(d, str(ts))] = np.asarray(v['T_WB'], np.float64)
    gt = np.stack([truth[tuple(k.split('/'))] for k in keys])
    e0 = np.linalg.norm(t0[:, :3, 3] - gt[:, :3, 3], axis=1)

    rng = np.random.default_rng(5)
    pick = rng.choice(np.where(e0 < 0.5)[0], N, replace=False)
    rows = []
    for j, i in enumerate(pick):
        key = keys[i]
        date, ts = key.split('/')
        path = os.path.join(DATA, 'NCLT', date, 'velodyne_sync', ts + '.bin')
        raw = F.decode(path)
        r = np.linalg.norm(raw, axis=1)
        raw = raw[(r > 3.0) & (r < 60.0)]
        if len(raw) > 40000:
            raw = raw[::max(1, len(raw) // 40000)]
        n_q, planar_q = normals(raw, 10, 0.02)
        q = raw
        centre = t0[i][:3, 3]
        mp = maps.local(centre, maps.nearby(centre)[:3])
        n_map, planar_m = normals(mp, 25, 0.02)
        tree = cKDTree(mp)
        out = dict(i=int(i), frac_planar_q=float(planar_q.mean()),
                   frac_planar_m=float(planar_m.mean()))
        for tag, P_ in (('t0', t0[i]), ('gt', gt[i])):
            qw = q @ P_[:3, :3].T + P_[:3, 3]
            nw = n_q @ P_[:3, :3].T
            d, idx = tree.query(qw, k=1, workers=4, distance_upper_bound=1.0)
            ok = np.isfinite(d) & planar_q & planar_m[np.clip(idx, 0, len(mp) - 1)]
            if ok.sum() < 50:
                out['n_' + tag] = 0
                out['nang_' + tag] = np.nan
                out['dist_' + tag] = np.nan
                continue
            cos = np.abs((nw[ok] * n_map[idx[ok]]).sum(1))
            out['n_' + tag] = int(ok.sum())
            out['nang_' + tag] = float(np.mean(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
            out['dist_' + tag] = float(np.mean(np.minimum(d[ok], 1.0)))
        rows.append(out)
        if (j + 1) % 20 == 0:
            print('  %d/%d' % (j + 1, N), flush=True)

    good = [r for r in rows if r.get('n_t0', 0) > 0 and r.get('n_gt', 0) > 0]
    print('\n## usable frames %d/%d ; planar fraction of the scan %.2f, of the map %.2f'
          % (len(good), len(rows), np.mean([r['frac_planar_q'] for r in rows]),
             np.mean([r['frac_planar_m'] for r in rows])))
    if not good:
        return
    a = {k: np.array([r.get(k, np.nan) for r in good], float)
         for k in ('nang_t0', 'nang_gt', 'dist_t0', 'dist_gt')}
    print('   matched planar pairs per frame: median %d' % int(np.median([r['n_t0'] for r in good])))
    print('   normal angle  t0 %.2f deg  truth %.2f deg   truth better %d/%d'
          % (a['nang_t0'].mean(), a['nang_gt'].mean(),
             int((a['nang_gt'] < a['nang_t0']).sum()), len(good)))
    print('   position cost t0 %.4f m  truth %.4f m   truth better %d/%d'
          % (a['dist_t0'].mean(), a['dist_gt'].mean(),
             int((a['dist_gt'] < a['dist_t0']).sum()), len(good)))
    np.savez(OUT, **a)
    print('wrote %s' % OUT)


if __name__ == '__main__':
    main()
