"""Three-branch orchestrator: LEADER gives T, three experts propose, a gate picks.

    T_leader ──┬─ route 2  visual candidate (frozen: retrieval+match+PnP+laser check)
               │                  accepted only if the frame is severely ambiguous AND
               │                  the candidate beats the baseline evidence by 0.10
               └─ route 3  V7/CoFAR pose -> gate 2 -> along-track search -> translation correction

    final = route 2 if accepted else route 3

Route 1 (LEADER) and route 2 (visual) are taken from frozen artifacts here; route 3 is
computed live by this script.  Nothing is written outside this directory.

    python three_route.py --workers 12 --gate2 0.01
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'stages/fast_search'))
sys.path.insert(0, os.path.join(HERE, 'stages/gate1'))
import fast_lib as F  # noqa: E402
import pipeline as P  # noqa: E402
import gate as G1  # noqa: E402

ROOT = HERE
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
VIS_MARGIN = 0.10
AMB_THRESHOLD = 0.21
_G = {}
_OPT = {}


def init():
    frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
    z = np.load(os.path.join(ROOT, 'inputs/initial.npz'), allow_pickle=True)
    cfg = P.Config()
    _G.update(frames=frames, cfg=cfg, poses=np.asarray(z['poses'], float),
              maps=F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                               radius=cfg.crop_radius),
              lib=G1.ReferenceLibrary())
    vis = []
    for i in range(len(frames)):
        rec = json.load(open(VERIFY % i))['poses']
        base = ((rec.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
        best = None
        for name, row in rec.items():
            if not name.startswith('candidate_'):
                continue
            fair = row.get('fair') or {}
            sc = fair.get('geometry_score')
            if sc is None:
                continue
            if best is None or sc > best[0]:
                best = (float(sc), np.asarray(row['pose'], float))
        vis.append((best, base))
    _G['vis'] = vis


def cost(tree, pts):
    return float(np.minimum(tree.query(pts, k=1, workers=1)[0], 1.0).mean())


def one(i):
    cfg = _G['cfg']
    T0 = _G['poses'][i]
    maps = _G['maps']
    mp = maps.local(T0[:3, 3], maps.nearby(T0[:3, 3])[:cfg.reference_count])
    scan = P.prepare_scan(_G['frames'][i], cfg)
    tree = cKDTree(mp)
    u = P.along_track_direction(T0)
    lat = np.cross(u, [0.0, 0.0, 1.0]); lat = lat / np.linalg.norm(lat)
    qc = scan[:min(cfg.coarse_points, len(scan))]

    # --- gate 1: along-track ambiguity of the incoming pose
    amb = G1.ambiguity_ratio(_G['lib'], _G['frames'][i]['scan_path'], T0)
    ratio = amb['ratio']

    # --- route 2: frozen visual candidate, accepted only when severe + margin
    best, base = _G['vis'][i]
    vis_ok = bool(best is not None and base is not None
                  and np.isfinite(ratio) and ratio < AMB_THRESHOLD
                  and best[0] > float(base) + VIS_MARGIN)

    # --- route 3: gate 2 -> search -> translation correction
    c0 = cost(tree, qc @ T0[:3, :3].T + T0[:3, 3])
    home = [cost(tree, qc @ T0[:3, :3].T + T0[:3, 3] + d * lat) for d in F.D_HOME]
    d_home = float(F.D_HOME[int(np.argmin(home))])
    best_c = np.inf
    for s in F.S_COARSE:
        for d in np.clip(d_home + F.D_COARSE, -2.0, 2.0):
            c = cost(tree, qc @ T0[:3, :3].T + T0[:3, 3] + s * u + d * lat)
            best_c = min(best_c, c)
    gain = c0 - best_c
    T3 = np.array(T0, float)
    ran = False
    if gain > _OPT['gate2']:
        pose, dg = P.localize(_G['frames'][i], T0, maps, cfg)
        T3 = np.asarray(pose, float)
        ran = True
    final = best[1] if vis_ok else T3
    diag = dict(key=_G['frames'][i]['key'], ratio=float(ratio), gain=float(gain),
                ran_fine=bool(ran), visual=vis_ok, chosen='visual' if vis_ok else 'route3')
    return i, final, diag


def _w(i):
    return one(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--gate2', type=float, default=0.01)
    ap.add_argument('--out', default='results/three_route_no_rotation.npz')
    a = ap.parse_args()
    _OPT.update(gate2=a.gate2)
    init()
    n = len(_G['frames'])
    with Pool(a.workers, initializer=init) as p:
        res = p.map(_w, range(n), chunksize=4)
    keys = [str(k) for k in np.load(os.path.join(ROOT, 'inputs/initial.npz'),
                                    allow_pickle=True)['keys']]
    poses = np.stack([r[1] for r in res])
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    np.savez(os.path.join(ROOT, a.out), keys=np.array(keys), poses=poses)
    with open(os.path.join(ROOT, 'results/three_route_no_rotation_diag.jsonl'), 'w') as fh:
        for _, _, dg in res:
            fh.write(json.dumps(dg) + '\n')
    print('frames %d  visual %d  fine %d  gate2 %.3f'
          % (n, sum(r[2]['visual'] for r in res), sum(r[2]['ran_fine'] for r in res), a.gate2))


if __name__ == '__main__':
    main()
