"""Two experts side by side: the control-flow change, and nothing else.

The cascade (three_route_unified.py, mode unified) reads

    geometry branch -> T3                     always
    ambiguity ratio  -> rho                   always
    visual branch    -> T_vis   only when  rho < 0.21  and a candidate exists
    arbitration      -> k * sigma_C switch between T3 and T_vis, on gated frames only

This driver keeps every one of those pieces and changes one thing: the visual hypothesis
is produced on EVERY frame, the ratio is still computed and still written to the
diagnostics, but it no longer stops the visual expert from running.  Concretely the block

    if best is not None and np.isfinite(ratio) and ratio < AMB_THRESHOLD:

becomes

    if best is not None:            # --visual all : rho is a report field only
        ...
    else:                           # --visual gate: the cascade's condition, verbatim
        ...

Everything else in one() is the cascade's one() line for line: same map crop, same scan
preparation, same coarse surface, same adaptive fine search, same rotation block, same
ABR.decide() call with the same union evidence and the same k.  The visual hypothesis is
read by calling the cascade's own loader three_route_unified.loading_hook, so on the
frames where the gate used to fire the two structures see the identical tuple.

Modes
    --visual all    both experts on every frame (the parallel structure)
    --visual gate   the cascade's gating, for a bit-identity check against mode unified
    --visual none   geometry only; must equal three_route_unified.py --mode route3 bit for bit

    python parallel_hypo.py --visual all --workers 6 --out /tmp/parallel_all.npz
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'stages/fast_search'))
sys.path.insert(0, os.path.join(HERE, 'stages/gate1'))
sys.path.insert(0, os.path.join(HERE, 'stages'))

import fast_lib as F                 # noqa: E402
import pipeline as P                 # noqa: E402
import ambiguity as A                # noqa: E402
import arbitration as ARB            # noqa: E402
import three_route_unified as TU     # noqa: E402

ROOT = HERE
COST_TRUNC = 1.0
ARB_POINTS = ARB.ARB_POINTS
AMB_THRESHOLD = A.THRESHOLD
DEFAULT_K = 8.4
_G = {}
_OPT = {}


def init():
    frames = json.load(open(os.path.join(ROOT, 'inputs/frames.json')))
    refs = json.load(open(os.path.join(ROOT, 'inputs/refs.json')))
    z = np.load(os.path.join(ROOT, 'inputs/initial.npz'), allow_pickle=True)
    cfg = P.Config(solver=_OPT.get('solver', 'adaptive'),
                   coarse_points=_OPT.get('coarse_points', 1000),
                   fine_points=_OPT.get('fine_points', 1500),
                   workers=_OPT.get('threads', 4))
    _G.update(frames=frames, cfg=cfg, poses=np.asarray(z['poses'], float),
              maps=F.MapSource(refs,
                               np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                               radius=cfg.crop_radius))
    if _OPT['visual'] == 'none':
        return
    # the cascade's own accessor: highest fair score among the frame's frozen candidates
    _G['vis'] = [TU.loading_hook(i) for i in range(len(frames))]


def one(i):
    cfg = _G['cfg']
    T0 = _G['poses'][i]
    maps = _G['maps']
    rec = _G['frames'][i]
    mp = maps.local(T0[:3, 3], maps.nearby(T0[:3, 3])[:cfg.reference_count])
    scan = P.prepare_scan(rec, cfg)
    tree = cKDTree(mp)
    u = P.along_track_direction(T0)
    lat = np.cross(u, [0.0, 0.0, 1.0])
    lat = lat / np.linalg.norm(lat)

    # --- stage A: the coarse surface, built once (cascade, verbatim)
    cs = F.coarse_stage(mp, scan, T0, u, workers=1,
                        n_coarse=cfg.coarse_points, topk=cfg.topk)
    qc = scan[:min(cfg.coarse_points, len(scan))]
    c_zero = TU.cost(cs['tree'], cs['base_c'] + T0[:3, 3])
    gain = c_zero - float(cs['surface'].min())

    T3 = np.array(T0, float)
    ran = False
    if gain > _OPT['gate2']:
        pose, _ = P.localize(rec, T0, maps, cfg, coarse=cs, scan=scan)
        T3 = np.asarray(pose, float)
        ran = True

    rotated = False
    if _OPT['rot']:
        q = scan[np.linalg.norm(scan, axis=1) <= 50]
        cells = np.floor(q / 0.3).astype(np.int64)
        _, ids = np.unique(cells, axis=0, return_index=True)
        T_rot = TU.rot_only(tree, mp, q[np.sort(ids)], T3)
        rotated = bool(np.abs(T_rot[:3, :3] - T3[:3, :3]).max() > 1e-12)
        T3 = T_rot

    diag = dict(key=rec['key'], gain=float(gain), ran_fine=bool(ran), rotated=rotated,
                visual_mode=_OPT['visual'])

    if _OPT['visual'] == 'none':
        diag.update(visual=False, chosen='route3')
        return i, T3, diag

    # --- the proposal gate is measured on every frame and is a report field only
    _, amb = A.decide(cs['tree'], scan, T0, u)
    ratio = float(amb['ratio'])
    diag['ratio'] = ratio
    diag['gate_open'] = bool(np.isfinite(ratio) and ratio < AMB_THRESHOLD)

    best = _G['vis'][i][0]
    if _OPT['visual'] == 'all':
        asked = best is not None
    else:                                   # --visual gate : the cascade's condition
        asked = best is not None and np.isfinite(ratio) and ratio < AMB_THRESHOLD
    diag['asked'] = bool(asked)
    if not asked:
        diag.update(visual=False, chosen='route3', c_route3=float('nan'),
                    c_visual=float('nan'))
        return i, T3, diag

    cv_centre = best[1][:3, 3]
    mp_v = maps.local(cv_centre, maps.nearby(cv_centre)[:cfg.reference_count])
    evidence = ARB.Evidence(ARB.union_maps(mp, mp_v), scan, trunc=COST_TRUNC,
                            n_points=ARB_POINTS)
    hyps = [ARB.Hypothesis(T3, 'route3', {'incumbent': True, 'ratio': float(ratio)}),
            ARB.Hypothesis(best[1], 'visual', {'external_score': float(best[0])})]
    chosen, arb = ARB.decide(hyps, evidence, _OPT['k'], scan=scan)
    accepted = chosen.source == 'visual'
    diag.update(arb_sigma=arb['sigma_c'], arb_sigma_std=arb['sigma_c_std'],
                arb_sigma_n=arb['sigma_c_n'], arb_thresh=arb['thresh'],
                arb_margin=arb['margin'], arb_k=arb['k'], arb_best=arb['best'],
                arb_switched=arb['switched'], arb_decided=arb['decided'],
                arb_n_map=evidence.n_map,
                arb_dist_v=float(np.linalg.norm(cv_centre - T0[:3, 3])),
                visual_fair=float(best[0]))
    c3 = arb['costs']['route3']
    cv = arb['costs']['visual']
    diag.update(c_route3=float(c3), c_visual=float(cv), dcost=float(cv - c3),
                visual=bool(accepted), chosen='visual' if accepted else 'route3')
    return i, (best[1] if accepted else T3), diag


def _w(i):
    return one(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--visual', choices=('all', 'gate', 'none'), default='all')
    ap.add_argument('--solver', choices=('adaptive', 'prune', 'full'), default='adaptive')
    ap.add_argument('--coarse-points', type=int, default=1000)
    ap.add_argument('--fine-points', type=int, default=1500)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--gate2', type=float, default=-1.0)
    ap.add_argument('--k', type=float, default=DEFAULT_K)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--out', default='/tmp/parallel_all.npz')
    a = ap.parse_args()
    _OPT.update(visual=a.visual, gate2=a.gate2, k=a.k, rot=True, solver=a.solver,
                coarse_points=a.coarse_points, fine_points=a.fine_points,
                threads=a.threads, stride=a.stride)
    n = len(json.load(open(os.path.join(ROOT, 'inputs/frames.json'))))
    idx = list(range(0, n if a.limit <= 0 else min(a.limit, n), a.stride))
    if a.workers > 1:
        with Pool(a.workers, initializer=init) as p:
            res = p.map(_w, idx, chunksize=4)
    else:
        init()
        res = [_w(i) for i in idx]
    keys = [str(k) for k in np.load(os.path.join(ROOT, 'inputs/initial.npz'),
                                    allow_pickle=True)['keys']][:len(idx) * a.stride]
    keys = [keys[i] for i in idx]
    poses = np.stack([r[1] for r in res])
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + '.tmp.npz'
    np.savez(tmp, keys=np.array(keys), poses=poses)
    os.replace(tmp, out)
    diag_path = os.path.splitext(out)[0] + '_diag.jsonl'
    with open(diag_path + '.tmp', 'w') as fh:
        for _, _, dg in res:
            fh.write(json.dumps(dg) + '\n')
    os.replace(diag_path + '.tmp', diag_path)
    asked = sum(1 for r in res if r[2].get('asked'))
    adopted = sum(1 for r in res if r[2].get('visual'))
    gate = sum(1 for r in res if r[2].get('gate_open'))
    print('visual=%s frames=%d gate_open=%d asked=%d adopted=%d k=%.2f -> %s'
          % (a.visual, len(idx), gate, asked, adopted, a.k, out))


if __name__ == '__main__':
    main()
