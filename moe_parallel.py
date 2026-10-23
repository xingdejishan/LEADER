"""Two experts in parallel, one shared evidence, the calibrated switch: the real MoE.

    geometry expert   the V7 pose plus the along-track / lateral search, translation
                      only, rotation of V7 held: exactly the stages of
                      three_route_unified.py, imported and not modified
    visual expert     visual_expert.PoseFreeVisualExpert, pose-free retrieval: the whole
                      library ranked by the DINOv2 CLS descriptor alone, top 20, no
                      position and no pose anywhere in the query.  6-DoF PnP, no
                      orientation in, pose out.

Acceptance (round 3, the one change)
    The expert's candidate solutions are filtered by the frozen provider's consensus rule:
    a pose survives only if at least PnP.min_supporting_reference_images distinct
    references have a qualified solution within PnP.consensus_translation_m (5 m) and
    PnP.consensus_rotation_deg (10 deg) of it; inside that set the highest fair score
    (geometry_score, the provider's own laser verification) becomes the expert's single
    hypothesis.  No set, no hypothesis: the frame then goes to arbitration with the
    geometry hypothesis alone.  The constants are read from the frozen plan.

What is shared and what is not.  The geometry expert reports a position, and the visual
expert takes exactly that: `geo_position = np.array(T3[:3, 3])`, three numbers copied out
of the pose, used to decide which library frames are worth matching.  The 4x4 pose is not
an argument to anything in the visual path, so the rotation cannot travel with it.  In the
other direction the visual pose reaches the search only as a hypothesis that the
arbitration may or may not keep; it never seeds the search and never edits the rotation.

Arbitration
    evidence   the union of the two crops each hypothesis would use on its own, and one
               query set for both: the same first 4000 prepared scan points.  Chosen over
               a per-hypothesis crop with a density correction because every ingredient
               of the cost is then literally identical for the two hypotheses -- same map
               points, same query points, same truncation, same KD parameters -- so no
               normalisation constant has to be invented.  Duplicate map points cannot
               bias a nearest-neighbour distance, and each hypothesis still gets a full
               crop around itself, so neither is scored blind to its own area.
    switch     arbitration.decide(hypotheses, evidence, k), k = 8.4, sigma_C calibrated
               per frame from 12 tiny perturbations of the incumbent.

Modes
    --visual all    the visual expert runs on every frame (parallel structure, default)
    --visual gate   only where the along-track ambiguity ratio is below 0.21 (cascade)
    --visual none   geometry only, for a same-run baseline
"""
import argparse
import json
import os
import sys
import time

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'

from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'stages/fast_search'))
sys.path.insert(0, os.path.join(HERE, 'stages'))

import fast_lib as F                       # noqa: E402
import pipeline as P                       # noqa: E402
import ambiguity as A                      # noqa: E402
import arbitration as ARB                  # noqa: E402
import three_route_unified as TU            # noqa: E402
import visual_branch as VB                 # noqa: E402
from visual_expert import PoseFreeVisualExpert, POSITION_RADIUS   # noqa: E402

ROOT = HERE
COST_TRUNC = 1.0
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
    if _OPT.get('visual', 'all') != 'none':
        _G['expert'] = PoseFreeVisualExpert(scratch=_OPT['scratch'],
                                            top_k=_OPT['topk'],
                                            device='cpu')


def own_cost(tree, query, pose, trunc=COST_TRUNC):
    """The same statistic on an explicit point set, for the support-set diagnostics."""
    points = query @ pose[:3, :3].T + pose[:3, 3]
    d = tree.query(points, k=1, workers=1, distance_upper_bound=trunc)[0]
    return float(np.where(np.isfinite(d), d, trunc).mean())


def one(i):
    started = time.time()
    cfg = _G['cfg']
    T0 = _G['poses'][i]
    maps = _G['maps']
    rec = _G['frames'][i]
    mp = maps.local(T0[:3, 3], maps.nearby(T0[:3, 3])[:cfg.reference_count])
    scan = P.prepare_scan(rec, cfg)
    tree = cKDTree(mp)
    u = P.along_track_direction(T0)

    # ---- geometry expert: coarse surface, fine search, rotation block (unchanged)
    cs = F.coarse_stage(mp, scan, T0, u, workers=1, n_coarse=cfg.coarse_points,
                        topk=cfg.topk)
    gain = TU.cost(cs['tree'], cs['base_c'] + T0[:3, 3]) - float(cs['surface'].min())
    T3 = np.array(T0, float)
    ran = False
    if gain > _OPT['gate2']:
        pose, _ = P.localize(rec, T0, maps, cfg, coarse=cs, scan=scan)
        T3 = np.asarray(pose, float)
        ran = True
    rotated = False
    if _OPT['rot']:
        q_rot = scan[np.linalg.norm(scan, axis=1) <= 50]
        cells = np.floor(q_rot / 0.3).astype(np.int64)
        _, ids = np.unique(cells, axis=0, return_index=True)
        T_rot = TU.rot_only(tree, mp, q_rot[np.sort(ids)], T3)
        rotated = bool(np.abs(T_rot[:3, :3] - T3[:3, :3]).max() > 1e-12)
        T3 = T_rot

    _, amb = A.decide(cs['tree'], scan, T0, u)
    ratio = float(amb['ratio'])
    diag = dict(key=rec['key'], gain=float(gain), ran_fine=bool(ran),
                rotated=rotated, ratio=ratio, gate_open=bool(ratio < AMB_THRESHOLD),
                visual_mode=_OPT['visual'], k=_OPT['k'],
                accept=_OPT.get('accept', 'consensus'))

    # ---- the only thing handed over: where the geometry expert thinks the frame is.
    # A fresh copy of three numbers.  The pose above is not an argument to the call below
    # and no view of it is created, so nothing about its orientation can be carried along.
    geo_position = np.array(T3[:3, 3], dtype=np.float64)

    # ---- visual expert: a position in, a pose out
    mode = _OPT['visual']
    ask = (mode == 'all') or (mode == 'gate' and ratio < AMB_THRESHOLD)
    visual = None
    t_visual = 0.0
    if ask:
        image = os.path.join(VB.DATA, 'NCLT', rec['key'].split('/')[0], 'cam5',
                             rec['key'].split('/')[1] + '.jpg')
        t_start = time.time()
        try:
            # the frame's own scan, decoded the way the visual branch decodes it, is what
            # the fair score is measured on; the search's prepared subset is not reused
            raw_scan = _G['expert'].branch.C.decode(rec['scan_path'])
            cand = _G['expert'].candidates_posefree(image, scan=raw_scan)
        except Exception as exc:                        # noqa: BLE001
            diag['visual_error'] = repr(exc)
            cand = None
        t_visual = time.time() - t_start
        visual, accepted = accept_consensus(cand, _OPT.get('accept', 'consensus'))
        diag.update(accepted)
    diag['asked'] = bool(ask)
    diag['t_visual'] = round(t_visual, 3)
    if visual is None:
        diag.update(visual=False, chosen='geometry', t_frame=round(time.time() - started, 3))
        return i, T3, diag

    # ---- arbitration on one evidence
    T_v = visual['pose']
    mp_v = maps.local(T_v[:3, 3], maps.nearby(T_v[:3, 3])[:cfg.reference_count])
    evidence = ARB.Evidence(ARB.union_maps(mp, mp_v), scan, trunc=COST_TRUNC,
                            n_points=ARB.ARB_POINTS)
    query = evidence.base
    tree_v = cKDTree(mp_v)
    hyps = [ARB.Hypothesis(T3, 'geometry', {'incumbent': True, 'ratio': ratio}),
            ARB.Hypothesis(T_v, 'visual', dict(visual['detail']))]
    chosen, arb = ARB.decide(hyps, evidence, _OPT['k'], scan=scan)
    adopted = chosen.source == 'visual'
    detail = visual['detail']
    diag.update(visual=bool(adopted), chosen=chosen.source,
                c_geo_union=arb['costs']['geometry'],
                c_vis_union=arb['costs']['visual'],
                c_geo_own=own_cost(tree, query, T3),
                c_vis_own=own_cost(tree_v, query, T_v),
                c_geo_on_geocrop=own_cost(tree, query, T3),
                c_vis_on_geocrop=own_cost(tree, query, T_v),
                arb_margin=arb['margin'], arb_thresh=arb['thresh'],
                arb_sigma=arb['sigma_c'], arb_best=arb['best'],
                arb_switched=arb['switched'], arb_k=arb['k'],
                n_map=evidence.n_map,
                vis_dist=float(np.linalg.norm(T_v[:3, 3] - T3[:3, 3])),
                vis_inliers=detail.get('inliers'),
                vis_consensus=detail.get('consensus'),
                vis_channel=detail.get('channel'),
                vis_capped=detail.get('capped'),
                vis_qualified=detail.get('qualified'),
                vis_nrefs=detail.get('n_refs'),
                vis_pose=[[float(x) for x in row] for row in T_v],
                t_frame=round(time.time() - started, 3))
    return i, (T_v if adopted else T3), diag


def accept_consensus(cand, mode='consensus'):
    """The acceptance, at the point where the visual expert's output becomes a hypothesis.

    consensus (default)  the pose is used only if the frame's own references agree with it:
                         at least PnP.min_supporting_reference_images distinct references
                         whose qualified solutions sit within PnP.consensus_translation_m
                         and PnP.consensus_rotation_deg of it.  Inside that set the highest
                         fair score wins and becomes the expert's single hypothesis.  With
                         no such set the expert says nothing and the arbitration sees the
                         geometry hypothesis alone.
    pnp                  round 1's rule, kept only as a reproduction check: the qualified
                         solution with the most unique inliers, no consensus test.

    The three agreement constants and both pose tolerances are read from the frozen
    provider plan through the branch; nothing here re-types them.
    """
    info = dict(n_hypotheses=0, n_qualified=0, n_accepted=0, n_consensus=0,
                consensus_need=None, consensus_translation_m=None,
                consensus_rotation_deg=None)
    if not cand:
        return None, info
    pnp = _G['expert'].branch.pnp_cfg
    need = int(pnp['min_supporting_reference_images'])
    info.update(n_hypotheses=int(cand['n_hypotheses']), n_qualified=int(cand['n_qualified']),
                n_accepted=int(cand['n_accepted']), consensus_need=need,
                consensus_translation_m=float(pnp['consensus_translation_m']),
                consensus_rotation_deg=float(pnp['consensus_rotation_deg']))
    agreed = [r for r in cand['accepted_rows'] if r['consensus'] >= need]
    info['n_consensus'] = len(agreed)
    base = dict(rule=cand['retrieval']['rule'], top_k=cand['retrieval'].get('top_k'),
                candidates=int(cand['retrieval'].get('candidates', 0)),
                n_refs=len(cand['retrieval'].get('cosine') or []),
                capped=int(cand['capped']), hypotheses=int(cand['n_hypotheses']),
                qualified=int(cand['n_qualified']), accepted=len(cand['accepted_rows']),
                pnp_calls=int(cand['pnp_calls']), batched=bool(cand['batched']))
    if mode == 'pnp':
        rows = list(cand['qualified_rows'])
        if not rows:
            return None, info
        rows.sort(key=lambda r: (-r['unique_query_inliers'], -r['coverage_cells'],
                                 r['median_reprojection_px']))
        best = rows[0]
        info.update(fair=None, consensus=int(best['consensus']),
                    channel=best.get('channel'), n_refs_used=len(best['reference_ids']))
        detail = dict(base, channel=best.get('channel'),
                      inliers=int(best['unique_query_inliers']),
                      inlier_ratio=float(best['inlier_ratio']),
                      coverage_cells=int(best['coverage_cells']),
                      reprojection_px=float(best['median_reprojection_px']),
                      consensus=int(best['consensus']), fair=None,
                      reference_ids=list(best['reference_ids']))
        return dict(pose=np.asarray(best['pose'], dtype=np.float64), detail=detail), info
    if not agreed:
        return None, info
    agreed.sort(key=lambda r: (-r['fair'], -r['consensus'], -r['unique_query_inliers'],
                               r['median_reprojection_px']))
    best = agreed[0]
    info.update(fair=float(best['fair']), fair_all=[float(r['fair']) for r in agreed],
                consensus=int(best['consensus']), channel=best.get('channel'),
                n_refs_used=len(best['reference_ids']))
    detail = dict(base, channel=best.get('channel'),
                  inliers=int(best['unique_query_inliers']),
                  inlier_ratio=float(best['inlier_ratio']),
                  coverage_cells=int(best['coverage_cells']),
                  reprojection_px=float(best['median_reprojection_px']),
                  consensus=int(best['consensus']), fair=float(best['fair']),
                  verified=bool(best['verified']),
                  reference_ids=list(best['reference_ids']))
    return dict(pose=np.asarray(best['pose'], dtype=np.float64), detail=detail), info

def _w(i):
    result = one(i)
    if _OPT.get('verbose'):
        print('frame %d %s done' % (i, result[2].get('key')), flush=True)
    return result


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
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--accept', choices=('consensus', 'pnp'), default='consensus',
                    help='consensus: >=2 references within 5 m / 10 deg, then the best '
                         'fair score (round 3, default); pnp: round 1 rule, most inliers')
    ap.add_argument('--scratch', default='/tmp/dsh-posefree-vb')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--verbose', type=int, default=0)
    ap.add_argument('--out', default='results/moe_parallel.npz')
    a = ap.parse_args()
    _OPT.update(visual=a.visual, gate2=a.gate2, k=a.k, rot=True, solver=a.solver,
                coarse_points=a.coarse_points, fine_points=a.fine_points,
                threads=a.threads, topk=a.topk, accept=a.accept, scratch=a.scratch,
                stride=a.stride, verbose=a.verbose)
    # No init() in the parent before the pool: the visual expert pulls in torch, and
    # forking a process that has already started its thread pools parks the children at
    # 0 % CPU (measured, 21 minutes with nothing done).  The workers build their own.
    n = len(json.load(open(os.path.join(ROOT, 'inputs/frames.json'))))
    idx = list(range(0, n if a.limit <= 0 else min(a.limit, n), a.stride))
    if a.workers > 1:
        with Pool(a.workers, initializer=init) as p:
            res = p.map(_w, idx, chunksize=1)
    else:
        init()
        res = [_w(i) for i in idx]
    keys = [str(k) for k in np.load(os.path.join(ROOT, 'inputs/initial.npz'),
                                    allow_pickle=True)['keys']][:len(idx) * a.stride]
    keys = [keys[i] for i in idx]
    poses = np.stack([r[1] for r in res])
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, keys=np.array(keys), poses=poses)
    diag_path = os.path.splitext(out)[0] + '_diag.jsonl'
    with open(diag_path, 'w') as fh:
        for _, _, dg in res:
            fh.write(json.dumps(dg) + '\n')
    asked = sum(1 for r in res if r[2].get('asked'))
    got = sum(1 for r in res if 'arb_margin' in r[2])
    adopted = sum(1 for r in res if r[2].get('visual'))
    print('visual=%s topk=%d accept=%s frames=%d asked=%d hypothesised=%d adopted=%d '
          'k=%.2f -> %s' % (a.visual, a.topk, a.accept, len(idx), asked, got, adopted,
                            a.k, out))


if __name__ == '__main__':
    main()
