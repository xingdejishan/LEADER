"""One objective, hypothesis generators, one calibrated switching test.

The pose problem is the minimisation of a single robust cost: the mean truncated
nearest-neighbour distance between the transformed scan and the local reference
cloud (truncation 1 m). Everything below reads that one number.

    T_leader --> coarse cost surface (17 x 3, 1500 points)          stage A
                   |
                   +- descent depth  g = C(T_leader) - min C       -> compute gate
                   +- ambiguity ratio rho = (C_far - C_0) / C_0    -> proposal gate
                   +- fine grid (5 x 9 per candidate, adaptive)    -> T3
    visual branch ---------------------> Tvis (Hypothesis, source='visual')

The two branch outputs are now Hypothesis objects handed to arbitration.decide(),
which scores both on one shared Evidence (the map and KD-tree the search already
built) and switches only when

    C_route3 - C_visual > k * sigma_C

with sigma_C calibrated per frame from 12 tiny perturbations of the incumbent pose.
Parallel experts are the default: rho is diagnostic only. Cascade mode is retained
for explicit ablations; the final choice always uses the calibrated test.

The coarse surface is built once per frame and both the compute gate and the fine
stage read it; the fine stage is never charged for it a second time.

Modes (so the ablation is a one-line switch):
    --mode route3    no visual hypothesis: gate -> search
    --mode frozen    previous arbitration: rho < 0.21 and an external score margin > 0.10
    --mode unified   hypothesis generator + shared evidence + calibrated switch test

The ambiguity ratio has two sources, also one flag:
    --gate merged    measured on the map and KD-tree the search already built (default)
    --gate frozen    the standalone gate, which rebuilds its own map every frame

    python three_route_unified.py --mode unified --gate merged --workers 7
"""
import argparse
import cProfile
import json
import os
import sys
import time
from multiprocessing import Pool, current_process

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'stages/fast_search'))
sys.path.insert(0, os.path.join(HERE, 'stages/gate1'))
sys.path.insert(0, os.path.join(HERE, 'stages'))
import fast_lib as F  # noqa: E402
import pipeline as P  # noqa: E402
import ambiguity as A  # noqa: E402
import arbitration as ARB  # noqa: E402

ROOT = HERE
VERIFY = ('/data2/zhangcy/v11-arbiter-fast-20261018-v1/full1502/REF/'
          'verify-frozen_nopruneoptREF/%06d.json')
COST_TRUNC = 1.0        # the one cost function, in metres
ARB_POINTS = 4000       # query points used when scoring a hypothesis
AMB_THRESHOLD = A.THRESHOLD  # 0.21: below this the other modality is queried
FROZEN_MARGIN = 0.10    # previous arbitration: margin on the external geometry score
VIS_DATA = '/data2/zhangcy/leader-v2-full-trajectory-20261001'  # cam5 images, as online
VIS_ARM = 'frozen_noprune'   # the LightGlue arm the frozen table was produced with
VIS_BUDGET = 8               # online driver --visual-budget default: poses laser-scored
DEFAULT_K = 8.4         # switching threshold = k * sigma_C. Measured on the 84
                        # frames where the proposal gate fires: median sigma_C =
                        # 0.00597 m, so 8.4 * sigma_C = 0.050 m, the old delta.
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
              maps=F.MapSource(refs, np.asarray([r['world_from_body'] for r in refs])[:, :3, 3],
                               radius=cfg.crop_radius, max_refs=12, max_maps=4))
    if _OPT['mode'] != 'route3' and _OPT['gate'] == 'frozen':
        import gate as G1
        _G['gate'] = G1
        _G['lib'] = G1.ReferenceLibrary()

    if _OPT['mode'] == 'route3':
        return
    if _OPT.get('visual_source', 'frozen') in ('live', 'none'):
        return
    vis = []
    if _OPT['mode'] == 'frozen':
        for i in range(len(frames)):
            rec = json.load(open(VERIFY % i))['poses']
            base = ((rec.get('baseline_V7') or {}).get('fair') or {}).get('geometry_score')
            best = None
            for name, row in rec.items():
                if not name.startswith('candidate_'):
                    continue
                sc = (row.get('fair') or {}).get('geometry_score')
                if sc is None:
                    continue
                if best is None or sc > best[0]:
                    best = (float(sc), np.asarray(row['pose'], float))
            vis.append((best, base))
        _G['vis'] = vis
    else:
        _G['vis'] = [loading_hook(i) for i in range(len(frames))]


def loading_hook(i):
    """Pose-only view of the frozen visual table (the unified rule needs no score)."""
    rec = json.load(open(VERIFY % i))['poses']
    best = None
    for name, row in rec.items():
        if not name.startswith('candidate_'):
            continue
        sc = (row.get('fair') or {}).get('geometry_score')
        if sc is None:
            continue
        if best is None or sc > best[0]:
            best = (float(sc), np.asarray(row['pose'], float))
    return (best, None)


EXPERT = {}


def build_expert():
    """The online driver's visual expert, built inside the worker that uses it.

    CUDA is never initialised in the parent: forking a process that already owns a
    CUDA context or a thread pool parks the children (measured in an earlier round).
    """
    if EXPERT.get('branch') is not None:
        return EXPERT['branch']
    import torch
    from visual_ordered import OrderedBranch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(_OPT.get('gpu_fraction', 0.15)))
    identity = current_process()._identity
    wid = int(identity[0]) if identity else 0
    # OrderedBranch is VisualBranch with self._features / self._projections as
    # OrderedDict.  visual_branch.py evicts with dict.popitem(last=False), which a plain
    # dict does not implement: past 64 cached entries every call dies with
    #     TypeError: popitem() takes no keyword arguments
    # (measured this round, frame 16 of a solo bench).  Container type only: no rule,
    # no constant, no solver, and the cached values are identical.
    branch = OrderedBranch(device='cuda' if torch.cuda.is_available() else 'cpu',
                             arm=_OPT.get('visual_arm', VIS_ARM),
                             scratch=os.path.join(_OPT.get('scratch', os.path.join(HERE, '.runtime/offline-visual')),
                                                  'w%d' % wid))
    EXPERT['branch'] = branch
    return branch


def live_hypothesis(i):
    """One frame in, one visual hypothesis out: exactly the online cascade's call.

    Same function, same arguments as the gated path in the online driver:
    budget 8 candidates and nothing else -- the reference list is SALAD global
    cosine top-5, which is pose-free, so the front-end pose T0 is not passed and
    could not be used.  Every constant the expert uses is read from the frozen plan
    inside visual_branch.py.
    """
    rec = _G['frames'][i]
    key = str(rec['key'])
    date, stamp = key.split('/')
    image = os.path.join(VIS_DATA, 'NCLT', date, 'cam5', stamp + '.jpg')
    info = {'visual_seconds': 0.0}
    started = time.monotonic()
    try:
        hypothesis = build_expert().hypothesize(
            key, image, budget=int(_OPT.get('visual_budget', VIS_BUDGET)))
    except BaseException as exc:                 # a visual failure never voids the frame
        info['visual_seconds'] = round(time.monotonic() - started, 4)
        info['visual_error'] = '%s: %s' % (type(exc).__name__, exc)
        return None, None, info
    info['visual_seconds'] = round(time.monotonic() - started, 4)
    if hypothesis is None:
        info['visual_note'] = 'no_hypothesis'
        return None, None, info
    detail = hypothesis.get('detail') or {}
    info.update(visual_score=float(hypothesis['score']),
                visual_verified=bool(hypothesis.get('verified')),
                visual_candidates=int(hypothesis['n_candidates']),
                visual_channel=hypothesis.get('channel'),
                visual_refs=[int(x) for x in (hypothesis.get('reference_ids') or [])],
                visual_retrieval=detail.get('retrieval'),
                visual_pnp_calls=detail.get('pnp_calls'),
                visual_capped=detail.get('capped'),
                visual_batched=bool(detail.get('batched')),
                visual_times=detail.get('seconds'),
                visual_scored=[dict(channel=row['channel'],
                                    score=float(row['score']),
                                    verified=bool(row['verified']),
                                    reference_ids=[int(x) for x in row['reference_ids']],
                                    inliers=int(row['inliers']),
                                    consensus=int(row['consensus']),
                                    residual_median_m=row['residual_median_m'],
                                    residual_p90_m=row['residual_p90_m'],
                                    supported_bins=int(row['supported_bins']),
                                    top_bin_share=row['top_bin_share'])
                               for row in (detail.get('scores') or [])])
    return (float(hypothesis['score']), np.asarray(hypothesis['pose'], float)), None, info


def cost(tree, pts, trunc=COST_TRUNC):
    d = tree.query(pts, k=1, workers=1, distance_upper_bound=trunc)[0]
    return float(np.where(np.isfinite(d), d, trunc).mean())


def cost_at(pose, q, maps, cfg):
    """Score a hypothesis on the map cropped around its own position.

    Still exported because the online entry point imports it; the offline
    arbitration below no longer calls it, since both hypotheses are now scored on
    one shared Evidence instead of two separately cropped maps.
    """
    centre = pose[:3, 3]
    mp = maps.local(centre, maps.nearby(centre)[:cfg.reference_count])
    return cost(cKDTree(mp), q @ pose[:3, :3].T + centre)




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

    # --- stage A: the coarse surface, built once
    cs = F.coarse_stage(mp, scan, T0, u, workers=1,
                        n_coarse=cfg.coarse_points, topk=cfg.topk)
    qc = scan[:min(cfg.coarse_points, len(scan))]
    c_zero = cost(cs['tree'], cs['base_c'] + T0[:3, 3])
    gain = c_zero - float(cs['surface'].min())

    # --- gates: descend only if the surface actually drops
    T3 = np.array(T0, float)
    ran = False
    if gain > _OPT['gate2']:
        pose, _ = P.localize(rec, T0, maps, cfg, coarse=cs, scan=scan)
        T3 = np.asarray(pose, float)
        ran = True

    rotated = False

    diag = dict(key=rec['key'], gain=float(gain), ran_fine=bool(ran), rotated=rotated)

    if _OPT['mode'] == 'route3':
        diag.update(visual=False, chosen='route3')
        return i, T3, diag

    # --- ambiguity: is there a competing along-track position that fits as well?
    #     merged: same map and tree as the search, strided 4000-point sweep
    #     frozen: the standalone gate, with its own map built per frame
    if _OPT['gate'] == 'merged':
        _, amb = A.decide(cs['tree'], scan, T0, u)
    else:
        amb = _G['gate'].ambiguity_ratio(_G['lib'], rec['scan_path'], T0)
    ratio = amb['ratio']
    diag['ratio'] = float(ratio)
    diag['gate'] = _OPT['gate']

    # ---- control flow: the only thing this refactor changes ------------------
    # cascade : the proposal gate decides whether the visual expert is asked at all
    # parallel: both experts answer on every frame; rho is written to the report
    #           and never vetoes anything
    gated = bool(np.isfinite(ratio) and ratio < AMB_THRESHOLD)
    if _OPT['visual_source'] == 'none':
        asked = False                      # the ablation: no visual hypothesis at all
    elif _OPT['experts'] == 'parallel':
        asked = True                       # both experts answer on every frame
    else:
        asked = bool(gated)                # the cascade: the gate decides
    diag['gated'] = gated
    diag['asked'] = asked
    diag['experts'] = _OPT['experts']
    diag['visual_source'] = _OPT['visual_source']
    best, base, vinfo = None, None, {}
    if asked:
        if _OPT['visual_source'] == 'frozen':
            best, base = _G['vis'][i]
        else:
            best, base, vinfo = live_hypothesis(i)
    diag.update(vinfo)
    if best is not None and not len(maps.nearby(best[1][:3, 3])):
        # parallel-only degenerate case: the expert's pose can land where the reference
        # library has no cloud at all.  The frozen evidence assembly takes the union of
        # the two hypotheses' crops and has nothing to concatenate on that side, and the
        # hypothesis itself has no map to be scored against, so it is dropped before the
        # arbitration.  The cascade never met this: the gate never sent it such a frame.
        diag['visual_dropped'] = 'no_reference_near_visual'
        diag['visual_dropped_m'] = float(np.linalg.norm(best[1][:3, 3] - T0[:3, 3]))
        best = None
    accepted = False
    c3 = cv = float('nan')
    if best is not None:
        # rho has done its job: the frame is ambiguous enough to ask the other
        # modality. What happens next is a hypothesis test, not a threshold.
        if _OPT['mode'] == 'frozen':
            accepted = bool(base is not None and best[0] > float(base) + FROZEN_MARGIN)
        else:
            # evidence: the crops both hypotheses would have used on their own,
            # merged into one cloud, so neither side is scored blind
            cv_centre = best[1][:3, 3]
            mp_v = maps.local(cv_centre, maps.nearby(cv_centre)[:cfg.reference_count])
            evidence = ARB.Evidence(ARB.union_maps(mp, mp_v), scan,
                                    trunc=COST_TRUNC, n_points=ARB_POINTS)
            hyps = [ARB.Hypothesis(T3, 'route3',
                                   {'incumbent': True, 'ratio': float(ratio)}),
                    ARB.Hypothesis(best[1], 'visual',
                                   {'external_score': float(best[0])})]
            chosen, arb = ARB.decide(hyps, evidence, _OPT['k'], scan=scan)
            accepted = chosen.source == 'visual'
            c3 = arb['costs']['route3']
            cv = arb['costs']['visual']
            diag.update(arb_sigma=arb['sigma_c'], arb_sigma_std=arb['sigma_c_std'],
                        arb_sigma_n=arb['sigma_c_n'], arb_thresh=arb['thresh'],
                        arb_margin=arb['margin'], arb_k=arb['k'],
                        arb_best=arb['best'], arb_switched=arb['switched'],
                        arb_decided=arb['decided'], arb_n_map=evidence.n_map,
                        arb_dist_v=float(np.linalg.norm(cv_centre - T0[:3, 3])))
    diag.update(c_route3=float(c3), c_visual=float(cv),
                dcost=float(cv - c3) if np.isfinite(c3) and np.isfinite(cv) else float('nan'),
                visual=bool(accepted), chosen='visual' if accepted else 'route3')
    return i, (best[1] if accepted else T3), diag


def _w(i):
    return one(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('route3', 'frozen', 'unified'), default='unified')
    ap.add_argument('--gate', choices=('merged', 'frozen'), default='merged',
                    help='merged: ratio from the search map; frozen: standalone gate')
    ap.add_argument('--solver', choices=('adaptive', 'prune', 'full'), default='adaptive')
    ap.add_argument('--coarse-points', type=int, default=1000)
    ap.add_argument('--fine-points', type=int, default=1500)
    ap.add_argument('--threads', type=int, default=1,
                    help='KD-tree query threads inside one frame')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--gate2', type=float, default=-1.0,
                    help='descent threshold; -1 disables the gate (default, see README)')
    ap.add_argument('--k', type=float, default=DEFAULT_K,
                    help='switching threshold in units of sigma_C (metres of cost)')
    ap.add_argument('--experts', choices=('cascade', 'parallel'), default='parallel',
                    help='cascade: ask the visual expert only where the proposal gate '
                         'fires (the current structure); parallel: both experts answer '
                         'on every frame and rho becomes a report field')
    ap.add_argument('--visual-source', choices=('frozen', 'live', 'none'),
                    default='live',
                    dest='visual_source',
                    help='frozen: the frozen visual table; live: the online driver\'s '
                         'VisualBranch, same call and same constants, on the GPU; '
                         'none: geometry only (the ablation)')
    ap.add_argument('--visual-arm', default=VIS_ARM)
    ap.add_argument('--visual-budget', type=int, default=VIS_BUDGET)
    ap.add_argument('--scratch', default=os.path.join(HERE, '.runtime/offline-visual'))
    ap.add_argument('--gpu-fraction', type=float, default=0.15)
    ap.add_argument('--gpu', default='3')
    ap.add_argument('--shard', default='0/1', help='i/n stripes, one per GPU')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0, help='0 = all frames')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.visual_source == 'live' and a.workers != 1:
        ap.error('live visual mode requires --workers 1; use separate GPU shards instead')
    if not 0 < a.gpu_fraction <= 0.15 or not 1 <= a.threads <= 4 or not a.gpu.isdigit():
        ap.error('invalid GPU, GPU fraction or thread safety limit')
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    _OPT.update(mode=a.mode, gate=a.gate, gate2=a.gate2, k=a.k,
                solver=a.solver, coarse_points=a.coarse_points,
                fine_points=a.fine_points, threads=a.threads,
                experts=a.experts, visual_source=a.visual_source,
                visual_arm=a.visual_arm, visual_budget=a.visual_budget,
                scratch=a.scratch, gpu_fraction=a.gpu_fraction)
    if a.mode == 'route3':
        default_out = 'results/three_route_route3.npz'
    elif a.gate == 'merged':
        default_out = 'results/three_route_merged.npz'
    else:
        default_out = 'results/three_route_%s.npz' % a.mode
    out = a.out or default_out

    init()
    n = len(_G['frames'])
    idx = list(range(n if a.limit <= 0 else min(a.limit, n)))
    shard_index, shard_count = (int(x) for x in a.shard.split('/'))
    idx = idx[shard_index::shard_count][::a.stride]
    if a.workers > 1:
        with Pool(a.workers, initializer=init) as p:
            res = p.map(_w, idx, chunksize=4)
    else:
        res = [_w(i) for i in idx]

    all_keys = [str(k) for k in np.load(os.path.join(ROOT, 'inputs/initial.npz'),
                                       allow_pickle=True)['keys']]
    keys = [all_keys[i] for i in idx]
    poses = np.stack([r[1] for r in res])
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    np.savez(os.path.join(ROOT, out), keys=np.array(keys), poses=poses)
    diag_path = os.path.splitext(out)[0] + '_diag.jsonl'
    diag_path = diag_path if os.path.isabs(diag_path) else os.path.join(ROOT, diag_path)
    with open(diag_path, 'w') as fh:
        for _, _, dg in res:
            fh.write(json.dumps(dg) + '\n')
    vis = sum(1 for r in res if r[2].get('visual'))
    fine = sum(1 for r in res if r[2]['ran_fine'])
    print('mode=%s gate=%s frames=%d visual=%d fine=%d gate2=%.3f k=%.4f -> %s'
          % (a.mode, a.gate, len(idx), vis, fine, a.gate2, a.k, out))


if __name__ == '__main__':
    main()

