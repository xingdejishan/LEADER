"""Multi-window geometry expert: candidate road centres in, arbitrated pose out.

What changes, and what does not
--------------------------------
Only the *origin* of the search changes.  For each candidate centre c the expert builds
the same local map the deployed code builds (60 m crop, 0.3 m voxel, the 3 nearest
other-date references), plants a pseudo initial pose there (translation c, rotation from
the frame's own T0) and runs the *existing* along-track / lateral search.  The cost
function, the query points, the truncation, the solver, the grid and every constant are
the ones already in the repo.

The candidate poses are then arbitrated with the *existing* test: one shared evidence
cloud (the union of the crops each candidate would have used on its own), the same 4000
query points, sigma_C calibrated on the incumbent from the same 12 perturbations, and the
same switch rule

    switch  iff  C_incumbent - C_best > k * sigma_C          k = 8.4

With ``multi_window=False`` the function calls ``pipeline.localize`` once from T0 and
returns its pose untouched: that is the deployed single-window expert, bit for bit.
Nothing here reads truth and nothing is tuned per frame.
"""
import time

import numpy as np

import arbitration as ARB
import pipeline as P

K_DEFAULT = 8.4          # three_route_unified.DEFAULT_K, unchanged
COST_TRUNC = 1.0         # arbitration.COST_TRUNC, unchanged
ARB_POINTS = 4000        # arbitration.ARB_POINTS, unchanged


def window_pose(T0, centre):
    """Pseudo initial pose: translation at the window centre, rotation from T0."""
    pose = np.asarray(T0, dtype=np.float64).copy()
    pose[:3, 3] = np.asarray(centre, dtype=np.float64).reshape(3)
    return pose


def _crop(maps, centre, cfg):
    """The deployed local map at `centre` (radius 60 m, 0.3 m voxel, 3 references)."""
    triple = maps.nearby(centre)[:cfg.reference_count]
    if len(triple) == 0:
        return None, triple
    points = maps.local(centre, triple)
    if len(points) == 0:
        return None, triple
    return points, triple


def run_frame(record, T0, maps, cfg, windows=(), multi_window=True, k=K_DEFAULT):
    """One frame.  Returns (pose, info).

    record    {'key', 'scan_path'}
    T0        4x4 initial pose
    windows   output of mw_windows.candidate_windows (ignored when multi_window=False)
    """
    T0 = np.asarray(T0, dtype=np.float64)
    info = dict(multi_window=bool(multi_window), k=float(k),
                n_windows=0 if not multi_window else int(len(windows)),
                windows=[], skipped=[])

    started = time.time()
    scan = P.prepare_scan(record, cfg)
    info['t_scan_s'] = time.time() - started

    # --- the incumbent: the deployed expert, one window centred on T0
    started = time.time()
    T_inc, d_inc = P.localize(record, T0, maps, cfg, scan=scan)
    info['t_incumbent_s'] = time.time() - started
    info['incumbent'] = dict(pose=np.asarray(T_inc, float).tolist(),
                             s=float(d_inc['s']), d=float(d_inc['d']),
                             cost_own=float(d_inc['cost']),
                             moved=float(np.linalg.norm(T_inc[:3, 3] - T0[:3, 3])),
                             map_points=int(d_inc['map_points']))
    if not multi_window:
        info['chosen'] = 'geo_t0'
        info['switched'] = False
        return T_inc, info

    hyps = [ARB.Hypothesis(T_inc, 'geo_t0',
                           {'incumbent': True, 'cost_own': float(d_inc['cost'])})]
    crops = []
    t_map = 0.0
    t_search = 0.0
    for j, w in enumerate(windows):
        centre = np.asarray(w['centre'], dtype=np.float64)
        name = 'win%02d' % j
        started = time.time()
        points, triple = _crop(maps, centre, cfg)
        t_map += time.time() - started
        if points is None:
            info['skipped'].append(dict(name=name, ref=int(w['ref']),
                                        reason='no_reference_near_centre'))
            continue
        crops.append(points)
        started = time.time()
        pose_c, d_c = P.localize(record, window_pose(T0, centre), maps, cfg, scan=scan)
        t_search += time.time() - started
        hyps.append(ARB.Hypothesis(pose_c, name,
                                   {'cost_own': float(d_c['cost']), 's': float(d_c['s']),
                                    'd': float(d_c['d'])}))
        info['windows'].append(dict(name=name, ref=int(w['ref']),
                                    channel=w['channel'], also_cls=bool(w['also_cls']),
                                    rank=int(w['rank']),
                                    centre=centre.tolist(),
                                    distance_to_t0=float(w['distance_to_t0']),
                                    pose=np.asarray(pose_c, float).tolist(),
                                    s=float(d_c['s']), d=float(d_c['d']),
                                    cost_own=float(d_c['cost']),
                                    map_points=int(d_c['map_points']),
                                    triple=[int(x) for x in triple],
                                    moved=float(np.linalg.norm(pose_c[:3, 3] - centre))))
    info['t_map_s'] = t_map
    info['t_search_s'] = t_search

    if len(hyps) == 1:
        # every candidate window was unusable: the incumbent stands, nothing to decide
        info.update(chosen='geo_t0', switched=False, decided=False, n_used=0)
        return T_inc, info

    # --- shared evidence: the union of the crops each candidate would have used
    started = time.time()
    mp0, _ = _crop(maps, T0[:3, 3], cfg)
    parts = list(crops)
    if mp0 is not None:
        parts.insert(0, mp0)
    evidence = ARB.Evidence(ARB.union_maps(*parts), scan,
                            trunc=COST_TRUNC, n_points=ARB_POINTS)
    info['t_evidence_s'] = time.time() - started

    started = time.time()
    chosen, arb = ARB.decide(hyps, evidence, k, scan=scan)
    info['t_decide_s'] = time.time() - started

    info['n_used'] = len(info['windows'])
    info['n_map'] = int(evidence.n_map)
    info['n_query'] = int(evidence.n_query)
    info['sigma_c'] = float(arb['sigma_c'])
    info['thresh'] = float(arb['thresh'])
    info['margin'] = float(arb['margin'])
    info['switched'] = bool(arb['switched'])
    info['chosen'] = str(chosen.source)
    info['best'] = str(arb['best'])
    for row in info['windows']:
        row['cost_shared'] = float(arb['costs'][row['name']])
    info['incumbent']['cost_shared'] = float(arb['costs']['geo_t0'])
    return np.asarray(chosen.pose, float), info
