"""Adaptive fine-window expansion (from the coarse-fine-search 2026-10-21 patch).

Replaces the gradient-based pruning, whose lower bound was not valid: the cost is
not linear over a 0.64 m step, and points beyond the truncation are counted as a
constant although they can move inside it. The rule here is a plain heuristic,
not a claimed bound:

  1. take the candidate centres the coarse stage already picked
  2. re-score those centres at the full fine point budget
  3. expand a candidate's 5x9 window only while its centre cost is within
     `thresh` of the best centre, always expanding at least one

Everything evaluated still uses the exact truncated KD-tree cost; nothing is
approximated. `coarse` lets the caller pass the surface the compute gate already
built, so the grid is not paid for twice.
"""
import numpy as np

import fast_lib as F

THRESH = 0.10


def solve_adaptive(map_points, scan, initial, direction, thresh=None, n_coarse=1000,
                   n_fine=1500, workers=4, topk=6, coarse=None):
    """Same contract as fast_lib.solve, plus the number of fine windows expanded.

    Returns (s, d, cost, n_evals, expanded).
    """
    thresh = THRESH if thresh is None else thresh
    if coarse is None:
        coarse = F.coarse_stage(map_points, scan, initial, direction, workers,
                                n_coarse, topk)
    tree = coarse['tree']
    rot = coarse['rot']
    origin = coarse['origin']
    lateral = coarse['lateral']
    surface = coarse['surface']
    d_values = coarse['d_values']
    picks = coarse['picks']

    q_fine = F.spread_subset(scan, n_fine)
    base_f = q_fine @ rot.T

    # --- candidate centres at the full fine budget
    centre = np.empty(len(picks))
    for a, (i, j) in enumerate(picks):
        s0, d0 = float(F.S_COARSE[i]), float(d_values[j])
        centre[a] = float(np.minimum(
            F.query_dist(tree, base_f + (origin + s0 * direction + d0 * lateral), workers),
            F.TRUNC).mean())
    order = np.argsort(centre)
    base = centre[order[0]]

    # --- adaptive expansion
    best = (np.inf, 0.0, 0.0)
    expanded = 0
    evals = coarse['n_evals'] + len(picks)
    for a in order:
        if expanded > 0 and centre[a] - base > thresh:
            continue
        i, j = picks[a]
        s0, d0 = float(F.S_COARSE[i]), float(d_values[j])
        poses = [(s0 + ds, d0 + dd) for ds in F.S_FINE for dd in F.D_FINE
                 if -8.0 <= s0 + ds <= 8.0 and -2.0 <= d0 + dd <= 2.0]
        if not poses:
            continue
        probe = np.empty((len(poses) * len(q_fine), 3))
        for k, (s, d) in enumerate(poses):
            probe[k * len(q_fine):(k + 1) * len(q_fine)] = base_f + (
                origin + s * direction + d * lateral)
        costs = np.minimum(F.query_dist(tree, probe, workers), F.TRUNC).reshape(
            len(poses), -1).mean(1)
        evals += len(poses)
        expanded += 1
        t = int(np.argmin(costs))
        if costs[t] < best[0]:
            best = (float(costs[t]), poses[t][0], poses[t][1])
    return best[1], best[2], best[0], evals, expanded
