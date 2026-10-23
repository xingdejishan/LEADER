#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sequence-level (Viterbi) along-track correction.

The per-frame branch chooses every frame's translation correction on its own: the
argmin of a truncated scan-to-map cost over the project's 17 (along-track, 1 m) x 3
(lateral, 0.5 m) coarse grid, then a local refinement around the winning cell.  Two
measured properties make that choice weak:

  * on ordinary frames the front-end pose scores better than the truth on this cost
    (74% of the time), so the argmin carries a systematic bias;
  * 84% of the frames the branch damages have equally bad neighbours, so nothing in
    one frame's surface separates a good correction from a bad one.

This module replaces the independent argmin with one path through each date's whole
sequence (Viterbi).  The emission is the same cost surface the per-frame branch
already computes; the transition is a penalty on how far the correction moves
between consecutive frames, plus a prior against correcting at all.

    keys, t0, _ = load_pose_table()
    order, bounds = sequence_order(keys)
    info   = build_profiles('/home/zhangcy/.dsh_scratch/dp_profiles', order=order, processes=40)
    merged = merge_blocks('/home/zhangcy/.dsh_scratch/dp_profiles',
                          'results/dp-profiles.npz', order=order, keys=keys, t0=t0)
    prof   = attach_motion(load_profiles(merged), keys, t0, order)
    chosen = viterbi(prof, lam=0.01, eta=0.01)         # one state per frame
    poses  = refine_selected(keys, t0, chosen, order)  # sub-metre offset, or the front end

The measured outcome on all 104,728 frames is in results/along-track-dp.md.

State space
-----------
18 states: the 17 along-track values of the project's own coarse grid, plus one
"keep the front-end pose" state.  The lateral value is *not* a state: for a chosen
along-track value it is the argmin over that frame's three lateral cells.  The lateral
grid is anchored per frame (d_home + [-0.5, 0, 0.5], clipped to +-2 m), so those three
numbers differ between frames and a 51-state product would compare unlike offsets;
the 17 along-track values are the same 17 metres on every frame.

Cost convention
---------------
Everything is in the project's own unit: metres of mean truncated nearest-neighbour
distance (truncation 1 m).  Costs run ~0.2-0.8 m and the spread between the best and
worst grid cell is 1e-3 to 1e-1 m, so `lam` (transition penalty per metre of change)
and `eta` (penalty for any non-zero correction) are in the same metres and only small
values can matter.  See results/along-track-dp.md.

No trainable parameters and no new data: map, scan, coarse grid and refinement are the
project's own functions (`fast_lib`, `pipeline`, `adaptive`).
"""
import json
import os
import re
import sys
import time

import numpy as np

PROJECT = os.path.dirname(os.path.abspath(__file__))
FAST_SEARCH = os.path.join(PROJECT, 'stages', 'fast_search')
STAGES = os.path.join(PROJECT, 'stages')
for _p in (FAST_SEARCH, STAGES, PROJECT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fast_lib as F
import pipeline as PL
import adaptive

RESULTS = os.path.join(PROJECT, 'results')
REFS_JSON = os.path.join(PROJECT, 'inputs', 'refs.json')
TRUTH_DIR = '/data2/zhangcy/leader-v2-full-trajectory-20261001/evaluator-only-truth'
POSE_TABLE = os.path.join(RESULTS, 'full-online.npz')
SCAN_FMT = '/data2/dataset/NCLT/%s/velodyne_sync/%s.bin'

# the project's own coarse grid; never redefined here
S_AXIS = np.asarray(F.S_COARSE, np.float64)     # 17 along-track values, metres
N_STATE = len(S_AXIS) + 1                       # + the "no correction" state
STATE_ZERO = 0
BIG = 1.0e6

BLOCK_KEYS = ('surface', 'origin', 'd_values', 'argmin_a', 'argmin_b', 'ok',
              'map_points', 'scan_points', 'seconds')


# --------------------------------------------------------------------------- #
# frame table and sequence order                                               #
# --------------------------------------------------------------------------- #
def load_pose_table(path=None):
    """keys, t0 (front-end poses), poses (the project's own final table)."""
    path = POSE_TABLE if path is None else path
    with np.load(path, allow_pickle=True) as z:
        keys = [str(k) for k in z['keys']]
        t0 = np.asarray(z['t0'], np.float64)
        poses = np.asarray(z['poses'], np.float64)
    return keys, t0, poses


def sequence_order(keys):
    """Global index order sorted by (date, timestamp) and the per-date slices.

    The keys are '<date>/<16-digit timestamp>', so sorting on the integer timestamp
    is acquisition order.  Every sequence operation must work in this order, never in
    the table's own order.
    """
    order = np.array(sorted(range(len(keys)),
                            key=lambda i: (keys[i].split('/')[0],
                                           int(keys[i].split('/')[1]))), np.int64)
    return order, date_bounds(keys, order)


def date_bounds(keys, order):
    """[(date, start, stop)] over the positions of `order`."""
    dates = [keys[int(i)].split('/')[0] for i in order]
    bounds, start = [], 0
    for i in range(1, len(dates) + 1):
        if i == len(dates) or dates[i] != dates[start]:
            bounds.append((dates[start], start, i))
            start = i
    return bounds


def motion_proxies(keys, order, t0):
    """Front-end-motion proxies between consecutive frames of the same date.

    Returns (rel_trans (n,3), rel_yaw (n,), gap (n,)) aligned with `order`:
    rel_trans[r] is the translation of inv(T0[r-1]) @ T0[r] and rel_yaw[r] the angle
    about z, both from the front-end poses themselves.  A frame that starts a date
    gets a zero delta and gap=1, so no sequence is ever linked across a discontinuity.
    """
    n = len(order)
    rel_trans = np.zeros((n, 3), np.float64)
    rel_yaw = np.zeros(n, np.float64)
    gap = np.ones(n, np.int8)
    dates = [keys[i].split('/')[0] for i in order]
    for r in range(1, n):
        if dates[r] != dates[r - 1]:
            continue
        rel = np.linalg.inv(t0[order[r - 1]]) @ t0[order[r]]
        rel_trans[r] = rel[:3, 3]
        rel_yaw[r] = np.degrees(np.arctan2(rel[1, 0], rel[0, 0]))
        gap[r] = 0
    return rel_trans, rel_yaw, gap


# --------------------------------------------------------------------------- #
# pass 1: the 17 x 3 cost surface per frame                                    #
# --------------------------------------------------------------------------- #
_G = {}


def _prepare(scan_path):
    """scan decode -> 1-60 m annulus -> 0.3 m voxel (pipeline.prepare_scan)."""
    points = F.decode(scan_path)
    radius = np.linalg.norm(points, axis=1)
    return F.voxel(points[(radius > 1.0) & (radius < 60.0)], 0.3)


def worker_init(workers=1):
    """One MapSource per worker: the project's own, radius 60, 3 references."""
    keys, t0, _ = load_pose_table()
    refs = json.load(open(REFS_JSON))
    for row in refs:
        row['scan_path'] = os.path.realpath(row['scan_path'])
    positions = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]
    _G['keys'] = keys
    _G['t0'] = t0
    _G['maps'] = F.MapSource(refs, positions, radius=60.0)
    _G['workers'] = workers
    return _G


def frame_profile(i, workers=None):
    """One frame's full cost profile: 17x3 surface, origin cost, argmin cell.

    Everything comes from the project's own coarse stage, so this surface is the same
    one the per-frame branch scores.
    """
    i = int(i)
    key = _G['keys'][i]
    date, stamp = key.split('/')
    row = dict(index=i, ok=False, reason='', surface=None, origin=np.nan,
               d_values=None, argmin_a=-1, argmin_b=-1, map_points=0, scan_points=0)
    workers = _G.get('workers', 1) if workers is None else workers
    centre = _G['t0'][i][:3, 3]
    triple = _G['maps'].nearby(centre)[:3]
    if len(triple) < 2:
        row['reason'] = 'no_reference'
        return row
    map_points = _G['maps'].local(centre, triple)
    row['map_points'] = int(len(map_points))
    if len(map_points) < 500:
        row['reason'] = 'thin_map'
        return row
    scan = _prepare(SCAN_FMT % (date, stamp))
    row['scan_points'] = int(len(scan))
    if len(scan) < 100:
        row['reason'] = 'thin_scan'
        return row
    T0 = _G['t0'][i]
    u = PL.along_track_direction(T0)
    coarse = F.coarse_stage(map_points, scan, T0, u, workers=workers,
                            n_coarse=1000, topk=6)
    surface = np.asarray(coarse['surface'], np.float64)
    a, b = np.unravel_index(int(np.argmin(surface)), surface.shape)
    row.update(ok=True, surface=surface,
               origin=float(F.query_dist(coarse['tree'],
                                         coarse['base_c'] + centre, workers=workers).mean()),
               d_values=np.asarray(coarse['d_values'], np.float64),
               argmin_a=int(a), argmin_b=int(b))
    return row


def _savez(path, **arrays):
    """np.savez with a retry loop: this box's / runs at 100% and loses ENOSPC races."""
    tmp = path[:-4] + '.partial.npz'
    last = None
    for attempt in range(30):
        try:
            with open(tmp, 'wb') as fh:
                np.savez(fh, **arrays)
            os.replace(tmp, path)
            return path
        except OSError as exc:
            last = exc
            try:
                os.unlink(tmp)
            except OSError:
                pass
            time.sleep(3.0 + 2.0 * attempt)
    raise last


def _block_path(out_dir, start):
    return os.path.join(out_dir, 'prof_%06d.npz' % start)


def _write_block(out_dir, start, rows):
    n = len(rows)
    surface = np.full((n, len(S_AXIS), 3), np.nan, np.float32)
    d_values = np.full((n, 3), np.nan, np.float32)
    origin = np.full(n, np.nan, np.float32)
    argmin_a = np.full(n, -1, np.int8)
    argmin_b = np.full(n, -1, np.int8)
    flag = np.zeros(n, np.int8)
    map_points = np.zeros(n, np.int32)
    scan_points = np.zeros(n, np.int32)
    secs = np.zeros(n, np.float32)
    index = np.zeros(n, np.int64)
    for k, r in enumerate(rows):
        index[k] = r['index']
        secs[k] = r.get('seconds', 0.0)
        map_points[k] = r['map_points']
        scan_points[k] = r['scan_points']
        if not r['ok']:
            continue
        flag[k] = 1
        surface[k] = r['surface']
        d_values[k] = r['d_values']
        origin[k] = r['origin']
        argmin_a[k] = r['argmin_a']
        argmin_b[k] = r['argmin_b']
    path = _block_path(out_dir, start)
    return _savez(path, index=index, surface=surface, origin=origin, d_values=d_values,
                  argmin_a=argmin_a, argmin_b=argmin_b, ok=flag,
                  map_points=map_points, scan_points=scan_points, seconds=secs)


def _segment(task):
    """One worker task: a contiguous slice of the sequence order, block by block."""
    out_dir, order, start, stop, block = task
    worker_init()
    written, n, t0 = [], 0, time.time()
    for s in range(start, stop, block):
        if os.path.exists(_block_path(out_dir, s)):
            continue
        rows = []
        for r in range(s, min(s + block, stop)):
            t = time.time()
            row = frame_profile(order[r])
            row['seconds'] = time.time() - t
            rows.append(row)
        written.append(_write_block(out_dir, s, rows))
        n += len(rows)
    return dict(frames=n, blocks=len(written), seconds=time.time() - t0)


def build_profiles(out_dir, order=None, processes=32, block=500, limit=None,
                   options=None):
    """Pass 1: cost surfaces, one npz block of `block` frames at a time.

    Blocks start at multiples of `block` in the sequence order, so they do not depend
    on how the work was split and an existing block is skipped: the pass is resumable
    and a crash costs at most one block.
    """
    import multiprocessing as mp
    keys, _, _ = load_pose_table()
    if order is None:
        order, _ = sequence_order(keys)
    order = np.asarray(order, np.int64)
    if limit:
        order = order[:int(limit)]
    os.makedirs(out_dir, exist_ok=True)

    step = max(block, ((len(order) // max(processes, 1)) // block + 1) * block)
    spans = [(s, min(s + step, len(order))) for s in range(0, len(order), step)]
    todo = [t for t in spans
            if any(not os.path.exists(_block_path(out_dir, s))
                   for s in range(t[0], t[1], block))]
    out = dict(out_dir=out_dir, frames=int(len(order)), spans=len(spans),
               tasks=len(todo), block=block, processes=processes)
    if not todo:
        out['skipped'] = True
        return out
    tasks = [(out_dir, order, a, b, block) for a, b in todo]
    t_all = time.time()
    with mp.get_context('fork').Pool(processes) as pool:
        for res in pool.imap_unordered(_segment, tasks):
            print('[pass1] %6d frames in %6.1f s -> %d blocks'
                  % (res['frames'], res['seconds'], res['blocks']), flush=True)
    out['seconds'] = time.time() - t_all
    return out


def merge_blocks(out_dir, out_path=None, order=None, keys=None, t0=None):
    """Concatenate every block into one npz, already in sequence order.

    The table's own key order is *not* acquisition order (the three full-run shards
    interleave the dates), so the merged rows are re-arranged into `order` space:
    row r is the profile of frame `order[r]`.  `index` still records that global
    index, so a row can always be traced back to the table.  Passing `keys` and `t0`
    also stores the frame keys and the front-end motion proxies, which is what makes
    the file self-contained for the sequence model.
    """
    names = sorted(f for f in os.listdir(out_dir)
                   if re.match(r'prof_\d{6}\.npz$', f))
    if not names:
        raise IOError('no profile blocks in %s' % out_dir)
    parts = {k: [] for k in BLOCK_KEYS}
    index = []
    for name in names:
        with np.load(os.path.join(out_dir, name)) as z:
            index.append(np.asarray(z['index'], np.int64))
            for k in BLOCK_KEYS:
                parts[k].append(np.asarray(z[k]))
    idx = np.concatenate(index)
    stacked = {k: np.concatenate(parts[k]) for k in BLOCK_KEYS}
    if order is None:
        order = np.arange(len(idx), dtype=np.int64)   # rows already in table order
    order = np.asarray(order, np.int64)
    if len(order) != len(idx):
        raise ValueError('profile rows (%d) do not match the frame order (%d)'
                         % (len(idx), len(order)))
    where = np.full(int(idx.max()) + 1 if len(idx) else 0, -1, np.int64)
    where[idx] = np.arange(len(idx))
    take = where[order]
    missing = int((take < 0).sum())
    if missing:
        raise ValueError('%d frames of the order have no profile row' % missing)
    stacked = {k: v[take] for k, v in stacked.items()}
    stacked['index'] = order
    if keys is not None and t0 is not None:
        rel_trans, rel_yaw, gap = motion_proxies(keys, order, t0)
        stacked.update(keys=np.array([keys[int(i)] for i in order]),
                       rel_trans=rel_trans, rel_yaw=rel_yaw, gap=gap)
    out_path = out_path or os.path.join(RESULTS, 'dp-profiles.npz')
    _savez(out_path, **stacked)
    return out_path, int(len(order))


def load_profiles(path):
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.keys()}


def attach_motion(prof, keys, t0, order):
    """Make sure a profile dict carries the motion proxies and the date split.

    `merge_blocks` already stores them; this only fills them in for a profile dict
    built some other way, so the sequence model always has what it needs.
    """
    prof = dict(prof)
    if not {'rel_yaw', 'gap', 'rel_trans'} <= set(prof):
        rel_trans, rel_yaw, gap = motion_proxies(keys, order, t0)
        prof['rel_trans'], prof['rel_yaw'], prof['gap'] = rel_trans, rel_yaw, gap
    prof['order'] = np.asarray(order, np.int64)
    prof['bounds'] = date_bounds(keys, order)
    return prof


# --------------------------------------------------------------------------- #
# emission                                                                    #
# --------------------------------------------------------------------------- #
def emissions(prof):
    """Per-frame state costs and each state's own (s, d) offset in metres.

    State 0 is "keep the front-end pose"; its cost is the surface's own origin cost,
    not a grid corner.  A frame without a usable map gets BIG everywhere except state 0.
    """
    surface = np.asarray(prof['surface'], np.float64)
    origin = np.asarray(prof['origin'], np.float64)
    d_values = np.asarray(prof['d_values'], np.float64)
    ok = np.asarray(prof['ok'], np.int8).astype(bool)
    n = len(surface)
    finite = np.isfinite(surface)
    best_b = np.argmin(np.where(finite, surface, BIG), axis=2)              # (n,17)
    rows = np.arange(n)[:, None]
    grid_cost = surface[rows, np.arange(len(S_AXIS))[None, :], best_b]
    grid_d = d_values[rows, best_b]
    good = ok & finite.any(axis=2).any(axis=1)
    cost = np.full((n, N_STATE), BIG, np.float64)
    d = np.zeros((n, N_STATE), np.float64)
    cost[good, STATE_ZERO] = origin[good]
    cost[good, 1:] = grid_cost[good]
    d[good, 1:] = grid_d[good]
    return dict(cost=cost, s=np.concatenate([[0.0], S_AXIS]), d=d, good=good)


# --------------------------------------------------------------------------- #
# pass 2: Viterbi                                                             #
# --------------------------------------------------------------------------- #
def transition(s_prev, d_prev, s_now, d_now, lam, lateral_weight=0.5, yaw_deg=0.0,
               yaw_scale=None):
    """lam * (|ds| + w |dd|) for every state pair, in metres.

    `yaw_scale` (degrees) relaxes the penalty where the front end turned sharply
    between the two frames: a correction that flips with a real turn is not a jump.
    """
    delta = (np.abs(s_prev[:, None] - s_now[None, :])
             + lateral_weight * np.abs(d_prev[:, None] - d_now[None, :]))
    lam_eff = lam if not yaw_scale else lam / (1.0 + abs(yaw_deg) / float(yaw_scale))
    return lam_eff * delta


def viterbi(prof, emis=None, lam=0.0, eta=0.0, lateral_weight=0.5, yaw_scale=None):
    """Viterbi over each date's sequence; returns the chosen state per global index.

    Objective: sum_i [cost_i(state_i) + eta * 1{state_i != no-correction}]
             + sum_i lam * (|ds_i| + w |dd_i|)
    `eta` is in metres of cost: the minimum improvement a correction has to deliver
    on its own before any transition penalty.
    """
    emis = emissions(prof) if emis is None else emis
    cost = emis['cost'] + eta * (np.arange(N_STATE) > 0)[None, :]
    s, d = emis['s'], emis['d']
    rel_yaw, gap = np.asarray(prof['rel_yaw']), np.asarray(prof['gap'])
    chosen = np.zeros(len(cost), np.int16)
    for _, a, b in prof['bounds']:
        chosen[a:b] = _viterbi_one(cost[a:b], s, d[a:b], rel_yaw[a:b], gap[a:b],
                                  lam, lateral_weight, yaw_scale)
    return chosen


def _viterbi_one(cost, s, d, rel_yaw, gap, lam, lateral_weight, yaw_scale):
    n = len(cost)
    back = np.zeros((n, N_STATE), np.int16)
    prev = cost[0].copy()
    ar = np.arange(N_STATE)
    for i in range(1, n):
        if gap[i]:
            trans = np.zeros((N_STATE, N_STATE))
        else:
            trans = transition(s, d[i - 1], s, d[i], lam, lateral_weight,
                               rel_yaw[i], yaw_scale)
        cand = prev[:, None] + trans            # [previous state, current state]
        back[i] = np.argmin(cand, axis=0)       # best predecessor for each state
        prev = cand[back[i], ar] + cost[i]
    path = np.empty(n, np.int16)
    path[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def chosen_offsets(emis, chosen):
    """(s, d) in metres for a chosen path, and the implied translation shift."""
    idx = np.arange(len(chosen))
    return emis['s'][np.asarray(chosen)], emis['d'][idx, np.asarray(chosen)]


def coarse_poses(t0, emis, chosen):
    """Poses built straight from the coarse cell (no fine refinement).

    Kept for the cheap pre-sweep and for reporting how much the fine stage moves.
    """
    s, d = chosen_offsets(emis, chosen)
    poses = np.array(t0, np.float64, copy=True)
    for i in np.nonzero(chosen != STATE_ZERO)[0]:
        u = PL.along_track_direction(t0[i])
        lateral = np.cross(u, [0.0, 0.0, 1.0])
        lateral /= np.linalg.norm(lateral)
        poses[i, :3, 3] = t0[i][:3, 3] + s[i] * u + d[i] * lateral
    return poses


# --------------------------------------------------------------------------- #
# pass 3: refinement of the chosen cells                                       #
# --------------------------------------------------------------------------- #
def frame_axes(t0_seq):
    """Along-track and lateral unit vectors for a sequence of poses (vectorised)."""
    t0_seq = np.asarray(t0_seq, np.float64)
    u = t0_seq[:, :3, 0].copy()
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    lateral = np.cross(u, np.array([0.0, 0.0, 1.0]))
    lateral /= np.maximum(np.linalg.norm(lateral, axis=1, keepdims=True), 1e-12)
    return u, lateral


def shift_poses(t0_seq, s, d, chosen=None):
    """Apply (s, d) along/lateral offsets to front-end poses (vectorised)."""
    poses = np.array(t0_seq, np.float64, copy=True)
    u, lateral = frame_axes(t0_seq)
    keep = np.ones(len(poses), bool) if chosen is None else (np.asarray(chosen) != STATE_ZERO)
    poses[keep, :3, 3] = (poses[keep, :3, 3] + np.asarray(s)[keep, None] * u[keep]
                          + np.asarray(d)[keep, None] * lateral[keep])
    return poses


def _refine_init(keys, t0):
    _G['keys'] = keys
    _G['t0'] = t0
    _G['maps'] = None


def _refine_task(task):
    keys, t0, order, pairs = task
    if _G.get('maps') is None:
        worker_init()
    out = []
    for r, a in pairs:
        t = time.time()
        i = int(order[r])
        date, stamp = keys[i].split('/')
        a = int(a) - 1
        centre = t0[i][:3, 3]
        triple = _G['maps'].nearby(centre)[:3]
        map_points = _G['maps'].local(centre, triple)
        scan = _prepare(SCAN_FMT % (date, stamp))
        T0 = t0[i]
        u = PL.along_track_direction(T0)
        coarse = F.coarse_stage(map_points, scan, T0, u, workers=1, n_coarse=1000, topk=6)
        surface = np.asarray(coarse['surface'], np.float64)
        b = int(np.argmin(surface[a]))
        coarse['picks'] = [(a, b)]       # exactly this cell: one window, no search
        s, dd, _c, _evals, _expanded = adaptive.solve_adaptive(
            map_points, scan, T0, u, n_coarse=1000, n_fine=1500, workers=1,
            topk=6, coarse=coarse)
        out.append((int(r), int(a) + 1, float(s), float(dd), time.time() - t))
    return out


def refine_cells(keys, t0, pairs, order=None, processes=32):
    """Pass 3: refine a set of (position in order, state) pairs, one per row.

    Returns (s (m,), d (m,), state (m,), seconds (m,)) aligned with `pairs`.  The
    refinement is the project's adaptive solver restricted to the chosen cell's
    5x9 fine window (`picks` is replaced by that single cell), so it is the same
    sub-metre step the per-frame branch runs after its own argmin.
    """
    order = np.asarray(order if order is not None else np.arange(len(keys)), np.int64)
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    m = len(pairs)
    s_out = np.full(m, np.nan)
    d_out = np.full(m, np.nan)
    secs = np.zeros(m, np.float32)
    if m == 0:
        return s_out, d_out, pairs[:, 1].copy(), secs
    pos = {tuple(p): k for k, p in enumerate(pairs.tolist())}
    if len(pos) != m:
        raise ValueError('refine_cells needs unique (position, state) pairs')
    step = max(1, m // max(processes, 1) + 1)
    tasks = [(keys, t0, order, pairs[k:k + step]) for k in range(0, m, step)]
    with __import__('multiprocessing').get_context('fork').Pool(
            processes, initializer=_refine_init, initargs=(keys, t0)) as pool:
        for res in pool.imap_unordered(_refine_task, tasks):
            for r, a, s, dd, sec in res:
                k = pos[(r, a)]
                s_out[k], d_out[k], secs[k] = s, dd, sec
    return s_out, d_out, pairs[:, 1].copy(), secs


def refine_selected(keys, t0, chosen, order=None, rows=None, processes=32):
    """Pass 3 convenience: chosen state per position -> poses, in sequence order.

    A zero state keeps the front-end pose untouched; any other state is refined by
    `refine_cells`.  `rows` selects positions in the order (default: all non-zero).
    The returned poses are aligned with `order` (sequence order), not with `keys`.
    """
    order = np.asarray(order if order is not None else np.arange(len(keys)), np.int64)
    rows = [r for r in (range(len(order)) if rows is None else rows)
            if chosen[r] != STATE_ZERO]
    front = t0[order]
    poses = np.array(front, np.float64, copy=True)
    s_pos = np.zeros(len(order))
    d_pos = np.zeros(len(order))
    cell = np.zeros(len(order), np.int8)
    secs = np.zeros(len(order), np.float32)
    if rows:
        pairs = np.stack([np.asarray(rows, np.int64),
                          np.asarray([chosen[r] for r in rows], np.int64)], axis=1)
        s_v, d_v, _c, t_v = refine_cells(keys, t0, pairs, order, processes)
        s_pos[pairs[:, 0]] = s_v
        d_pos[pairs[:, 0]] = d_v
        cell[pairs[:, 0]] = pairs[:, 1]
        secs[pairs[:, 0]] = t_v
        poses = shift_poses(front, s_pos, d_pos, cell)
    return poses, dict(refined=cell > 0, s=s_pos, d=d_pos, cell=cell, seconds=secs)


# --------------------------------------------------------------------------- #
# scoring                                                                     #
# --------------------------------------------------------------------------- #
def load_truth(directory=None):
    directory = TRUTH_DIR if directory is None else directory
    truth = {}
    for name in sorted(os.listdir(directory)):
        date = name.replace('.json', '')
        for stamp, v in json.load(open(os.path.join(directory, name))).items():
            truth[(date, str(stamp))] = np.asarray(v['T_WB'], np.float64)
    return truth


def truth_arrays(keys, truth=None):
    """Ground truth as (position (n,3), rotation (n,3,3), present (n,)) for `keys`."""
    truth = load_truth() if truth is None else truth
    pos = np.full((len(keys), 3), np.nan)
    rot = np.tile(np.eye(3), (len(keys), 1, 1))
    have = np.zeros(len(keys), bool)
    for i, k in enumerate(keys):
        date, stamp = k.split('/')
        g = truth.get((date, stamp))
        if g is None:
            continue
        pos[i] = g[:3, 3]
        rot[i] = g[:3, :3]
        have[i] = True
    return pos, rot, have


def rot_deg(a, b):
    r = np.swapaxes(a, -1, -2) @ b
    c = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def errors(keys, poses, truth=None, arrays=None):
    """Per-frame position and rotation error against the evaluator truth."""
    if arrays is None:
        arrays = truth_arrays(keys, truth)
    pos, rot, have = arrays
    poses = np.asarray(poses, np.float64)
    e = np.linalg.norm(poses[:, :3, 3] - pos, axis=1)
    r = rot_deg(poses[:, :3, :3], rot)
    e[~have] = np.nan
    r[~have] = np.nan
    return e, r


def summarise(e, r, date_of=None, per_date=False):
    """The five numbers the project reports, plus the median and the counts."""
    m = np.isfinite(e) & np.isfinite(r)
    out = dict(n=int(m.sum()), mpe=float(e[m].mean()), moe=float(r[m].mean()),
               median=float(np.median(e[m])),
               ok=float(((e[m] <= 1) & (r[m] <= 2)).mean()),
               gt1=int((e[m] > 1).sum()), gt5=int((e[m] > 5).sum()),
               gt10=int((e[m] > 10).sum()), missing=int((~m).sum()))
    if per_date and date_of is not None:
        out['per_date'] = {}
        for d in sorted(set(date_of[m])):
            sel = m & (date_of == d)
            out['per_date'][d] = dict(
                n=int(sel.sum()), mpe=float(e[sel].mean()), moe=float(r[sel].mean()),
                median=float(np.median(e[sel])),
                ok=float(((e[sel] <= 1) & (r[sel] <= 2)).mean()),
                gt1=int((e[sel] > 1).sum()), gt5=int((e[sel] > 5).sum()))
    return out


def evaluate(keys, poses, truth=None, per_date=False, arrays=None):
    e, r = errors(keys, poses, truth, arrays)
    date_of = np.array([k.split('/')[0] for k in keys])
    out = summarise(e, r, date_of, per_date)
    out.update(err=e, rot=r, date=date_of)
    return out


def fmt_row(tag, m, extra=''):
    return ('%-30s %6d  %.4f  %.4f  %.4f  %5.1f%%  %5d  %5d%s'
            % (tag, m['n'], m['mpe'], m['moe'], m['median'], 100.0 * m['ok'],
               m['gt1'], m['gt5'], extra))


HEADER = ('%-30s %6s  %6s  %6s  %6s  %6s  %5s  %5s'
          % ('config', 'n', 'MPE', 'MOE', 'median', 'ok', '>1m', '>5m'))
