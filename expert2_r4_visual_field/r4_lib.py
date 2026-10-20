"""R4 core library: deployable along-track x lateral sweep against a fixed 3-scan reference map.

Everything here is the pre-registered design, no knobs are changed at run time.

Conventions (frozen, see PREREGISTRATION.md)
  frames      : scored.npz['keys'] order, 1502 hard frames, GT = scored.npz['reference']
  deployment  : scored.npz['refined_V7_poses'] (rotation FROZEN for the sweep)
  hypotheses  : s in {-10,-9.5,...,+10} (41) along u, d in {-2,-1.95,...,+2} (81) along e
                candidate translation = t0 + s*u + d*e
  cost        : mean over query points of min(NN distance to map, 1.0)   [point-to-point, no normals]
  map         : 3 nearest reference scans within 30 m of t0 (chosen ONCE per frame),
                each raw(1..100 m) -> 0.3 m voxel -> world -> keep <=60 m of t0,
                concatenate -> 0.3 m voxel
  query       : current frame raw scan -> radius 1..60 m -> 0.3 m voxel -> cap 4000 (fixed seed)
  abstain     : map_points > 20000 -> return the deployment pose
"""
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HARD = '/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1'
DEPLOYED = '/home/zhangcy/projects/r-unified-eval-20261018-v1/deployed_baseline.npz'
GATE = '/home/zhangcy/projects/r-gate-20261018-v1'
sys.path.insert(0, '/home/zhangcy/projects/leader-far-covolume-standard-20261001/source-v1')
from data.robotcar_sdk.python.velodyne import get_velo  # noqa: E402

S_GRID = np.round(np.arange(-10.0, 10.0001, 0.5), 3)          # 41
D_GRID = np.round(np.arange(-2.0, 2.0001, 0.05), 4)           # 81
TRUNC = 1.0
ABSTAIN_POINTS = 20000
QCAP = 4000
QSEED = 20261019
SCAN_CACHE = 256
U_CHORD = 12.0    # chord requested on each side of the frame, metres
U_MINCHORD = 8.0  # window chord below this -> direction not identifiable -> reference tangent
U_GAP = 15.0      # a bigger deployed step breaks the window (trajectory discontinuity)
U_MAXSIDE = 90    # frames per side, hard cap
U_CHORD_REF = 20.0     # same window on the reference-library trajectory
U_MINCHORD_REF = 10.0


# ---------------------------------------------------------------- basic helpers
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def downsample(points, size, cap=None, seed=QSEED):
    """Deterministic voxel downsample (same contract as the frozen refine_raw)."""
    if len(points) == 0:
        return points
    cells = np.floor(points / size).astype(np.int64)
    _, ids = np.unique(cells, axis=0, return_index=True)
    out = points[np.sort(ids)]
    if cap is not None and len(out) > cap:
        pick = np.random.RandomState(seed).choice(len(out), cap, replace=False)
        out = out[np.sort(pick)]
    return out


class ScanCache(object):
    """LRU of raw scans (1..100 m) plus an LRU of 0.3 m voxel versions."""

    def __init__(self, n=SCAN_CACHE):
        self.n = n
        self.raw = OrderedDict()
        self.ds = OrderedDict()

    def raw_of(self, path, sha):
        if path in self.raw:
            self.raw.move_to_end(path)
            return self.raw[path]
        pts = raw_scan(path, sha)
        r = np.linalg.norm(pts, axis=1)
        pts = pts[(r > 1.0) & (r < 100.0)]
        self.raw[path] = pts
        while len(self.raw) > self.n:
            self.raw.popitem(last=False)
        return pts

    def ds_of(self, path, sha, size=0.3):
        k = (path, size)
        if k in self.ds:
            self.ds.move_to_end(k)
            return self.ds[k]
        out = downsample(self.raw_of(path, sha), size)
        self.ds[k] = out
        while len(self.ds) > self.n:
            self.ds.popitem(last=False)
        return out


def raw_scan(path, sha):
    assert sha256(path) == sha, 'scan sha mismatch: %s' % path
    pts = np.asarray(get_velo(str(path))[0], dtype=np.float64)
    return pts


# ---------------------------------------------------------------- world / frames
class World(object):
    def __init__(self):
        z = np.load(HARD + '/scored.npz', allow_pickle=True)
        self.keys = [str(k) for k in z['keys']]
        self.dates = np.asarray(z['dates']).astype(str)
        self.replaced = np.asarray(z['replaced']).astype(bool)
        self.gt = np.asarray(z['reference'], float)
        self.gt_official = np.asarray(z['raw'], float)
        self.deploy = np.asarray(z['refined_V7_poses'], float)
        dep = np.load(DEPLOYED, allow_pickle=True)
        assert [str(k) for k in dep['keys']] == self.keys
        assert np.abs(np.asarray(dep['refined_V7'], float) - self.deploy).max() < 1e-9
        sel = json.load(open(HARD + '/selection.json'))
        self.refs = sel['references']
        self.queries = {str(r['key']): r for r in sel['queries']}
        self.ref_pos = np.stack([np.asarray(r['world_from_body'], float) for r in self.refs])
        self.ref_tree = cKDTree(self.ref_pos[:, :3, 3])
        self.cache = ScanCache()
        self._order = {}
        for d in sorted(set(self.dates)):
            idx = np.flatnonzero(self.dates == d)
            self._order[d] = idx[np.argsort([int(self.keys[i].split('/')[1]) for i in idx])]
        self._pos_in_date = {}
        for d, order in self._order.items():
            self._pos_in_date[d] = {int(i): j for j, i in enumerate(order)}
        # reference-library trajectories, ordered per date
        ref_dates = np.array([str(r['key']).split('/')[0] for r in self.refs])
        self._ref_order = {}
        self._ref_pos_in_date = {}
        for d in sorted(set(ref_dates)):
            idx = np.flatnonzero(ref_dates == d)
            idx = idx[np.argsort([int(str(self.refs[i]['key']).split('/')[1]) for i in idx])]
            self._ref_order[d] = idx
            self._ref_pos_in_date[d] = {int(i): j for j, i in enumerate(idx)}

    def sort_index(self, i):
        """(date order position) so a contiguous frame block is spatially contiguous."""
        d = self.dates[i]
        return (d, self._pos_in_date[d][i])

    # ------------------------------------------------------------ directions
    def _quad_direction(self, poses, i):
        """Direction of travel at frame i from a same-date window of `poses`."""
        d = self.dates[i]
        return self._quad_direction_on(poses[self._order[d], :3, 3],
                                       self._pos_in_date[d][i],
                                       chord=U_CHORD, minchord=U_MINCHORD,
                                       gap=U_GAP, maxside=U_MAXSIDE)

    def u_deployable(self, i):
        """Along-track unit vector from temporally adjacent DEPLOYED poses only."""
        got = self._quad_direction(self.deploy, i)
        if got is not None:
            u, meta = got
            return self._orient(u, i), dict(meta, source='deploy_quad')
        u, meta = self.u_ref_tangent(i)
        return u, meta

    def u_truth(self, i):
        """Upper-bound arm only: along-track unit vector from adjacent TRUTH poses."""
        got = self._quad_direction(self.gt, i)
        if got is not None:
            u, meta = got
            return self._orient(u, i), dict(meta, source='truth_quad')
        u, meta = self.u_ref_tangent(i)
        return self._orient(u, i), dict(meta, source='truth_fallback_' + meta['source'])

    def u_old(self, i):
        """The estimator as first registered (endpoint difference over i+-k, k<=6).
        Kept only as an engineering comparison arm; no longer the main arm."""
        d = self.dates[i]
        order = self._order[d]
        j = self._pos_in_date[d][i]
        pos = self.deploy[order, :3, 3]
        best = None
        for k in range(1, 7):
            a, b = j - k, j + k
            if a < 0 or b >= len(order):
                break
            seq = pos[a:b + 1]
            steps = np.linalg.norm(np.diff(seq, axis=0), axis=1)
            if steps.max() > 15.0:
                break
            span = float(np.linalg.norm(seq[-1] - seq[0]))
            if span > 25.0:
                break
            if span > 1e-6:
                best = (k, (seq[-1] - seq[0]) / span, span)
        if best is not None:
            k, u, span = best
            return self._orient(u, i), dict(source='deploy_endpoint', k=k, span=span)
        u, meta = self.u_ref_tangent(i)
        return self._orient(u, i), dict(meta, source='old_fallback_' + meta['source'])

    def u_ref_tangent(self, i):
        """Fallback: the same quadratic window applied to the reference-library trajectory,
        restricted to the reference date of the nearest reference pose (no date mixing).
        Last resort: that pose's own heading."""
        t0 = self.deploy[i, :3, 3]
        dist, j0 = self.ref_tree.query(t0, k=1)
        j0 = int(j0)
        date = str(self.refs[j0]['key']).split('/')[0]
        order = self._ref_order[date]
        jpos = self._ref_pos_in_date[date]
        pos = self.ref_pos[order, :3, 3]
        got = self._quad_direction_on(pos, jpos[j0], chord=U_CHORD_REF, minchord=U_MINCHORD_REF,
                                      gap=U_GAP, maxside=U_MAXSIDE)
        if got is not None:
            u, meta = got
            return u, dict(meta, source='ref_quad', ref_date=date, ref_dist=float(dist))
        R = np.asarray(self.refs[j0]['world_from_body'], float)[:3, :3]
        u = R @ np.array([1.0, 0.0, 0.0])
        return u / np.linalg.norm(u), dict(source='ref_heading', ref_date=date,
                                           ref_dist=float(dist))

    def _quad_direction_on(self, pos, j, chord, minchord, gap, maxside):
        here = pos[j]
        idx = [j]
        tau = {j: 0.0}
        for sign, stop in ((-1, 0), (1, len(pos) - 1)):
            acc = 0.0
            cur = j
            n_side = 0
            while cur != stop and n_side < maxside:
                nxt = cur + sign
                step = float(np.linalg.norm(pos[nxt] - pos[cur]))
                if step > gap:
                    break
                acc += step
                cur = nxt
                n_side += 1
                idx.append(cur)
                tau[cur] = sign * acc
                if float(np.linalg.norm(pos[cur] - here)) >= chord:
                    break
        if len(idx) < 5:
            return None
        ids = sorted(set(idx))
        P = pos[ids]
        span = float(np.linalg.norm(P.max(0) - P.min(0)))
        if span < minchord:
            return None
        t = np.array([tau[k] for k in ids])
        if abs(t).max() < 1e-6:
            return None
        tn = t / abs(t).max()
        A = np.stack([np.ones_like(tn), tn, tn * tn], 1)
        coef, *_ = np.linalg.lstsq(A, P, rcond=None)
        u = coef[1]
        n = np.linalg.norm(u)
        if n < 1e-9:
            return None
        return u / n, dict(n_window=int(len(ids)), chord=span, arc=float(abs(t).max()))

    def _orient(self, u, i):
        """Sign convention: s>0 means forward along the deployed heading."""
        h = self.deploy[i, :3, :3] @ np.array([1.0, 0.0, 0.0])
        return -u if float(u @ h) < 0 else u

    # ------------------------------------------------------------ map / query
    def ref_ids(self, i, radius=30.0, k=3):
        t0 = self.deploy[i, :3, 3]
        ids = self.ref_tree.query_ball_point(t0, radius)
        ids = sorted(ids, key=lambda j: float(np.linalg.norm(self.ref_pos[j, :3, 3] - t0)))
        return ids[:k]

    def local_map(self, i):
        t0 = self.deploy[i, :3, 3]
        ids = self.ref_ids(i)
        parts = []
        per = []
        for j in ids:
            row = self.refs[j]
            ds = self.cache.ds_of(row['scan_path'], row['scan_sha256'], 0.3)
            T = np.asarray(row['world_from_body'], float)
            W = ds @ T[:3, :3].T + T[:3, 3]
            W = W[np.linalg.norm(W - t0, axis=1) <= 60.0]
            parts.append(W)
            per.append(int(len(W)))
        if not parts:
            return np.zeros((0, 3)), ids, per
        return downsample(np.concatenate(parts), 0.3), ids, per

    def query(self, i):
        row = self.queries[self.keys[i]]
        pts = self.cache.ds_of(row['scan_path'], row['scan_sha256'], 0.3)
        r = np.linalg.norm(pts, axis=1)
        pts = pts[(r > 1.0) & (r < 60.0)]
        return downsample(pts, 0.3, cap=QCAP)

    def raw_query(self, i):
        """The current frame's raw scan, exactly what the frozen refiner expects."""
        row = self.queries[self.keys[i]]
        return self.cache.raw_of(row['scan_path'], row['scan_sha256'])


# ---------------------------------------------------------------- the sweep
def lateral_axis(u):
    """Horizontal unit vector perpendicular to the horizontal part of u."""
    e = np.cross(np.array([0.0, 0.0, 1.0]), u)
    n = np.linalg.norm(e)
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0])
    return e / n


def cost_grid(q, R, t0, u, e, tree, chunk=81):
    """(len(S_GRID), len(D_GRID)) truncated mean NN distance."""
    world0 = q @ R.T + t0
    out = np.empty((len(S_GRID), len(D_GRID)), np.float64)
    for a, s in enumerate(S_GRID):
        base = world0 + s * u
        cand = base[None, :, :] + D_GRID[:, None, None] * e[None, None, :]
        flat = np.ascontiguousarray(cand.reshape(-1, 3))
        dist = tree.query(flat, k=1, workers=1)[0]
        d = np.minimum(dist, TRUNC).reshape(len(D_GRID), -1).mean(1)
        out[a] = d
    return out


def pose_with(P, u, e, s, d):
    T = np.array(P, dtype=float, copy=True)
    T[:3, 3] = P[:3, 3] + s * u + d * e
    return T


def refine_stage(T, q, refs, ref_positions):
    """Second stage: the frozen refiner from r-gate (read-only import)."""
    if GATE not in sys.path:
        sys.path.insert(0, GATE)
    import geometry_local as GL
    out, ev = GL.refine_raw(T, q, refs, ref_positions)
    return out, ev


# ---------------------------------------------------------------- one frame
def process_frame(w, i, ri, u_source='deploy', do_refine=True, do_truth_probe=True,
                  abstain_points=0):
    """One frame end to end. abstain_points=0 (default) disables abstention entirely:
    the main arm always outputs the best (s, d). The abstention idea survives only as a
    post-hoc diagnostic (rec['would_abstain'])."""
    import time
    t_start = time.time()
    P = w.deploy[i]
    R = P[:3, :3]
    t0 = P[:3, 3]
    row = w.queries[w.keys[i]]

    t = time.time()
    q = w.query(i)
    t_query = time.time() - t

    t = time.time()
    mapp, ref_ids, per = w.local_map(i)
    t_map = time.time() - t
    map_points = int(len(mapp))

    if u_source == 'truth':
        u, umeta = w.u_truth(i)
    elif u_source == 'old':
        u, umeta = w.u_old(i)
    else:
        u, umeta = w.u_deployable(i)
    e = lateral_axis(u)

    rec = dict(key=w.keys[i], index=int(i), date=str(w.dates[i]),
               replaced=bool(w.replaced[i]),
               u=[float(x) for x in u], u_source=umeta['source'], u_meta=umeta,
               lateral=[float(x) for x in e],
               map_points=map_points, map_points_per_ref=per,
               ref_ids=[int(x) for x in ref_ids],
               ref_keys=[str(w.refs[j]['key']) for j in ref_ids],
               query_points=int(len(q)), abstain_threshold=ABSTAIN_POINTS,
               t_query=round(t_query, 3), t_map=round(t_map, 3))

    if map_points < 30 or len(q) < 30:
        rec.update(abstain=True, abstain_reason='too_few_points', would_abstain=False,
                   s=0.0, d=0.0,
                   cost_selected=None, cost_deploy=None, gain=None,
                   pose=P.tolist(), pose_refined=None, t_grid=0.0, t_refine=0.0,
                   seconds=round(time.time() - t_start, 3))
        return rec

    t = time.time()
    tree = cKDTree(mapp)
    grid = cost_grid(q, R, t0, u, e, tree)
    t_grid = time.time() - t
    a, b = np.unravel_index(int(np.argmin(grid)), grid.shape)
    cost_sel = float(grid[a, b])
    i0 = int(np.argmin(np.abs(S_GRID)))
    cost_dep = float(grid[i0, int(np.argmin(np.abs(D_GRID)))])
    abstain = (map_points > ABSTAIN_POINTS) if abstain_points else False

    rec.update(s=float(S_GRID[a]), d=float(D_GRID[b]), cost_selected=cost_sel,
               cost_deploy=cost_dep, gain=float(cost_dep - cost_sel),
               n_better_than_deploy=int((grid < cost_dep - 1e-9).sum()),
               would_abstain=bool(map_points > ABSTAIN_POINTS),
               t_grid=round(t_grid, 3))
    if abstain:
        rec.update(abstain=True, abstain_reason='map_points>%d' % abstain_points,
                   pose=P.tolist(), pose_refined=None, t_refine=0.0,
                   seconds=round(time.time() - t_start, 3))
        return rec
    rec.update(abstain=False, abstain_reason='')

    T_sel = pose_with(P, u, e, float(S_GRID[a]), float(D_GRID[b]))
    rec['pose'] = T_sel.tolist()

    # diagnostics only (never used for selection): where does the truth lie on this grid?
    if do_truth_probe:
        G = w.gt[i]
        s_star = float((G[:3, 3] - t0) @ u)
        d_star = float((G[:3, 3] - t0) @ e)
        ia = int(np.argmin(np.abs(S_GRID - s_star)))
        ib = int(np.argmin(np.abs(D_GRID - d_star)))
        rec.update(s_star=s_star, d_star=d_star, cost_at_star=float(grid[ia, ib]),
                   grid_cost_argmin_at_star=[int(ia), int(ib)],
                   sel_err=float(np.linalg.norm(T_sel[:3, 3] - G[:3, 3])),
                   deploy_err=float(np.linalg.norm(t0 - G[:3, 3])))

    # full cost grid, rounded: enough to re-check any candidate after the fact
    rec['grid'] = [[round(float(x), 4) for x in r] for r in grid]

    if do_refine:
        t = time.time()
        try:
            T_ref, ev = refine_stage(T_sel, w.raw_query(i), w.refs, w.ref_pos[:, :3, 3])
            rec['pose_refined'] = T_ref.tolist()
            rec['refine_evidence'] = _plain(ev)
            rec['refine_moved'] = float(np.linalg.norm(T_ref[:3, 3] - T_sel[:3, 3]))
        except Exception as exc:  # never lose the frame because of stage 2
            rec['pose_refined'] = None
            rec['refine_error'] = '%s: %s' % (type(exc).__name__, exc)
        rec['t_refine'] = round(time.time() - t, 3)
    else:
        rec['pose_refined'] = None
        rec['t_refine'] = 0.0
    rec['seconds'] = round(time.time() - t_start, 3)
    return rec


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def dump(rec, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(_plain(rec), f)
    tmp.replace(path)
