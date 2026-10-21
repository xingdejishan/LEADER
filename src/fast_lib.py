"""Along-track / lateral candidate search over a static LiDAR reference map.

No trainable parameters: a candidate pose is scored by the mean truncated
nearest-neighbour distance between the current scan and the reference cloud.

Stage 1 (coarse)  : 21 along-track positions x 3 lateral values, 1500 query points.
Stage 2 (fine)    : for the best 6 coarse cells, a local 5x9 grid at 4000 points.
Stage 3 (optional): hand the winning pose to an external local refiner.
"""
import json
import numpy as np
from collections import OrderedDict
from scipy.spatial import cKDTree

SCAN_DTYPE = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('i', 'u1'), ('l', 'u1')])

S_COARSE = np.arange(-8.0, 8.0001, 1.0)     # 17 along-track positions
D_HOME = np.arange(-1.0, 1.0001, 0.25)      # anchor solve over +-1 m
D_COARSE = np.array([-0.5, 0.0, 0.5])       # 3 lateral values around the anchor
S_FINE = np.arange(-2, 3) * 0.25            # +-0.5 m at 0.25 m
D_FINE = np.arange(-0.4, 0.4001, 0.1)       # +-0.4 m at 0.1 m
HALF_S = 0.5
HALF_D = 0.4
TRUNC = 1.0


def decode(path):
    """Read a raw scan (uint16 mm-triples) into metres."""
    raw = np.fromfile(path, dtype=SCAN_DTYPE)
    out = np.empty((len(raw), 3), np.float64)
    for col, name in enumerate(('x', 'y', 'z')):
        np.multiply(raw[name], 0.005, out=out[:, col])
        out[:, col] -= 100.0
    return out


def voxel(points, size, packed=True):
    """Voxel-grid downsampling, one representative per cell.

    packed=True packs the three integer cell coordinates into one int64 key.
    This is bit-identical to np.unique(axis=0) but ~6x faster.
    """
    key = np.floor(points / size).astype(np.int64)
    if packed:
        key = key - key.min(0)
        span = int(key.max()) + 2
        flat = (key[:, 0] * span + key[:, 1]) * span + key[:, 2]
        _, idx = np.unique(flat, return_index=True)
    else:
        _, idx = np.unique(key, axis=0, return_index=True)
    return points[np.sort(idx)]


class MapSource:
    """Reference-map provider with a cache keyed on the reference triple.

    The merged cloud of three fixed references does not depend on the query
    frame, so it is built once and only cropped per frame.
    """

    def __init__(self, refs, positions, radius=60.0, max_refs=600, max_maps=400):
        self.refs = refs
        self.pos = positions
        self.tree = cKDTree(positions)
        self.radius = radius
        self.max_refs = max_refs
        self.max_maps = max_maps
        self._ref_cache = OrderedDict()
        self._map_cache = OrderedDict()
        self.hit = 0
        self.miss = 0

    def _ref_cloud(self, j):
        cloud = self._ref_cache.get(j)
        if cloud is not None:
            self._ref_cache.move_to_end(j)
            return cloud
        row = self.refs[j]
        t = np.asarray(row['world_from_body'])
        cloud = voxel(decode(row['scan_path']), 0.3) @ t[:3, :3].T + t[:3, 3]
        self._ref_cache[j] = cloud
        while len(self._ref_cache) > self.max_refs:
            self._ref_cache.popitem(last=False)
        return cloud

    def nearby(self, centre):
        idx = self.tree.query_ball_point(centre, 30.0)
        return tuple(sorted(idx, key=lambda n: np.linalg.norm(self.pos[n] - centre))[:3])

    def local(self, centre, triple=None):
        triple = self.nearby(centre) if triple is None else triple
        merged = self._map_cache.get(triple)
        if merged is None:
            self.miss += 1
            merged = voxel(np.concatenate([self._ref_cloud(j) for j in triple]), 0.3)
            self._map_cache[triple] = merged
            while len(self._map_cache) > self.max_maps:
                self._map_cache.popitem(last=False)
        else:
            self.hit += 1
            self._map_cache.move_to_end(triple)
        return merged[np.linalg.norm(merged - centre, axis=1) <= self.radius]


def query_dist(tree, points, workers=4, trunc=TRUNC):
    """Nearest-map-point distance, capped at the truncation radius.

    The upper bound lets the tree prune branches that cannot contain a point
    within `trunc`; the returned cost is identical to an unbounded query.
    """
    d = tree.query(points, k=1, workers=workers, distance_upper_bound=trunc)[0]
    return np.where(np.isfinite(d), d, trunc)


def solve(map_points, scan, initial, direction, workers=4, n_coarse=1500, n_fine=4000,
          topk=6, prune=True):
    """Return (s, d, cost, n_evals).

    `direction` is the along-track unit vector (taken from the initial pose's
    forward axis). Rotation is held fixed at `initial`.
    """
    q_coarse = scan[:min(n_coarse, len(scan))]
    q_fine = scan[:min(n_fine, len(scan))]
    tree = cKDTree(map_points)
    rot = initial[:3, :3]
    origin = initial[:3, 3]
    lateral = np.cross(direction, [0.0, 0.0, 1.0])
    lateral /= np.linalg.norm(lateral)
    base_c = q_coarse @ rot.T
    base_f = q_fine @ rot.T

    # --- anchor: solve the lateral offset at s = 0 once
    probe = np.empty((len(D_HOME) * len(q_coarse), 3))
    for a, d in enumerate(D_HOME):
        probe[a * len(q_coarse):(a + 1) * len(q_coarse)] = base_c + (origin + d * lateral)
    cost0 = np.minimum(query_dist(tree, probe, workers), TRUNC).reshape(len(D_HOME), -1).mean(1)
    d_home = float(D_HOME[int(np.argmin(cost0))])
    d_values = np.clip(d_home + D_COARSE, -2.0, 2.0)

    # --- coarse grid
    offsets = np.outer(d_values, lateral)
    probe = np.empty((len(S_COARSE) * len(d_values) * len(q_coarse), 3))
    for a, s in enumerate(S_COARSE):
        blk = (base_c[None] + (origin + s * direction + offsets)[:, None]).reshape(-1, 3)
        probe[a * len(d_values) * len(q_coarse):(a + 1) * len(d_values) * len(q_coarse)] = blk
    coarse = np.minimum(query_dist(tree, probe, workers), TRUNC).reshape(
        len(S_COARSE), len(d_values), -1).mean(2)

    # --- pick candidates with along-track non-maximum suppression
    picks = []
    for flat in np.argsort(coarse, axis=None):
        a, b = np.unravel_index(int(flat), coarse.shape)
        if all(abs(S_COARSE[a] - S_COARSE[pa]) >= 1.5 for pa, _ in picks):
            picks.append((a, b))
        if len(picks) >= topk:
            break

    # --- fine stage with a gradient lower bound for pruning
    best = (np.inf, 0.0, 0.0)
    evals = len(D_HOME) + len(S_COARSE) * len(d_values)
    for a, b in picks:
        s0, d0 = float(S_COARSE[a]), float(d_values[b])
        centre_pts = base_f + (origin + s0 * direction + d0 * lateral)
        dist, idx = tree.query(centre_pts, k=1, workers=workers, distance_upper_bound=TRUNC)
        dist = np.where(np.isfinite(dist), dist, TRUNC)
        evals += 1
        if prune:
            inside = np.isfinite(dist) & (dist > 1e-9) & (dist < TRUNC)
            if inside.any():
                grad = centre_pts[inside] - map_points[idx[inside]]
                grad /= np.maximum(dist[inside], 1e-12)[:, None]
                slack = HALF_S * np.abs(grad @ direction) + HALF_D * np.abs(grad @ lateral)
                bound_in = np.minimum(np.maximum(dist[inside] - slack, 0.0), TRUNC).mean()
                bound = (bound_in * inside.sum() + float((~inside).sum())) / len(dist)
                if bound >= best[0]:
                    continue
        poses = [(s0 + ds, d0 + dd) for ds in S_FINE for dd in D_FINE
                 if -8.0 <= s0 + ds <= 8.0 and -2.0 <= d0 + dd <= 2.0]
        if not poses:
            continue
        probe = np.empty((len(poses) * len(q_fine), 3))
        for t, (s, d) in enumerate(poses):
            probe[t * len(q_fine):(t + 1) * len(q_fine)] = base_f + (
                origin + s * direction + d * lateral)
        costs = np.minimum(query_dist(tree, probe, workers), TRUNC).reshape(len(poses), -1).mean(1)
        evals += len(poses)
        t = int(np.argmin(costs))
        if costs[t] < best[0]:
            best = (float(costs[t]), poses[t][0], poses[t][1])
    return best[1], best[2], best[0], evals
