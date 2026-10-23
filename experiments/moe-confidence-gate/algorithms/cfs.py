"""Coarse-to-fine along-track x lateral pose search on a static LiDAR map.

No trainable parameters. Every candidate translation is scored by the one
objective the whole project reads, the mean truncated nearest-neighbour distance
between the transformed scan q and the local reference cloud M:

    C(T) = mean_{x in q} min( dist(T x, M), tau ),   tau = 1 m

Truncation makes the cost insensitive to outliers and gives it a scale: residuals
beyond one metre are indistinguishable. Rotation is frozen at the initial pose;
only the along-track (s) and lateral (d) translations are searched.

Stages
------
1. anchor     lateral offset at s = 0 over +-1 m in 0.25 m steps
2. coarse     17 along-track x 3 lateral cells at `coarse_points` query points
3. candidates best `topk` cells, >= 1.5 m apart along the track (NMS)
4. fine       adaptive window expansion: a candidate's 5 x 9 window is scored at
              `fine_points` only while its centre cost is within `thresh` of the
              best centre, and at least one window always runs

The coarse surface S(s, d) is handed back to the caller because it is also what
a compute gate needs (the descent depth C(T0) - min S), so `solve` reuses it
instead of paying for the grid twice.
"""
import numpy as np
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


def decode_scan(path):
    """Read a raw scan (uint16 mm-triples, offset by 100 m) into metres."""
    raw = np.fromfile(path, dtype=SCAN_DTYPE)
    out = np.empty((len(raw), 3), np.float64)
    for col, name in enumerate(('x', 'y', 'z')):
        np.multiply(raw[name], 0.005, out=out[:, col])
        out[:, col] -= 100.0
    return out


def voxel_down(points, size, packed=True):
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


def spread_subset(points, n):
    """Evenly spaced subset covering the whole scan, first point to last.

    The scan array is in acquisition order, which is azimuthal for a spinning
    LiDAR, so any contiguous slice keeps a wedge rather than a sample. An even
    stride fixes that but degenerates into a prefix whenever the scan holds
    fewer than 2n points (`len // n` becomes 1): at n = 4000 that was 312 of
    1502 frames in the reference project. Indexing between the first and last
    point covers the full sweep at every scan size, and costs nothing.
    """
    if len(points) <= n:
        return points
    idx = np.linspace(0, len(points) - 1, num=n)
    return points[idx.astype(np.int64)]


def query_dist(tree, points, workers=4, trunc=TRUNC):
    """Nearest-map-point distance, capped at the truncation radius.

    The upper bound lets the tree prune branches that cannot contain a point
    within `trunc`; the returned cost is identical to an unbounded query.
    """
    d = tree.query(points, k=1, workers=workers, distance_upper_bound=trunc)[0]
    return np.where(np.isfinite(d), d, trunc)


class CFS:
    """The coarse-to-fine search: one class, one algorithm, no side effects."""

    def __init__(self, coarse_points=1000, fine_points=1500, topk=6, thresh=0.10,
                 trunc=1.0, voxel=0.3, min_range=1.0, max_range=60.0, prune=False):
        """Store the search budget and the scan-preparation constants.

        `prune=True` selects the old gradient-bound fine stage instead of the
        adaptive expansion; it is kept only to reproduce the earlier numbers and
        is documented there as an invalid bound.
        """
        self.coarse_points = int(coarse_points)
        self.fine_points = int(fine_points)
        self.topk = int(topk)
        self.thresh = float(thresh)
        self.trunc = float(trunc)
        self.voxel_size = float(voxel)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.prune = bool(prune)

    # --- scan handling ---------------------------------------------------
    def decode(self, path):
        """Raw scan file -> (N, 3) float64 in metres."""
        return decode_scan(path)

    def prepare_scan(self, points):
        """Decoded scan -> range annulus -> voxel downsample (0.3 m by default)."""
        radius = np.linalg.norm(points, axis=1)
        points = points[(radius > self.min_range) & (radius < self.max_range)]
        return voxel_down(points, self.voxel_size)

    def spread_subset(self, points, n):
        """Delegates to the module helper; kept on the class for discovery."""
        return spread_subset(points, n)

    def query_dist(self, tree, points, workers=4):
        """Delegates to the module helper, with this instance's truncation."""
        return query_dist(tree, points, workers, self.trunc)

    # --- stages ----------------------------------------------------------
    def coarse(self, map_points, scan, initial, direction, workers=4):
        """Stages 1-2: lateral anchor, 17 x 3 coarse grid, candidate NMS.

        Returns the cost surface in metres together with everything the fine
        stage needs. surface[a, b] is the cost at S_COARSE[a] along the track
        and d_values[b] across it; `base_c` is the coarse query set rotated into
        the world (`initial` rotation only), so a caller can score the incoming
        pose without touching the scan again.
        """
        q_coarse = spread_subset(scan, self.coarse_points)
        tree = cKDTree(map_points)
        rot = initial[:3, :3]
        origin = initial[:3, 3]
        lateral = np.cross(direction, [0.0, 0.0, 1.0])
        lateral /= np.linalg.norm(lateral)
        base_c = q_coarse @ rot.T

        # --- anchor: solve the lateral offset at s = 0 once
        probe = np.empty((len(D_HOME) * len(q_coarse), 3))
        for a, d in enumerate(D_HOME):
            probe[a * len(q_coarse):(a + 1) * len(q_coarse)] = base_c + (origin + d * lateral)
        cost0 = np.minimum(query_dist(tree, probe, workers), self.trunc).reshape(
            len(D_HOME), -1).mean(1)
        d_home = float(D_HOME[int(np.argmin(cost0))])
        d_values = np.clip(d_home + D_COARSE, -2.0, 2.0)

        # --- coarse grid
        offsets = np.outer(d_values, lateral)
        probe = np.empty((len(S_COARSE) * len(d_values) * len(q_coarse), 3))
        for a, s in enumerate(S_COARSE):
            blk = (base_c[None] + (origin + s * direction + offsets)[:, None]).reshape(-1, 3)
            probe[a * len(d_values) * len(q_coarse):(a + 1) * len(d_values) * len(q_coarse)] = blk
        surface = np.minimum(query_dist(tree, probe, workers), self.trunc).reshape(
            len(S_COARSE), len(d_values), -1).mean(2)

        # --- pick candidates with along-track non-maximum suppression
        picks = []
        for flat in np.argsort(surface, axis=None):
            a, b = np.unravel_index(int(flat), surface.shape)
            if all(abs(S_COARSE[a] - S_COARSE[pa]) >= 1.5 for pa, _ in picks):
                picks.append((a, b))
            if len(picks) >= self.topk:
                break

        return dict(tree=tree, rot=rot, origin=origin, lateral=lateral, base_c=base_c,
                    d_home=d_home, d_values=d_values, surface=surface, picks=picks,
                    n_evals=len(D_HOME) + len(S_COARSE) * len(d_values))

    def solve(self, map_points, scan, initial, direction, workers=4, coarse=None):
        """Stages 3-4: return dict(s, d, cost, evals, expanded, surface).

        `direction` is the along-track unit vector (the initial pose's forward
        axis); rotation stays at `initial`. `coarse` is the result of `coarse()`
        — passing it reuses that surface and the numbers are bit-identical to
        recomputing it.
        """
        if coarse is None:
            coarse = self.coarse(map_points, scan, initial, direction, workers)
        if self.prune:
            s, d, cost, evals = self._fine_pruned(
                map_points, coarse, scan, direction, workers)
            expanded = None
        else:
            s, d, cost, evals, expanded = self._fine_adaptive(
                coarse, scan, direction, workers)
        return dict(s=s, d=d, cost=cost, evals=evals, expanded=expanded,
                    surface=coarse['surface'])

    # --- fine strategies -------------------------------------------------
    def _window(self, coarse, base_f, q_fine, s0, d0, direction, workers):
        """Score the 5 x 9 window centred on (s0, d0) at the fine point budget.

        Returns (poses, costs); poses is empty when the window lies entirely
        outside the s in [-8, 8] / d in [-2, 2] search box. Both fine strategies
        call this, so their per-pose numbers are identical by construction.
        """
        origin = coarse['origin']
        lateral = coarse['lateral']
        tree = coarse['tree']
        poses = [(s0 + ds, d0 + dd) for ds in S_FINE for dd in D_FINE
                 if -8.0 <= s0 + ds <= 8.0 and -2.0 <= d0 + dd <= 2.0]
        if not poses:
            return poses, None
        probe = np.empty((len(poses) * len(q_fine), 3))
        for k, (s, d) in enumerate(poses):
            probe[k * len(q_fine):(k + 1) * len(q_fine)] = base_f + (
                origin + s * direction + d * lateral)
        costs = np.minimum(query_dist(tree, probe, workers), self.trunc).reshape(
            len(poses), -1).mean(1)
        return poses, costs

    def _fine_adaptive(self, coarse, scan, direction, workers):
        """Adaptive window expansion: the only fine strategy in production.

        Replaces gradient-based pruning, whose lower bound was not valid: the
        cost is not linear over a 0.64 m step, and points beyond the truncation
        are counted as a constant although they can move inside it. The rule
        here is a plain heuristic, not a claimed bound:

          1. take the candidate centres the coarse stage already picked
          2. re-score those centres at the full fine point budget
          3. expand a candidate's 5x9 window only while its centre cost is
             within `thresh` of the best centre, always expanding at least one

        Everything evaluated still uses the exact truncated KD-tree cost.
        Returns (s, d, cost, evals, expanded).
        """
        tree = coarse['tree']
        rot = coarse['rot']
        origin = coarse['origin']
        lateral = coarse['lateral']
        d_values = coarse['d_values']
        picks = coarse['picks']

        q_fine = spread_subset(scan, self.fine_points)
        base_f = q_fine @ rot.T

        # --- candidate centres at the full fine budget
        centre = np.empty(len(picks))
        for a, (i, j) in enumerate(picks):
            s0, d0 = float(S_COARSE[i]), float(d_values[j])
            centre[a] = float(np.minimum(
                query_dist(tree, base_f + (origin + s0 * direction + d0 * lateral), workers),
                self.trunc).mean())
        order = np.argsort(centre)
        base = centre[order[0]]

        # --- adaptive expansion
        best = (np.inf, 0.0, 0.0)
        expanded = 0
        evals = coarse['n_evals'] + len(picks)
        for a in order:
            if expanded > 0 and centre[a] - base > self.thresh:
                continue
            i, j = picks[a]
            s0, d0 = float(S_COARSE[i]), float(d_values[j])
            poses, costs = self._window(coarse, base_f, q_fine, s0, d0, direction, workers)
            if not poses:
                continue
            evals += len(poses)
            expanded += 1
            t = int(np.argmin(costs))
            if costs[t] < best[0]:
                best = (float(costs[t]), poses[t][0], poses[t][1])
        return best[1], best[2], best[0], evals, expanded

    def _fine_pruned(self, map_points, coarse, scan, direction, workers):
        """Legacy fine stage: full 5x9 grid per candidate, gradient-bound skip.

        HISTORY, not a proof: the bound below assumes the cost is linear over a
        0.64 m step and treats points beyond the truncation as a constant, so it
        is NOT a valid lower bound — it can skip a window that would have won.
        Kept only to reproduce runs made before the adaptive expansion, which is
        bit-identical to full expansion on this dataset. Returns (s, d, cost,
        evals).
        """
        q_fine = spread_subset(scan, self.fine_points)
        tree = coarse['tree']
        rot = coarse['rot']
        origin = coarse['origin']
        lateral = coarse['lateral']
        d_values = coarse['d_values']
        base_f = q_fine @ rot.T

        best = (np.inf, 0.0, 0.0)
        evals = coarse['n_evals']
        for a, b in coarse['picks']:
            s0, d0 = float(S_COARSE[a]), float(d_values[b])
            centre_pts = base_f + (origin + s0 * direction + d0 * lateral)
            dist, idx = tree.query(centre_pts, k=1, workers=workers,
                                   distance_upper_bound=self.trunc)
            dist = np.where(np.isfinite(dist), dist, self.trunc)
            evals += 1
            inside = np.isfinite(dist) & (dist > 1e-9) & (dist < self.trunc)
            if inside.any():
                grad = centre_pts[inside] - map_points[idx[inside]]
                grad /= np.maximum(dist[inside], 1e-12)[:, None]
                slack = HALF_S * np.abs(grad @ direction) + HALF_D * np.abs(grad @ lateral)
                bound_in = np.minimum(np.maximum(dist[inside] - slack, 0.0), self.trunc).mean()
                bound = (bound_in * inside.sum() + float((~inside).sum())) / len(dist)
                if bound >= best[0]:
                    continue
            poses, costs = self._window(coarse, base_f, q_fine, s0, d0, direction, workers)
            if not poses:
                continue
            evals += len(poses)
            t = int(np.argmin(costs))
            if costs[t] < best[0]:
                best = (float(costs[t]), poses[t][0], poses[t][1])
        return best[1], best[2], best[0], evals
