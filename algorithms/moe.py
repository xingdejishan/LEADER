"""The per-frame mixture-of-experts localiser: search, ambiguity, choice.

One objective, three decisions. The pose problem is the minimisation of a single
robust cost, the mean truncated nearest-neighbour distance between the scan and
the local reference cloud (truncation 1 m):

    C(T) = mean_{x in q} min( dist(T x, M), 1.0 )

    T0 (leader) --► coarse surface S(s, d), 17 x 3, coarse_points queries   (CFS)
                      |
                      +- descent depth g = C(T0) - min S     -> compute gate (gate2)
                      +- ambiguity ratio rho = (C_far - C0)/C0 -> modality query
                      +- adaptive fine stage (rotation preserved)
                              |
    visual hypothesis --------+--► the cheaper one wins, minus a switching penalty

The surface is built once per frame and every later decision reads it; the fine
stage is never charged for it a second time. Nothing here learns: the only
external inputs are the initial pose, the reference cloud and an optional visual
pose.

Stages, in order:

  1. map       three other-date references near the initial position, merged and
               voxelised once per triple, cropped per frame (MapSource)
  2. coarse    `CFS.coarse`: lateral anchor, 17 x 3 grid, candidate NMS
  3. gate      descend only if the surface actually drops (gate2, -1 = disabled)
  4. fine      `CFS.solve`: adaptive window expansion at the full fine budget
  5. ambiguity ratio on the same map and tree the search built
  6. choice    the visual hypothesis replaces the search pose only when it is
               cheaper by more than `delta` metres of mean truncated residual

No file writes, no argument parsing, no environment mutation at import time.
"""
from collections import OrderedDict

import numpy as np
from scipy.spatial import cKDTree

try:
    from .cfs import CFS, decode_scan, voxel_down
except ImportError:                     # package dir itself on sys.path
    from cfs import CFS, decode_scan, voxel_down

COST_TRUNC = 1.0        # the one cost function, in metres
ARB_POINTS = 4000       # query points used when scoring a hypothesis (prefix, as before)
AMB_GRID = np.round(np.arange(-8.0, 8.0001, 1.0), 3)   # 17 along-track positions
AMB_FAR_MIN = 5.0       # a "competing" offset is |s| > this
AMB_POINTS = 4000       # ambiguity query points (sensitive quantity: do not lower)
AMB_THRESHOLD = 0.21    # ratio below this -> ask the other modality
REF_VOXEL = 0.3         # reference / map voxel size, metres
DEFAULT_DELTA = 0.05    # switching penalty, in metres of mean truncated residual


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
        cloud = voxel_down(decode_scan(row['scan_path']), REF_VOXEL) @ t[:3, :3].T + t[:3, 3]
        self._ref_cache[j] = cloud
        while len(self._ref_cache) > self.max_refs:
            self._ref_cache.popitem(last=False)
        return cloud

    def nearby(self, centre):
        """The three nearest reference poses to `centre`, nearest first."""
        idx = self.tree.query_ball_point(centre, 30.0)
        return tuple(sorted(idx, key=lambda n: np.linalg.norm(self.pos[n] - centre))[:3])

    def local(self, centre, triple=None):
        """Crop the merged cloud of `triple` (or of the nearest three) to `radius`."""
        triple = self.nearby(centre) if triple is None else triple
        merged = self._map_cache.get(triple)
        if merged is None:
            self.miss += 1
            merged = voxel_down(np.concatenate([self._ref_cloud(j) for j in triple]),
                                REF_VOXEL)
            self._map_cache[triple] = merged
            while len(self._map_cache) > self.max_maps:
                self._map_cache.popitem(last=False)
        else:
            self.hit += 1
            self._map_cache.move_to_end(triple)
        return merged[np.linalg.norm(merged - centre, axis=1) <= self.radius]


def along_track_direction(initial_pose):
    """Along-track unit vector: the initial pose's own forward axis.

    Single frame, no neighbouring poses involved. Inherits the initial pose's
    heading error (median 1.3 deg on this dataset), which costs ~0.056 m of
    lateral solution per degree.
    """
    u = np.array(initial_pose[:3, 0], dtype=np.float64)
    n = np.linalg.norm(u)
    if n < 1e-9:
        raise ValueError('degenerate rotation in initial pose')
    return u / n




class MoE:
    """One prepared frame in, one corrected pose out."""

    DEFAULTS = dict(coarse_points=1000, fine_points=1500, topk=6, thresh=0.10,
                    trunc=COST_TRUNC, voxel=0.3, min_range=1.0, max_range=60.0,
                    workers=4, gate2=-1.0, delta=DEFAULT_DELTA, rot=False,
                    amb_threshold=AMB_THRESHOLD, arb_points=ARB_POINTS,
                    crop_radius=60.0, reference_count=3, prune=False)

    def __init__(self, refs, positions, cfg=None):
        """`refs` is the list of reference dicts, `positions` their (N, 4, 4) poses.

        `cfg` overrides any of DEFAULTS; unknown keys raise, so a typo cannot
        silently fall back to a default.
        """
        self.refs = refs
        self.positions = np.asarray(positions, dtype=np.float64)
        self.cfg = dict(self.DEFAULTS)
        if cfg:
            unknown = set(cfg) - set(self.DEFAULTS)
            if unknown:
                raise KeyError('unknown config keys: %s' % sorted(unknown))
            self.cfg.update(cfg)
        if self.cfg['rot']:
            raise ValueError('rotation-only refinement has been removed; rot must be False')
        self.cfs = CFS(coarse_points=self.cfg['coarse_points'],
                       fine_points=self.cfg['fine_points'],
                       topk=self.cfg['topk'],
                       thresh=self.cfg['thresh'],
                       trunc=self.cfg['trunc'],
                       voxel=self.cfg['voxel'],
                       min_range=self.cfg['min_range'],
                       max_range=self.cfg['max_range'],
                       prune=self.cfg['prune'])
        self.maps = MapSource(refs, self.positions, radius=self.cfg['crop_radius'])

    # --- cost and scan helpers -------------------------------------------
    def cost(self, tree, points):
        """Mean truncated nearest-neighbour distance, in metres."""
        d = tree.query(points, k=1, workers=1,
                       distance_upper_bound=self.cfg['trunc'])[0]
        return float(np.where(np.isfinite(d), d, self.cfg['trunc']).mean())

    def cost_at(self, pose, points):
        """Score a hypothesis on the map cropped around its own position."""
        centre = pose[:3, 3]
        mp = self.maps.local(centre, self.maps.nearby(centre)[:self.cfg['reference_count']])
        return self.cost(cKDTree(mp), points @ pose[:3, :3].T + centre)

    def prepare_scan(self, record):
        """Frame record -> prepared scan (decode, range annulus, voxel)."""
        return self.cfs.prepare_scan(self.cfs.decode(record['scan_path']))

    def ambiguity(self, tree, scan, pose, direction):
        """Along-track ambiguity of `pose`, on the map the search already built.

        A competing position five to ten metres away along the track must not
        explain the scan almost as well as the current pose:

            rho = (C_far - C_0) / C_0

        `tree` and `scan` are the search's own map and scan, so no frame pays for
        a second map. The query set is an even stride over the scan: the scan is
        in acquisition order, so a prefix covers only part of the sweep.
        Returns (rho, c0, cfar).
        """
        n = min(int(AMB_POINTS), len(scan))
        q = scan if n >= len(scan) else scan[np.linspace(0, len(scan) - 1, n).astype(np.int64)]
        q = q @ pose[:3, :3].T
        origin = pose[:3, 3]
        probe = np.empty((len(AMB_GRID) * len(q), 3))
        for a, s in enumerate(AMB_GRID):
            probe[a * len(q):(a + 1) * len(q)] = q + (origin + s * direction)
        trunc = self.cfg['trunc']
        d = tree.query(probe, k=1, workers=1, distance_upper_bound=trunc)[0]
        costs = np.where(np.isfinite(d), d, trunc).reshape(len(AMB_GRID), -1).mean(1)
        c0 = float(costs.min())
        cfar = float(costs[np.abs(AMB_GRID) > AMB_FAR_MIN].min())
        return (cfar - c0) / max(c0, 1e-6), c0, cfar


    # --- the pipeline ----------------------------------------------------
    def localize(self, scan, initial_pose, visual_pose=None, delta=DEFAULT_DELTA):
        """Run the full chain for one frame; returns (pose, diagnostics).

        `scan` is a prepared scan (see `prepare_scan`), `initial_pose` a 4x4.
        `visual_pose=None` is route 3: no visual hypothesis, so no arbitration.
        `delta` defaults to the configured switching penalty.
        """
        cfg = self.cfg
        delta = cfg['delta'] if delta is None else float(delta)
        T0 = np.asarray(initial_pose, dtype=np.float64)
        centre = T0[:3, 3].astype(np.float64)
        workers = int(cfg['workers'])

        # stage 1: reference triple and local map (cached per triple)
        triple = self.maps.nearby(centre)[:cfg['reference_count']]
        map_points = self.maps.local(centre, triple)

        # stage 2: direction, coarse surface (built once, reused below)
        u = along_track_direction(T0)
        lateral = np.cross(u, [0.0, 0.0, 1.0])
        lateral /= np.linalg.norm(lateral)
        coarse = self.cfs.coarse(map_points, scan, T0, u, workers)
        c_zero = self.cost(coarse['tree'], coarse['base_c'] + centre)
        gain = float(c_zero - float(coarse['surface'].min()))

        # stage 3-4: gate, then the fine search reusing the surface
        T3 = np.array(T0, dtype=np.float64)
        ran = False
        s = d = 0.0
        expanded = 0
        if gain > cfg['gate2']:
            sol = self.cfs.solve(map_points, scan, T0, u, workers, coarse=coarse)
            s, d = float(sol['s']), float(sol['d'])
            expanded = sol['expanded']
            T3 = np.array(T0, dtype=np.float64)
            T3[:3, 3] = centre + s * u + d * lateral
            ran = True

        rotated = False

        # stage 6: ambiguity of the incoming pose on the search's map
        ratio, _, _ = self.ambiguity(coarse['tree'], scan, T0, u)

        # stage 7: arbitration by the same cost, with a switching penalty
        accepted = False
        c_pose = v_pose = float('nan')
        if visual_pose is not None and np.isfinite(ratio) and ratio < cfg['amb_threshold']:
            q = scan[:min(cfg['arb_points'], len(scan))]
            c_pose = self.cost_at(T3, q)
            v_pose = self.cost_at(np.asarray(visual_pose, dtype=np.float64), q)
            accepted = bool(v_pose < c_pose - delta)

        diag = dict(gain=gain, ran_fine=bool(ran), s=s, d=d, expanded=expanded,
                    rotated=rotated, ratio=float(ratio), c_pose=float(c_pose),
                    v_pose=float(v_pose),
                    dcost=float(v_pose - c_pose)
                    if np.isfinite(c_pose) and np.isfinite(v_pose) else float('nan'),
                    chosen='visual' if accepted else 'route3')
        return (np.asarray(visual_pose, dtype=np.float64) if accepted else T3), diag

    def localize_many(self, items, workers=None):
        """Convenience loop over (key, scan, initial[, visual]) items.

        Single process by design: the per-frame work already uses `workers`
        query threads, and the reference driver keeps the multithreading outside
        so that the numbers per frame are unchanged. Returns [(key, pose, diag)].
        """
        prev = self.cfg['workers']
        if workers is not None:
            self.cfg['workers'] = int(workers)
        try:
            out = []
            for item in items:
                key, scan, initial = item[0], item[1], item[2]
                visual = item[3] if len(item) > 3 else None
                pose, diag = self.localize(scan, initial, visual_pose=visual)
                out.append((key, pose, diag))
            return out
        finally:
            self.cfg['workers'] = prev
