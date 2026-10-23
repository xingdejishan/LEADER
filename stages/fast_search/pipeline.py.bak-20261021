"""End-to-end chain: frame record -> corrected pose.

This module is the "pipeline" entry point. `fast_lib` holds the pieces; here
they are wired into one call so the chain is explicit and reusable.

    record ──► scan decode ──► voxel ─┐
                                      ├─► solve() ──► pose ──► (refiner) ──► pose
    initial pose ──► along-track dir ─┘        ▲
                                               │
    reference library ──► MapSource ───────────┘

Stages, in order:

  1. reference triple   three other-date references near the initial position
  2. map assembly       merged + voxelised once per triple, cropped per frame
  3. scan preparation   decode, range filter, voxel downsample
  4. direction          along-track unit vector from the initial pose's x axis
  5. coarse grid        17 along-track x 3 lateral, 1500 query points
  6. candidate NMS      best 6 cells, >= 1.5 m apart along-track
  7. fine grid          5 x 9 per candidate, 4000 query points, gradient bound
  8. refine (optional)  external local refiner, e.g. a point-to-point ICP pass

No stage learns anything; the only external input is the initial pose.
"""
import numpy as np

import fast_lib as F


class Config:
    """All tunables in one place. Defaults are the values validated on 1502 frames."""

    def __init__(self, coarse_points=1500, fine_points=4000, topk=6, prune=True,
                 min_scan_range=1.0, max_scan_range=60.0, voxel=0.3, workers=4,
                 crop_radius=60.0, reference_count=3):
        self.coarse_points = coarse_points
        self.fine_points = fine_points
        self.topk = topk
        self.prune = prune
        self.min_scan_range = min_scan_range
        self.max_scan_range = max_scan_range
        self.voxel = voxel
        self.workers = workers
        self.crop_radius = crop_radius
        self.reference_count = reference_count


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


def prepare_scan(record, cfg):
    """Decode one scan, keep a usable range annulus, voxel downsample."""
    points = F.decode(record['scan_path'])
    radius = np.linalg.norm(points, axis=1)
    points = points[(radius > cfg.min_scan_range) & (radius < cfg.max_scan_range)]
    return F.voxel(points, cfg.voxel)


def localize(record, initial_pose, maps, cfg=None):
    """Run the full chain for one frame.

    Returns (pose, diagnostics). `pose` is a 4x4 in world coordinates with the
    initial rotation preserved and the translation corrected.
    """
    cfg = cfg or Config()
    centre = np.asarray(initial_pose[:3, 3], dtype=np.float64)

    # stages 1-2: reference triple and local map (cached per triple)
    triple = maps.nearby(centre)[:cfg.reference_count]
    map_points = maps.local(centre, triple)

    # stage 3: scan
    scan = prepare_scan(record, cfg)

    # stage 4: direction
    u = along_track_direction(initial_pose)
    lateral = np.cross(u, [0.0, 0.0, 1.0])
    lateral /= np.linalg.norm(lateral)

    # stages 5-7: coarse grid, candidate selection, fine grid
    s, d, cost, evals = F.solve(
        map_points, scan, np.asarray(initial_pose, dtype=np.float64), u,
        workers=cfg.workers, n_coarse=cfg.coarse_points, n_fine=cfg.fine_points,
        topk=cfg.topk, prune=cfg.prune)

    pose = np.asarray(initial_pose, dtype=np.float64).copy()
    pose[:3, 3] = centre + s * u + d * lateral

    diagnostics = {
        's': s, 'd': d, 'cost': cost, 'evals': evals,
        'triple': triple, 'map_points': len(map_points), 'scan_points': len(scan),
        'along_track': u, 'lateral': lateral,
    }
    return pose, diagnostics


def localize_many(records, initial_poses, maps, cfg=None, refiner=None, pool=None):
    """Chain over many frames.

    `pool` is a multiprocessing.Pool; when given, the per-frame work is spread
    over it and `maps` must therefore be built inside each worker (see
    `worker_init`). `refiner(pose, record) -> pose` is applied last if given.
    """
    cfg = cfg or Config()

    def one(item):
        record, initial = item
        pose, diag = localize(record, initial, maps, cfg)
        if refiner is not None:
            pose = refiner(pose, record)
        diag['key'] = record['key']
        return pose, diag

    items = list(zip(records, initial_poses))
    if pool is None:
        return [one(it) for it in items]
    return pool.map(one, items)


def worker_init(module_globals, refs, positions, cfg):
    """Build a MapSource inside a worker process.

    Usage:
        pool = multiprocessing.Pool(n, initializer=worker_init,
                                    initargs=(globals(), refs, positions, cfg))
    """
    module_globals['MAPS'] = F.MapSource(refs, positions, radius=cfg.crop_radius)
