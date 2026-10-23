"""The frozen gate: along-track ambiguity ratio, single frame, no cross-frame input.

Per frame the gate reads exactly three things:
    current LiDAR scan, current pose (4x4 body->world), static reference library

It answers one question: does this pose have a competing along-track position,
5-10 m away, that explains the scan almost as well?  If the ratio between the
best far-offset cost and the best cost is below THRESHOLD, the frame is
escalated to the independent visual route.

Measured on the frozen sets (see replay_check.py):
    hard1502  84 flagged, 11/12 of the >8 m frames, 0.92% false alarm on <1 m
    ordinary  0 of 1000 flagged, minimum ratio 0.266
"""
import json

import numpy as np
from scipy.spatial import cKDTree

import config as C

__import__('sys').path.insert(
    0, '/home/zhangcy/projects/leader-far-covolume-standard-20261001/source-v1')
from data.robotcar_sdk.python.velodyne import get_velo  # noqa: E402


def downsample(points, size, cap=None, seed=C.QSEED):
    if len(points) == 0:
        return points
    cells = np.floor(points / size).astype(np.int64)
    _, ids = np.unique(cells, axis=0, return_index=True)
    out = points[np.sort(ids)]
    if cap is not None and len(out) > cap:
        pick = np.random.RandomState(seed).choice(len(out), cap, replace=False)
        out = out[np.sort(pick)]
    return out


class ReferenceLibrary(object):
    """Static reference scans, posed in world coordinates. Loaded once, shared."""

    def __init__(self, sel_path=C.SEL):
        sel = json.load(open(sel_path))
        self.refs = sel['references']
        pos = np.stack([np.asarray(r['world_from_body'], float) for r in self.refs])[:, :3, 3]
        self.tree = cKDTree(pos)
        self._cache = {}

    def scan(self, path):
        if path not in self._cache:
            pts = np.asarray(get_velo(str(path))[0], dtype=np.float64)
            r = np.linalg.norm(pts, axis=1)
            self._cache[path] = downsample(pts[(r > 1.0) & (r < 100.0)], C.VOXEL)
            while len(self._cache) > 60:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[path]


def ambiguity_ratio(lib, scan_path, pose):
    """Return dict(ratio, c0, cfar, nmap, nquery). ratio is nan when undecidable."""
    pose = np.asarray(pose, float)
    t0 = pose[:3, 3]
    R0 = pose[:3, :3]
    dist, ids = lib.tree.query(t0, k=64)
    ids = np.atleast_1d(ids)[np.argsort(dist)[:C.N_REF]]
    parts = []
    for j in ids:
        row = lib.refs[int(j)]
        T = np.asarray(row['world_from_body'], float)
        W = lib.scan(row['scan_path']) @ T[:3, :3].T + T[:3, 3]
        parts.append(W[np.linalg.norm(W - t0, axis=1) <= C.KEEP_RADIUS])
    world = downsample(np.concatenate(parts), C.VOXEL)

    q = lib.scan(scan_path)
    r = np.linalg.norm(q, axis=1)
    q = downsample(q[(r > 1.0) & (r < 60.0)], C.VOXEL, cap=C.QCAP)
    if len(world) == 0 or len(q) == 0:
        return dict(ratio=np.nan, c0=np.nan, cfar=np.nan, nmap=len(world), nquery=len(q))

    tree = cKDTree(world)
    Qw = q @ R0.T + t0
    u = R0[:, 0] / np.linalg.norm(R0[:, 0])      # the pose's own forward axis
    cost = np.empty(len(C.S_GRID))
    for k, s in enumerate(C.S_GRID):
        d, _ = tree.query(Qw + s * u, k=1)
        cost[k] = np.minimum(d, C.TRUNC).mean()
    c0 = float(cost.min())
    cfar = float(cost[np.abs(C.S_GRID) > C.FAR_MIN].min())
    return dict(ratio=(cfar - c0) / max(c0, 1e-6), c0=c0, cfar=cfar,
                nmap=len(world), nquery=len(q))


def decide(lib, scan_path, pose, threshold=None):
    """Gate decision for one frame. Returns (escalate, record)."""
    thr = C.THRESHOLD if threshold is None else threshold
    rec = ambiguity_ratio(lib, scan_path, pose)
    rec['threshold'] = thr
    rec['escalate'] = bool(np.isfinite(rec['ratio']) and rec['ratio'] < thr)
    return rec['escalate'], rec
