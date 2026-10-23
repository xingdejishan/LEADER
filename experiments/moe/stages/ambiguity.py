"""Along-track ambiguity of a pose, measured on the map the search already built.

The quantity is the same as the frozen gate: a competing position five to ten metres
away along the track must not explain the scan almost as well as the current pose, i.e.

    rho = (C_far - C_0) / C_0

with C the mean truncated nearest-neighbour distance. Two things differ from the
frozen implementation, both of them deliberate:

  * the local map and its KD-tree are the ones the along-track search built, so no
    frame pays for a second map;
  * the query set is an even stride over the prepared scan instead of the first N
    points. The scan is in acquisition order, so a prefix covers only part of the
    sweep; striding removes that bias at no cost (measured: agreement with the
    frozen gate 98.1% -> 99.7%).

Point count is what the statistic is sensitive to, not the along-track step: 4000
points at 1 m step and at 0.5 m step give the same 84 flags and 99.7% agreement,
while 2000 points over-admit (138 flags, agreement 94.1%). Keep N_POINTS at 4000.
"""
import numpy as np

GRID = np.round(np.arange(-8.0, 8.0001, 1.0), 3)   # 17 along-track positions
FAR_MIN = 5.0        # a "competing" offset is |s| > this
TRUNC = 1.0          # per-point cost cap, metres
THRESHOLD = 0.21     # ratio below this -> severely ambiguous
N_POINTS = 4000      # query points (see the module docstring: do not lower)


def sample(points, n=N_POINTS):
    """Even stride over the scan. A prefix is biased: the scan is in sweep order."""
    n = min(int(n), len(points))
    if n >= len(points):
        return points
    return points[np.linspace(0, len(points) - 1, n).astype(np.int64)]


def ratio(tree, points, pose, direction, grid=GRID, far_min=FAR_MIN, trunc=TRUNC,
          n_points=N_POINTS):
    """Return (rho, c0, cfar). `tree` and `points` are the search's map and scan."""
    q = sample(points, n_points) @ pose[:3, :3].T
    origin = pose[:3, 3]
    probe = np.empty((len(grid) * len(q), 3))
    for a, s in enumerate(grid):
        probe[a * len(q):(a + 1) * len(q)] = q + (origin + s * direction)
    d = tree.query(probe, k=1, workers=1, distance_upper_bound=trunc)[0]
    costs = np.where(np.isfinite(d), d, trunc).reshape(len(grid), -1).mean(1)
    c0 = float(costs.min())
    cfar = float(costs[np.abs(grid) > far_min].min())
    return (cfar - c0) / max(c0, 1e-6), c0, cfar


def decide(tree, points, pose, direction, threshold=THRESHOLD, **kw):
    """Return (severely_ambiguous, record)."""
    rho, c0, cfar = ratio(tree, points, pose, direction, **kw)
    return bool(np.isfinite(rho) and rho < threshold), dict(ratio=rho, c0=c0, cfar=cfar)
