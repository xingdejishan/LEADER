"""Exact geometry backend: share one map neighbourhood across a fine window.

Key property: the cost truncates at 1 m and every pose inside a fine window is
within rho = sqrt(0.5^2 + 0.4^2) = 0.6403 m of the window centre. So any map
point that can contribute a distance below 1 m for *any* pose in the window must
lie within 1 + rho of the centre. Gathering that neighbourhood once replaces 45
separate nearest-neighbour searches with arithmetic on a fixed candidate set.

The arithmetic is closed form. With diff_ij = base_i - m_ij,

    |base_i + shift - m_ij|^2 = A_ij + B_ij . shift + |shift|^2,
    A_ij = |diff_ij|^2,  B_ij = 2 * diff_ij

so all poses can be scored with one matmul instead of one query per pose.
"""
import numpy as np
import torch

RHO = float(np.hypot(0.5, 0.4))          # fine-window radius
SEARCH_R = 1.0 + RHO                      # neighbourhood radius that guarantees completeness
TRUNC = 1.0


def gather_neighbours(tree, base, k=24, workers=4):
    """K nearest map points for each base point, plus a completeness flag.

    A point is 'complete' when its K-th neighbour is at least SEARCH_R away: in
    that case every map point within SEARCH_R is inside the K-set, so the exact
    truncated cost for every pose in the window is computable from it.
    """
    dist, idx = tree.query(base, k=k, workers=workers)
    complete = dist[:, -1] >= SEARCH_R
    return idx, complete


class ExactFine:
    def __init__(self, map_points, base, shifts, k=24, workers=4, dev=None):
        self.dev = dev or (torch.device('cuda:%d' % 0) if torch.cuda.is_available()
                           else torch.device('cpu'))
        self.tree = None
        self.map_points = map_points
        from scipy.spatial import cKDTree
        self.tree = cKDTree(map_points)
        self.base = np.ascontiguousarray(base, dtype=np.float64)
        self.shifts = np.ascontiguousarray(shifts, dtype=np.float64)
        self.idx, self.complete = gather_neighbours(self.tree, self.base, k=k,
                                                    workers=workers)
        # gather neighbour coordinates; for incomplete points pad with a far point
        neigh = np.asarray(map_points)[self.idx]                     # (P,K,3)
        bad = ~self.complete
        if bad.any():
            far = self.base[bad].mean(0) + 1e4
            neigh[bad] = far
        self.neigh = neigh
        diff = self.base[:, None, :] - neigh                          # (P,K,3)
        A = (diff ** 2).sum(-1)                                       # (P,K)
        B = 2.0 * diff                                                # (P,K,3)
        sh = self.shifts
        d2 = (A[:, :, None]
              + torch.einsum('pkj,mj->pkm',
                             torch.from_numpy(B).to(self.dev).float(),
                             torch.from_numpy(sh).to(self.dev).float())
              + torch.from_numpy((sh ** 2).sum(-1)).to(self.dev).float()[None, None, :])
        dist = torch.sqrt(torch.clamp(d2, min=0.0))
        self.costs = torch.clamp(dist, max=TRUNC).mean(dim=1).cpu().numpy()   # (M,)
        self.n_incomplete = int(bad.sum())


def fine_costs_cpu(tree, base, shifts, k=None):
    """Reference: one KD query per pose (the current implementation)."""
    out = []
    for sh in shifts:
        d = tree.query(base + sh, k=1, workers=4, distance_upper_bound=TRUNC)[0]
        d = np.where(np.isfinite(d), d, TRUNC)
        out.append(float(np.minimum(d, TRUNC).mean()))
    return np.asarray(out)


def fine_costs_cpu_shared(tree, base, shifts):
    """Reference using the same shared neighbourhood, distances by brute force."""
    idx, complete = gather_neighbours(tree, base, k=24)
    neigh = np.asarray(tree.data)[idx]
    bad = ~complete
    if bad.any():
        neigh[bad] = base[bad].mean(0) + 1e4
    out = []
    for sh in shifts:
        diff = (base + sh)[:, None, :] - neigh
        d = np.sqrt((diff ** 2).sum(-1)).min(1)
        out.append(float(np.minimum(d, TRUNC).mean()))
    return np.asarray(out), int(bad.sum())
