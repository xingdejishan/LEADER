"""Hard visibility filtering + sparse z-buffer occlusion (spec sections 4-5).

Rules:
  - Hard reject only for: behind camera, outside image, black-border pixel,
    explicit nearer-surface occlusion.
  - Empty depth buffer is NOT a rejection: record `depth_evidence_valid`.
"""
import numpy as np


def black_border_mask(image, thresh=10):
    """Per-pixel True where image is NOT border-black.

    Border-black = ALL three channels near zero -> equivalent to
    max(channel) <= thresh. Using min() would also reject saturated
    single-channel pixels like [200,5,5].
    """
    return image.max(axis=-1) > thresh


class VisibilityChecker:
    def __init__(self, config):
        vis = config["visibility"]
        self.cell = vis["zbuffer_cell_px"]
        self.kernel = vis["zbuffer_kernel"]
        self.tau_base = vis["tau_occ_base_m"]
        self.tau_slope = vis["tau_occ_slope"]
        self.min_depth = vis["min_depth"]
        self.max_depth = vis["max_depth"]
        self.width = config["image"]["width"]
        self.height = config["image"]["height"]

    def build_zbuffer(self, uv, depth, valid, gw=None, gh=None):
        """Sparse z-buffer: min depth per cell. Returns (grid, gw, gh)."""
        gw = gw or (self.width + self.cell - 1) // self.cell
        gh = gh or (self.height + self.cell - 1) // self.cell
        grid = np.full((gh, gw), np.inf)
        cu = np.clip((uv[:, 0] // self.cell).astype(np.int64), 0, gw - 1)
        cv = np.clip((uv[:, 1] // self.cell).astype(np.int64), 0, gh - 1)
        sel = valid & np.isfinite(depth)
        np.minimum.at(grid, (cv[sel], cu[sel]), depth[sel])
        return grid, gw, gh

    def occlusion_query(self, grid, gw, gh, uv, depth, valid):
        """For each (point, cam): occluded / depth_margin / evidence_valid.

        Returns (occluded, depth_margin, evidence_valid) of shape (N,).
        Only points with `valid` True are queried; others get False/0/False.
        """
        n = uv.shape[0]
        occluded = np.zeros(n, dtype=bool)
        margin = np.zeros(n)
        evidence = np.zeros(n, dtype=bool)
        sel = np.where(valid & np.isfinite(depth))[0]
        if sel.size == 0:
            return occluded, margin, evidence
        cu = np.clip((uv[sel, 0] // self.cell).astype(np.int64), 0, gw - 1)
        cv = np.clip((uv[sel, 1] // self.cell).astype(np.int64), 0, gh - 1)
        k = self.kernel // 2
        zmin = np.full(sel.shape, np.inf)
        for dv in range(-k, k + 1):
            for du in range(-k, k + 1):
                uu = np.clip(cu + du, 0, gw - 1)
                vv = np.clip(cv + dv, 0, gh - 1)
                zmin = np.minimum(zmin, grid[vv, uu])
        with np.errstate(invalid="ignore"):
            has_ev = np.isfinite(zmin)
            dmargin = np.where(has_ev, depth[sel] - zmin, 0.0)
            tau = np.maximum(self.tau_base, self.tau_slope * depth[sel])
            occ = has_ev & (dmargin > tau)
        occluded[sel] = occ
        margin[sel] = dmargin
        evidence[sel] = has_ev
        return occluded, margin, evidence


