"""GPU truncated distance field + batched continuous 2-D refinement.

Cost  : D_trunc(x) = min(nearest_map_point_distance, 1.0)
Grad  : from the *untruncated* EDT, because the truncated field is flat (zero
        gradient) wherever the true distance exceeds the truncation, which
        stalls any gradient-based optimiser.

Measured outcome (see docs/DISTANCE_FIELD.md): the field is accurate enough to
rank coarse candidates (top-6 agreement 5-6 of 6) but NOT accurate enough to
replace the exact fine-grid enumeration.
"""
import numpy as np
import torch
import torch.nn.functional as TF
from scipy import ndimage


def pick_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    best, bestfree = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > bestfree:
            best, bestfree = i, free
    return torch.device('cuda:%d' % best)


class Field:
    """Truncated distance field over an axis-aligned box."""

    def __init__(self, points, voxel=0.4, trunc=1.0, dev=None):
        self.trunc = float(trunc)
        self.voxel = float(voxel)
        self.dev = dev or pick_device()
        lo = points.min(0) - (trunc + voxel)
        hi = points.max(0) + (trunc + voxel)
        dims = np.maximum(np.ceil((hi - lo) / voxel).astype(np.int64), 4)
        self.dims = dims
        self.lo = torch.from_numpy(lo).to(self.dev).float()
        occ = np.zeros(tuple(dims), dtype=bool)
        ijk = np.clip(np.floor((points - lo) / voxel).astype(np.int64), 0, dims - 1)
        occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
        full = (ndimage.distance_transform_edt(~occ).astype(np.float32) * np.float32(voxel))
        trunc_field = np.minimum(full, np.float32(trunc))
        gx, gy, gz = np.gradient(full, voxel, edge_order=1)
        self.trunc_t = torch.from_numpy(trunc_field).to(self.dev).view(1, 1, *dims)
        self.grad_t = torch.stack([torch.from_numpy(g.astype(np.float32))
                                   for g in (gx, gy, gz)]).to(self.dev).view(1, 3, *dims)
        self._scale = torch.tensor([2.0 / (dims[0] - 1), 2.0 / (dims[1] - 1),
                                    2.0 / (dims[2] - 1)], device=self.dev).float()

    def sample(self, pts):
        """pts: (M,3) float32 tensor on device -> (values (M,), grads (M,3))."""
        idx = (pts - self.lo) / self.voxel
        norm = idx * self._scale - 1.0
        hi = torch.tensor([d - 1.0 for d in self.dims], device=self.dev).float()
        inside = ((idx >= 0) & (idx <= hi)).all(dim=1)
        # grid_sample maps grid[..., 0] -> the LAST spatial dimension, so the
        # coordinate triple has to be handed over in (z, y, x) order. Getting
        # this wrong silently samples a transposed location and destroys the
        # ranking while looking perfectly fine numerically.
        grid = norm.flip(-1).contiguous().view(1, 1, 1, -1, 3)
        val = TF.grid_sample(self.trunc_t, grid, mode='bilinear',
                             padding_mode='border', align_corners=True).view(-1)
        val = torch.where(inside, val, torch.full_like(val, self.trunc))
        grad = TF.grid_sample(self.grad_t, grid, mode='bilinear',
                              padding_mode='border', align_corners=True).view(3, -1).T
        grad = torch.where(inside.unsqueeze(1), grad, torch.zeros_like(grad))
        return val, grad

    def cost_np(self, pts_np):
        pts = torch.from_numpy(np.ascontiguousarray(pts_np)).to(self.dev).float()
        v, _ = self.sample(pts)
        return float(v.mean().item())


def refine_batch(field, base, origin, u, n, s0, d0, iters=6, bound_s=0.5, bound_d=0.4):
    """Batched damped Gauss-Newton on (s, d) for K candidates at once.

    base  : (P,3) tensor, scan points already rotated into the world frame
    s0,d0 : (K,) starting offsets
    returns (s, d, cost), each (K,) on device
    """
    dev = field.dev
    K, P = s0.shape[0], base.shape[0]
    u = u.view(1, 1, 3)
    n = n.view(1, 1, 3)
    o = origin.view(1, 1, 3)
    base_e = base.view(1, P, 3)

    def points(s, d):
        return (base_e + o + s.view(K, 1, 1) * u + d.view(K, 1, 1) * n).reshape(-1, 3)

    def evaluate(s, d):
        val, grad = field.sample(points(s, d))
        val = val.view(K, P)
        grad = grad.view(K, P, 3)
        energy = val.mean(dim=1)
        gs = (grad * u).sum(dim=2).mean(dim=1)
        gd = (grad * n).sum(dim=2).mean(dim=1)
        return energy, gs, gd

    s, d = s0.clone(), d0.clone()
    energy, gs, gd = evaluate(s, d)
    h = 0.05
    eye = torch.eye(2, device=dev).float()
    for _ in range(iters):
        _, gs_s, gd_s = evaluate(s + h, d)
        _, gs_m, gd_m = evaluate(s - h, d)
        _, gs_d, gd_d = evaluate(s, d + h)
        _, gs_d0, gd_d0 = evaluate(s, d - h)
        H = torch.stack([torch.stack([(gs_s - gs_m) / (2 * h), (gs_d - gs_d0) / (2 * h)], dim=1),
                         torch.stack([(gd_s - gd_m) / (2 * h), (gd_d - gd_d0) / (2 * h)], dim=1)],
                        dim=1)
        g = torch.stack([gs, gd], dim=1)
        lam = (1e-3 * (H[:, 0, 0] + H[:, 1, 1]).abs() + 1e-6).view(K, 1, 1)
        try:
            step = -torch.linalg.solve(H + lam * eye, g.unsqueeze(2)).squeeze(2)
        except Exception:
            break
        step[:, 0] = step[:, 0].clamp(-bound_s, bound_s)
        step[:, 1] = step[:, 1].clamp(-bound_d, bound_d)
        improved = torch.zeros(K, dtype=torch.bool, device=dev)
        cur_s, cur_d, cur_e = s.clone(), d.clone(), energy.clone()
        for t in (1.0, 0.5, 0.25):
            trial_s = (s + t * step[:, 0]).clamp(s0 - bound_s, s0 + bound_s)
            trial_d = (d + t * step[:, 1]).clamp(d0 - bound_d, d0 + bound_d)
            e_new, gs_new, gd_new = evaluate(trial_s, trial_d)
            take = (~improved) & (e_new < cur_e)
            cur_s = torch.where(take, trial_s, cur_s)
            cur_d = torch.where(take, trial_d, cur_d)
            cur_e = torch.where(take, e_new, cur_e)
            gs = torch.where(take, gs_new, gs)
            gd = torch.where(take, gd_new, gd)
            improved = improved | take
        if not bool(improved.any()):
            break
        s, d, energy = cur_s, cur_d, cur_e
    return s, d, energy
