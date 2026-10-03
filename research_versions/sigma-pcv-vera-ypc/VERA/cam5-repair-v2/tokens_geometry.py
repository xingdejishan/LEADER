import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F


def raw_candidates(points, raw, contexts, validity, project, count=49, max_shift=1.5):
    points = np.asarray(points, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    if not len(raw) or not np.isfinite(raw).all() or not np.isfinite(points).all():
        raise ValueError('Finite measured support required')
    distance, index = cKDTree(raw).query(points, k=count, distance_upper_bound=max_shift, workers=1)
    candidates = raw[np.minimum(index, len(raw) - 1)]
    valid = np.isfinite(distance)
    camera_masks = np.stack([valid & validity(candidates, context) for context in contexts], 1)
    camera = camera_masks.sum(-1).argmax(-1)
    mask = camera_masks[np.arange(len(points)), camera]
    pixels = np.zeros((len(points), count, 2), dtype=np.float32)
    for ci, context in enumerate(contexts):
        select = camera == ci
        pixels[select] = project(candidates[select], context['camera'])[0]
    pixels = np.nan_to_num(pixels, nan=0., posinf=0., neginf=0.)
    candidates[~valid] = np.broadcast_to(points[:, None], candidates.shape)[~valid]
    return {'candidates': candidates.astype(np.float32), 'valid': mask, 'camera': camera, 'pixels': pixels}


class BalancedTokens(nn.Module):
    def __init__(self, stem):
        super().__init__()
        self.stem = stem
        self.sam_projection = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 32, bias=False))
        self.rgb_projection = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 32, bias=False))

    def forward(self, images, sam, geometry):
        stem = self.stem(images)
        pixels = torch.as_tensor(geometry['pixels'], device=images.device, dtype=images.dtype)
        camera = torch.as_tensor(geometry['camera'], device=images.device)
        valid = torch.as_tensor(geometry['valid'], device=images.device)
        result = images.new_zeros((*pixels.shape[:-1], 64))
        for ci in range(len(images)):
            selected = torch.where(camera == ci)[0]
            if not len(selected):
                continue
            uv = pixels[selected]
            grid = uv.clone()
            grid[..., 0] = uv[..., 0] * (1024 / 808) / 1024 * 2 - 1
            grid[..., 1] = uv[..., 1] * (781 / 616) / 1024 * 2 - 1
            sam_tokens = F.grid_sample(sam[ci:ci + 1], grid[None], align_corners=False, padding_mode='border')[0].permute(1, 2, 0)
            grid = uv.clone()
            grid[..., 0] = (uv[..., 0] / 4 + .5) / stem.shape[-1] * 2 - 1
            grid[..., 1] = (uv[..., 1] / 4 + .5) / stem.shape[-2] * 2 - 1
            rgb_tokens = F.grid_sample(stem[ci:ci + 1], grid[None], align_corners=False, padding_mode='border')[0].permute(1, 2, 0)
            sam_tokens = F.normalize(self.sam_projection(sam_tokens), dim=-1, eps=1e-6)
            rgb_tokens = F.normalize(self.rgb_projection(rgb_tokens), dim=-1, eps=1e-6)
            result[selected] = torch.cat((sam_tokens, rgb_tokens), -1) * valid[selected, :, None]
        return result
