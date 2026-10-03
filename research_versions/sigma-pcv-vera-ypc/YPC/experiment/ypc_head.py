import math

import torch
from torch import nn
from torch.nn import functional as F


def rz_degrees(angle):
    a = angle * (math.pi / 180)
    c, s, z = a.cos(), a.sin(), torch.zeros_like(a)
    return torch.stack((c, -s, z, s, c, z, z, z, z + 1), -1).reshape(*angle.shape, 3, 3)


def best_right_yaw(rotation, truth):
    relative = rotation.double().T @ truth.double()
    return torch.atan2(relative[1, 0] - relative[0, 1], relative[0, 0] + relative[1, 1]) * (180 / math.pi)


class YPCHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.SiLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(), nn.Conv2d(32, 32, 3, 1, 1))
        self.image_adapter = nn.Linear(32, 32, bias=False)
        self.field_adapter = nn.Linear(32, 32, bias=False)
        self.weight = nn.Sequential(nn.Linear(35, 32), nn.SiLU(), nn.Linear(32, 1))
        self.raw_tau = nn.Parameter(torch.tensor(math.log(math.expm1(.1))))
        self.register_buffer('grid', torch.linspace(-4, 4, 21))
        nn.init.zeros_(self.stem[-1].weight)
        nn.init.zeros_(self.stem[-1].bias)
        nn.init.eye_(self.image_adapter.weight)
        nn.init.zeros_(self.field_adapter.weight)
        nn.init.zeros_(self.weight[-1].weight)
        nn.init.zeros_(self.weight[-1].bias)

    def descriptor(self, rgb, pixels, observed):
        feature = self.stem(rgb[None])
        center = pixels / 4 + .5
        coords = center * pixels.new_tensor([2 / feature.shape[-1], 2 / feature.shape[-2]]) - 1
        value = F.grid_sample(feature, coords[None, :, None], align_corners=False, padding_mode='border')[0, :, :, 0].T
        return F.normalize(self.image_adapter(observed + value), dim=-1, eps=1e-8)

    def posterior(self, descriptor, field, points, grid):
        adapted = F.normalize(field + self.field_adapter(field), dim=-1, eps=1e-8)
        geometry = torch.stack((points[:, :2].norm(dim=-1) / 100, points[:, 2] / 30, points.norm(dim=-1) / 100), -1)
        weight = self.weight(torch.cat((descriptor, geometry), -1)).squeeze(-1).sigmoid() + .05
        cosine = (adapted * descriptor[:, None]).sum(-1)
        score = (cosine * weight[:, None]).sum(0) / weight.sum()
        logits = score / (.005 + F.softplus(self.raw_tau))
        q = logits.softmax(-1)
        mean = (q * grid).sum()
        variance = (q * (grid - mean).square()).sum() + .05**2 / 12
        return dict(logits=logits, probability=q, mean=mean, variance=variance, weight=weight)
