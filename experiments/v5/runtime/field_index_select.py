import types

import torch
from torch.nn import functional as F


def field_forward(self, points):
    scaled = points[None] / self.spacings[:, None, None]
    base = scaled.floor().long()
    fraction = scaled - base.to(points.dtype)
    corner = base[:, :, None] + self.corners
    index = (corner[..., 0] ^ (corner[..., 1] * 2654435761) ^ (corner[..., 2] * 805459861)) & self.mask
    axis_weight = torch.where(self.corners[None, None].bool(), fraction[:, :, None], 1 - fraction[:, :, None])
    weight = axis_weight[..., 0] * axis_weight[..., 1] * axis_weight[..., 2]
    level = torch.arange(len(self.spacings), device=points.device)[:, None, None]
    flat_index = (level*self.tables.shape[1]+index).reshape(-1)
    gathered = self.tables.reshape(-1, self.tables.shape[-1]).index_select(0, flat_index)
    gathered = gathered.reshape(*index.shape, self.tables.shape[-1])
    features = (gathered * weight[..., None]).sum(2)
    features = features.permute(1, 0, 2).reshape(len(points), -1)
    return F.normalize(self.mlp(features), dim=-1, eps=1e-8)


def enable(model):
    model.field.forward = types.MethodType(field_forward, model.field)
    return model
