import math

import torch
from torch import nn

from models.interaction_backbone import GeometryImageExchange, InterleavedInteraction
from models.relation_graph import VisualRelationTransport


class GeometryRelationExchange(GeometryImageExchange):
    def __init__(self, channels):
        super().__init__(channels)
        self.transport = VisualRelationTransport(64)

    def geometry_update(self, features, visual, sparse, offsets, support):
        coordinates = sparse.C.to(device=features.device, dtype=features.dtype)
        stride = features.new_tensor(sparse.tensor_stride)
        polar = coordinates[:, None, 1:]+stride*(offsets+.5)
        angle, radius, height = polar.unbind(-1)
        angle = angle*(2*math.pi/1024)
        points = torch.stack((radius*.2*angle.cos(), radius*.2*angle.sin(), height*.2), dim=-1)
        centers = (points*support[..., None]).sum(1)/support.sum(1).clamp_min(1)[:, None]
        geometry = self.geometry(features)
        delta = torch.zeros_like(geometry)
        for batch in coordinates[:, 0].unique():
            rows = torch.where((coordinates[:, 0] == batch) & support.any(-1))[0]
            if len(rows) > 1:
                delta[rows] = self.transport(geometry[rows], visual[rows], centers[rows])
        return features+self.output(delta)


class RelationalInteraction(InterleavedInteraction):
    def __init__(self):
        super().__init__()
        self.exchanges = nn.ModuleDict({'1': GeometryRelationExchange(64), '3': GeometryRelationExchange(256)})
