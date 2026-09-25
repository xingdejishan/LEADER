import torch
from torch import nn


@torch.no_grad()
def neighbor_edges(positions, neighbors=16):
    count = len(positions)
    if count < 2:
        return torch.empty((0, 2), device=positions.device, dtype=torch.long)
    chunks = []
    for start in range(0, count, 256):
        stop = min(start+256, count)
        distances = torch.cdist(positions[start:stop], positions)
        rows = torch.arange(start, stop, device=positions.device)
        distances[torch.arange(stop-start, device=positions.device), rows] = torch.inf
        targets = distances.topk(min(neighbors, count-1), largest=False).indices
        origins = rows[:, None].expand_as(targets)
        lower, upper = torch.minimum(origins, targets), torch.maximum(origins, targets)
        chunks.append((lower*count+upper).flatten())
    keys = torch.unique(torch.cat(chunks), sorted=True)
    return torch.stack((keys//count, keys%count), dim=-1)


class VisualRelationTransport(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.visual_norm = nn.LayerNorm(channels)
        self.edge = nn.Sequential(nn.Linear(channels*3+1, channels), nn.GELU(),
                                  nn.Linear(channels, channels), nn.Tanh())

    def forward(self, geometry, visual, positions):
        edges = neighbor_edges(positions)
        if not len(edges):
            return geometry*0
        left, right = edges.unbind(-1)
        visual = self.visual_norm(visual)
        difference = geometry[right]-geometry[left]
        distance = (positions[right]-positions[left]).norm(dim=-1, keepdim=True).log1p()
        relation = torch.cat((difference.abs(), (visual[right]-visual[left]).abs(),
                              visual[right]*visual[left], distance), dim=-1)
        degree = torch.bincount(edges.flatten(), minlength=len(geometry)).to(geometry.dtype)
        normalization = torch.maximum(degree[left], degree[right])[:, None]
        flux = self.edge(relation)*difference/normalization
        output = torch.zeros_like(geometry)
        output.index_add_(0, left, flux)
        output.index_add_(0, right, -flux)
        return output
