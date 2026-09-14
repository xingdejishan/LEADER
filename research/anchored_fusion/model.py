import torch
from torch import nn
from torch.nn import functional as F


class AnchoredFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(640, 32), nn.ReLU(), nn.Linear(32, 512))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, lidar, image, editable):
        lidar = lidar.detach()
        image = torch.where(editable[:, None], image.detach(), torch.zeros_like(image))
        residual = self.net(torch.cat([F.layer_norm(lidar, (512,)), F.layer_norm(image, (128,))], dim=-1))
        scale = (.05 * lidar.norm(dim=-1) / residual.norm(dim=-1).clamp_min(1e-12)).clamp(max=1)
        residual = residual * scale[:, None]
        return lidar + torch.where(editable[:, None], residual, torch.zeros_like(residual))


def contrastive(query, anchors, positive, negative):
    logits = (F.normalize(query, dim=-1)[:, None] * anchors.detach()).sum(-1) / .1
    denominator = logits.masked_fill(~(positive | negative), -float('inf')).logsumexp(-1)
    numerator = logits.masked_fill(~positive, -float('inf')).logsumexp(-1)
    return denominator - numerator
