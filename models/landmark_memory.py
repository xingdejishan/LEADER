import torch
from torch import nn
from torch.nn import functional as F


class LandmarkMemory(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lidar = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, hidden), nn.GELU())
        self.visual = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, hidden), nn.GELU())
        self.edge = nn.Sequential(nn.Linear(hidden * 4 + 7, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU())
        self.score = nn.Linear(hidden, 1)
        self.output = nn.Linear(hidden + 3, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, lidar, image, predicted, reference_lidar, reference_image,
                reference_world, reference_error, reference_confidence, valid, return_context=False):
        ql, qv = self.lidar(lidar), self.visual(image)
        rl, rv = self.lidar(reference_lidar), self.visual(reference_image)
        relative = (reference_world - predicted[:, None]) / 3.0
        error = reference_error / 3.0
        edge = self.edge(torch.cat((ql[:, None] * rl, (ql[:, None] - rl).abs(),
                                    qv[:, None] * rv, (qv[:, None] - rv).abs(),
                                    relative, error, reference_confidence[..., None]), dim=-1))
        logits = self.score(edge).squeeze(-1).masked_fill(~valid, -1e4)
        attention = logits.softmax(-1) * valid
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        context = (edge * attention[..., None]).sum(1)
        transported = (reference_error * attention[..., None]).sum(1)
        delta = self.output(torch.cat((context, transported), dim=-1)).tanh()
        delta = delta * valid.any(-1, keepdim=True)
        return (delta, context) if return_context else delta
