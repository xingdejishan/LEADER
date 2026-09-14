import math

import torch
from torch import nn


class QueryFusion(nn.Module):
    def __init__(self, gaussian=False):
        super().__init__()
        self.gaussian = gaussian
        self.lidar_norm = nn.LayerNorm(512)
        self.image_norm = nn.LayerNorm(128)
        self.query = nn.Linear(512, 64)
        self.key = nn.Linear(128, 64)
        self.value = nn.Linear(128, 64)
        self.position = nn.Linear(2, 64, bias=False)
        self.output = nn.Linear(64, 512, bias=False)
        self.gate = nn.Linear(512, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.gate.bias, -2.)

    def forward(self, lidar, image, mask, offsets, return_attention=False):
        normalized = self.lidar_norm(lidar)
        image = self.image_norm(torch.where(mask[..., None], image, torch.zeros_like(image)))
        keys = self.key(image) + self.position(offsets)
        logits = (self.query(normalized)[:, None] * keys).sum(-1) / math.sqrt(64)
        if self.gaussian:
            logits = logits - .5 * offsets.square().sum(-1)
        logits = logits.masked_fill(~mask, -1e4)
        attention = logits.softmax(-1) * mask
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        context = (attention[..., None] * self.value(image)).sum(1)
        delta = self.gate(normalized).sigmoid() * self.output(context)
        fused = lidar + torch.where(mask.any(-1, keepdim=True), delta, torch.zeros_like(delta))
        return (fused, attention) if return_attention else fused
