import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, source, target, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(source, target, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(target, target, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(target)
        self.bn2 = nn.BatchNorm2d(target)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None if stride == 1 else nn.Sequential(nn.Conv2d(source, target, 1, stride, bias=False), nn.BatchNorm2d(target))

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + (x if self.downsample is None else self.downsample(x)))


class LoFTRLocal(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(BasicBlock(128, 128), BasicBlock(128, 128))
        self.layer2 = nn.Sequential(BasicBlock(128, 196, 2), BasicBlock(196, 196))
        self.layer3 = nn.Sequential(BasicBlock(196, 256, 2), BasicBlock(256, 256))

    def stem(self, x):
        return self.layer2(self.layer1(self.relu(self.bn1(self.conv1(x)))))

    def forward(self, x):
        return self.layer3(self.stem(x))

    def load_pretrained(self, path):
        state = torch.load(path, map_location='cpu')['state_dict']
        own = self.state_dict()
        state = {k.removeprefix('backbone.'): v for k, v in state.items() if k.startswith('backbone.') and k.removeprefix('backbone.') in own}
        self.load_state_dict(state, strict=True)
        return self


class LPGRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.lidar_norm = nn.LayerNorm(512)
        self.image_norm = nn.LayerNorm(128)
        self.lidar_gate_proj = nn.Linear(512, 64)
        self.image_gate_proj = nn.Linear(128, 64)
        self.gate = nn.Sequential(nn.Linear(130, 64), nn.LeakyReLU(.01), nn.Linear(64, 1), nn.Sigmoid())
        self.image_to_delta = nn.Linear(128, 512, bias=False)
        self.alpha = nn.Parameter(torch.tensor(.1))

    def forward(self, lidar, image, valid, distance):
        image = self.image_norm(torch.where(valid[:, None], image, torch.zeros_like(image)))
        ql = F.leaky_relu(self.lidar_gate_proj(self.lidar_norm(lidar)), .01)
        qi = F.leaky_relu(self.image_gate_proj(image), .01)
        gate = self.gate(torch.cat([ql, qi, (distance[:, None]/100).clamp(0, 1), valid[:, None].float()], -1))
        delta = self.alpha * gate * self.image_to_delta(image)
        return lidar + torch.where(valid[:, None], delta, torch.zeros_like(delta))


class Distillation(nn.Module):
    def __init__(self):
        super().__init__()
        self.lidar_proj = nn.Linear(512, 64)
        self.image_proj = nn.Linear(128, 64)

    def forward(self, lidar, image, valid, reliable):
        image = torch.where(valid[:, None], image, torch.zeros_like(image))
        hl = F.normalize(self.lidar_proj(lidar.detach()), dim=-1)
        hi = F.normalize(self.image_proj(image), dim=-1)
        weight = (valid & reliable).float()
        return ((1 - (hl * hi).sum(-1))*weight).sum() / weight.sum().clamp_min(1)
