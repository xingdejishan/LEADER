"""Compact visual encoder: RGB -> 64x64x64 code -> INT8 -> adapter -> 64x64x256.

Replaces the frozen SAM ViT-L feature extractor at inference time. The teacher is
the cached SAM feature map; the student only ever sees the RGB image, preprocessed
exactly the way SAM was (resize longest side to 1024, SAM mean/std, zero pad to
1024x1024, patch 16 -> 64x64 grid), so the resulting grid is token-aligned.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

SAM_MEAN = (123.675, 116.28, 103.53)
SAM_STD = (58.395, 57.12, 57.375)
INPUT = 1024
GRID = 64
CODE_CHANNELS = 64
FEATURE_CHANNELS = 256


def sam_preprocess(rgb, resized_hw):
    """rgb: (3,H,W) float in [0,1]; resized_hw: (781,1024) style target size."""
    h, w = int(resized_hw[0]), int(resized_hw[1])
    x = F.interpolate(rgb.unsqueeze(0), size=(h, w), mode="bilinear",
                      align_corners=False, antialias=True)
    x = x * 255.0
    mean = x.new_tensor(SAM_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(SAM_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    pad = x.new_zeros(1, 3, INPUT, INPUT)
    pad[:, :, :h, :w] = x
    return pad


class FPNLite(nn.Module):
    """Two-level lateral fusion at stride 16 with an extra upsample from stride 32."""

    def __init__(self, c16, c32, out=128):
        super().__init__()
        self.lat16 = nn.Conv2d(c16, out, 1)
        self.lat32 = nn.Conv2d(c32, out, 1)
        self.smooth = nn.Sequential(
            nn.Conv2d(out, out, 3, padding=1, groups=out, bias=False),
            nn.BatchNorm2d(out), nn.SiLU(inplace=True),
            nn.Conv2d(out, out, 1, bias=False), nn.BatchNorm2d(out), nn.SiLU(inplace=True),
        )

    def forward(self, f16, f32):
        a = self.lat16(f16)
        b = F.interpolate(self.lat32(f32), size=a.shape[-2:], mode="nearest")
        return self.smooth(a + b)


class CompactEncoder(nn.Module):
    def __init__(self, backbone="small", width=128, code=CODE_CHANNELS,
                 feat=FEATURE_CHANNELS, pretrained=True):
        super().__init__()
        net = mobilenet_v3_small(weights="DEFAULT" if pretrained else None) \
            if backbone == "small" else \
            mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
        layers = list(net.features.children())
        # locate the stride-16 and stride-32 stages by probing
        with torch.no_grad():
            probe = torch.zeros(1, 3, INPUT, INPUT)
            shapes, x = [], probe
            for layer in layers:
                x = layer(x)
                shapes.append(x.shape[1:])
        stride16 = max(i for i, s in enumerate(shapes) if s[1] == GRID)
        stride32 = max(i for i, s in enumerate(shapes) if s[1] == GRID // 2)
        self.stem16 = nn.Sequential(*layers[:stride16 + 1])
        self.stem32 = nn.Sequential(*layers[stride16 + 1:stride32 + 1])
        c16 = shapes[stride16][0]
        c32 = shapes[stride32][0]
        self.fpn = FPNLite(c16, c32, width)
        self.head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
            nn.BatchNorm2d(width), nn.SiLU(inplace=True),
            nn.Conv2d(width, code, 1),
        )
        self.adapter = nn.Conv2d(code, feat, 1)
        nn.init.zeros_(self.adapter.bias)

    def forward(self, x):
        f16 = self.stem16(x)
        f32 = self.stem32(f16)
        code = self.head(self.fpn(f16, f32))
        return code

    def expand(self, code):
        return self.adapter(code)


def quantize_code(code, bits=8):
    """Per-channel symmetric INT8 on the 64-d code; returns (dequantized, scale)."""
    scale = code.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.clamp(torch.round(code / scale), -127, 127)
    return q * scale, scale


class LinearHead(nn.Module):
    """Lightweight 256 -> 64 output head on frozen SAM tokens, task-trained.

    Initialised from the closed-form PCA-64 basis, so training starts at the
    unsupervised solution instead of at random.
    """

    def __init__(self, mean, basis):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).reshape(-1))
        self.weight = nn.Parameter(torch.as_tensor(basis, dtype=torch.float32).clone())

    def forward(self, sam):
        if sam.dim() == 3:
            sam = sam.unsqueeze(0)
        channels = sam.shape[1]
        flat = sam[0].permute(1, 2, 0).reshape(-1, channels)
        code = (flat - self.mean) @ self.weight
        return code.t().reshape(1, self.weight.shape[1], sam.shape[2], sam.shape[3])

    def expand(self, code):
        if code.dim() == 3:
            code = code.unsqueeze(0)
        flat = code[0].permute(1, 2, 0).reshape(-1, code.shape[1])
        out = flat @ self.weight.t() + self.mean
        return out.t().reshape(1, self.weight.shape[0], code.shape[2], code.shape[3])


def expand_with(code, weight, mean):
    """Expand a 64-d code with an external basis: x = code @ weight^T + mean."""
    if code.dim() == 3:
        code = code.unsqueeze(0)
    flat = code[0].permute(1, 2, 0).reshape(-1, code.shape[1])
    out = flat @ weight.t() + mean
    return out.t().reshape(1, weight.shape[0], code.shape[2], code.shape[3])