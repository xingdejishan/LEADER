import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn
from torchvision.models import resnet18

from repair import EndpointRepair, repair_supervision
from tokens_geometry import BalancedTokens, raw_candidates

REFERENCE_ROOT = Path('/home/zhangcy/projects/leader-vera-main-20261002/cam5-support-fov-v1')
sys.path.insert(0, str(REFERENCE_ROOT))
from common import PLAN, STORE, guard, verify_source
from fov_visibility_v2 import project, validity

spec = importlib.util.spec_from_file_location('vera_repair_reference_runtime', REFERENCE_ROOT / 'runtime.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
Dataset = reference.Dataset


class RepairSystem(nn.Module):
    def __init__(self, device):
        super().__init__()
        verify_source()
        self.base = reference.System('leader', device).eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        visual = resnet18(weights=None)
        visual.load_state_dict(torch.load(STORE / 'resnet18-f37072fd.pth', map_location='cpu'), strict=True)
        self.tokens = BalancedTokens(nn.Sequential(visual.conv1, visual.bn1, visual.relu, visual.maxpool, visual.layer1))
        self.repair = EndpointRepair()
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        for module in self.tokens.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def encode(self, batch, device):
        with torch.no_grad():
            return self.base.encode(batch, device)

    def from_encoded(self, features, points, extra):
        with torch.no_grad():
            base = self.base.leader.decoder(features)
        geometry = raw_candidates(points.detach().cpu().double().numpy(), extra['raw'], extra['visibility'], validity, project, max_shift=self.repair.max_shift_m)
        tokens = self.tokens(extra['images'], extra['sam'], geometry)
        candidates = torch.as_tensor(geometry['candidates'], device=points.device, dtype=points.dtype)
        valid = torch.as_tensor(geometry['valid'], device=points.device)
        result = self.repair(features, points, base, tokens, candidates, valid)
        return result, base, geometry

    def forward(self, batch, extra, device):
        features, points = self.encode(batch, device)
        return self.from_encoded(features, points, extra)

    def loss(self, result, base, original_points, batch, device):
        transform = batch['T'].to(device)[0] @ torch.linalg.inv(batch['T_corr'].to(device)[0])
        rotation = transform[:3, :3]
        translation = transform[:3, 3] - self.base.center
        target = result.anchors @ rotation.T + translation
        trr, raw = reference.trr_loss(target, result.output[:, :3], result.output[:, 3], torch.zeros(len(target), device=device, dtype=torch.long))
        auxiliary, evidence = repair_supervision(result, base, original_points, rotation, translation)
        return trr + auxiliary, {'trr': trr, 'raw': raw, **evidence}

    def pose(self, result, batch, seed, device):
        selected = result.selected
        source = result.anchors[selected].float()
        target = result.output[selected, :3].float()
        if len(source) < 3:
            raise ValueError('Insufficient selected correspondences')
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        initial = reference.make_matcher(device).estimator(source[None], target[None])[0]
        pose = reference.full_pool_refine(initial, source, target)
        pose[:3, 3] += self.base.center
        pose = pose @ batch['T_corr'].to(device)[0]
        if not torch.isfinite(pose).all():
            raise ValueError('Nonfinite pose')
        return pose, len(selected)
