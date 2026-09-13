import math
from dataclasses import dataclass, field
from typing import Type

import torch
import torch.nn.functional as F

from scrstudio.configs.base_config import InstantiateConfig
from scrstudio.model_components.losses import ReproLossConfig, RobustLoss, RobustLossConfig


class FiniteRobustLoss(RobustLoss):
    def __call__(self, loss, step):
        if loss.numel() == 0:
            return loss.sum()
        return super().__call__(loss, step)


@dataclass
class FiniteRobustLossConfig(RobustLossConfig):
    _target: Type = field(default_factory=lambda: FiniteRobustLoss)


def geometry_loss(prediction, batch):
    with torch.amp.autocast('cuda', enabled=False):
        return _geometry_loss(prediction, batch)


def _geometry_loss(prediction, batch):
    valid = batch['geometry_valid'].bool()
    if not valid.any():
        return prediction.sum() * 0
    prediction = prediction[valid].float()
    pose = batch['gt_poses_inv'][valid].float()
    camera = torch.bmm(pose, F.pad(prediction, (0, 1), value=1).unsqueeze(-1)).squeeze(-1)
    target = batch['gt_coords'][valid].float()
    target = torch.bmm(pose, F.pad(target, (0, 1), value=1).unsqueeze(-1)).squeeze(-1)
    rays = torch.bmm(batch['intrinsics_inv'][valid].float(), F.pad(batch['target_px'][valid].float(), (0, 1), value=1).unsqueeze(-1)).squeeze(-1)
    rays = F.normalize(rays, dim=1)
    projection = (camera * rays).sum(1)
    parallel = (projection - (target * rays).sum(1)) / batch['sigma_parallel_m'][valid]
    perpendicular = (camera - projection[:, None] * rays).norm(dim=1) / batch['sigma_perpendicular_m'][valid]
    residual = F.huber_loss(parallel, torch.zeros_like(parallel), reduction='none')
    residual += F.huber_loss(perpendicular, torch.zeros_like(perpendicular), reduction='none')
    quality = batch['geometry_quality'][valid].float()
    return (residual * quality).sum() / quality.sum().clamp_min(1e-8)


@dataclass
class PersistentGeometryLossConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: PersistentGeometryLoss)
    final_reprojection: ReproLossConfig = field(default_factory=ReproLossConfig)
    coarse_reprojection: ReproLossConfig = field(default_factory=ReproLossConfig)
    minimum_weight: float = .25


class PersistentGeometryLoss(torch.nn.Module):
    def __init__(self, config, total_iterations, **kwargs):
        super().__init__()
        self.config = config
        self.total_iterations = total_iterations
        self.final = config.final_reprojection.setup(total_iterations=total_iterations)
        self.coarse = config.coarse_reprojection.setup(total_iterations=total_iterations)

    def forward(self, batch):
        known = batch['geometry_valid'].bool()
        count = len(known)
        zero = batch['sc'].sum() * 0
        total = zero
        metrics = {}
        unknown = ~known
        if unknown.any():
            fallback = {k: v[unknown] if torch.is_tensor(v) and v.ndim and len(v) == count else v for k, v in batch.items() if k != 'metrics'}
            fallback = self.final(fallback)
            fallback = self.coarse(fallback)
            total = total + fallback['metrics']['loss'] * unknown.float().mean()
        if known.any():
            with torch.amp.autocast('cuda', enabled=False):
                camera = torch.bmm(batch['gt_poses_inv'][known].float(), F.pad(batch['sc'][known].float(), (0, 1), value=1).unsqueeze(-1)).squeeze(-1)
                pixel = torch.bmm(batch['intrinsics'][known].float(), camera.unsqueeze(-1)).squeeze(-1)
                error = (pixel[:, :2] / pixel[:, 2:].clamp_min(.1) - batch['target_px'][known]).norm(dim=1)
                total = total + self.final.robust_loss(error, batch['step']) * known.float().mean()
        weight = self.config.minimum_weight + (1 - self.config.minimum_weight) * .5 * (1 + math.cos(math.pi * min(batch['step'] / self.total_iterations, 1)))
        geom = .5 * geometry_loss(batch['sc0'], batch) + geometry_loss(batch['sc'], batch)
        metrics.update(loss=total + weight * geom, geometry_loss=geom, geometry_weight=weight, geometry_fraction=known.float().mean())
        batch['metrics'] = metrics
        return batch
