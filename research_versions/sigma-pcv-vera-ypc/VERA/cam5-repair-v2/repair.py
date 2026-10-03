import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def fourier(delta):
    frequency = 2. ** torch.arange(8, device=delta.device, dtype=delta.dtype)
    angle = delta[..., None] * frequency * math.pi
    return torch.cat((angle.sin(), angle.cos()), -1).flatten(-2)


def determinant3(matrix):
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def baseline_rotation(points, prediction):
    n = len(points)
    k = max(min(50, n), n // 2)
    index = prediction[:, 3].topk(k).indices
    identity = torch.eye(3, device=points.device, dtype=points.dtype)
    if k < 3:
        return identity, False
    with torch.no_grad():
        x = points[index]
        y = prediction[index, :3]
        weights = torch.ones(k, device=x.device, dtype=x.dtype)
        rotation = identity
        for threshold in (float('inf'), 1.2, .6):
            if int((weights > 0).sum()) < 6:
                return identity, False
            weights = weights / weights.sum().clamp_min(1e-12)
            cx = (x * weights[:, None]).sum(0)
            cy = (y * weights[:, None]).sum(0)
            covariance = (x - cx).T @ ((y - cy) * weights[:, None])
            u, singular, vh = torch.linalg.svd(covariance)
            if not torch.isfinite(singular).all() or singular[1] <= singular[0].clamp_min(1e-12) * 1e-6:
                return identity, False
            correction = identity.clone()
            correction[-1, -1] = torch.where(determinant3(vh.T @ u.T) < 0, -1., 1.)
            rotation = vh.T @ correction @ u.T
            translation = cy - rotation @ cx
            residual = (x @ rotation.T + translation - y).norm(dim=-1)
            if math.isfinite(threshold):
                weights = (1 - (residual / threshold).square()).clamp_min(0).square()
        return rotation, bool(torch.isfinite(rotation).all())


def protected_pool(base, anchors, output, admitted, base_error, new_error, margin=.05, swap_fraction=.25):
    n = len(base)
    k = max(min(50, n), n // 2)
    original = base[:, 3].topk(k).indices
    core = original[:max(1, k // 2)]
    protected = torch.zeros(n, dtype=torch.bool, device=base.device)
    protected[core] = True
    if not admitted.any():
        return original, protected, 0
    with torch.no_grad():
        original_mask = torch.zeros_like(protected)
        original_mask[original] = True
        outsiders = torch.where(admitted & ~original_mask)[0]
        effective = torch.where(admitted, new_error, base_error)
        tail = original[~protected[original]]
        tail_order = np.lexsort((tail.cpu().numpy(), -effective[tail].cpu().double().numpy()))
        outsider_order = np.lexsort((outsiders.cpu().numpy(), effective[outsiders].cpu().double().numpy()))
        tail = tail[torch.as_tensor(tail_order, device=tail.device)]
        outsiders = outsiders[torch.as_tensor(outsider_order, device=outsiders.device)]
        selected = original.clone()
        swaps = 0
        limit = min(len(tail), len(outsiders), int(k * swap_fraction))
        for number in range(limit):
            old, new = tail[number], outsiders[number]
            if effective[new] + margin >= effective[old]:
                break
            selected[selected == old] = new
            swaps += 1
        assert len(selected.unique()) == k and (core[:, None] == selected[None]).any(-1).all()
        assert not admitted[core].any()
    return selected, protected, swaps


@dataclass
class RepairResult:
    output: torch.Tensor
    anchors: torch.Tensor
    selected: torch.Tensor
    admitted: torch.Tensor
    protected: torch.Tensor
    candidate_prediction: torch.Tensor
    candidates: torch.Tensor
    valid: torch.Tensor
    attention: torch.Tensor
    base_error: torch.Tensor
    candidate_error: torch.Tensor
    rotation: torch.Tensor
    rotation_valid: bool
    swaps: int


class EndpointRepair(nn.Module):
    def __init__(self, temperature=.2, position_bound=.1, max_shift_m=1.5, residual_bound_m=.5, margin_m=.05, chunk=128):
        super().__init__()
        self.temperature = float(temperature)
        self.position_bound = float(position_bound)
        self.max_shift_m = float(max_shift_m)
        self.residual_bound_m = float(residual_bound_m)
        self.margin_m = float(margin_m)
        self.chunk = int(chunk)
        if self.temperature <= 0 or self.max_shift_m <= 0 or self.chunk <= 0:
            raise ValueError('Invalid repair bounds')
        self.feature = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 64), nn.GELU())
        self.query = nn.Linear(64, 64, bias=False)
        self.key = nn.Linear(64, 64, bias=False)
        self.position = nn.Linear(48, 1, bias=False)
        self.residual = nn.Sequential(nn.Linear(176, 64), nn.GELU(), nn.Linear(64, 3))
        self.base_quality = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.quality_delta = nn.Sequential(nn.Linear(176, 64), nn.GELU(), nn.Linear(64, 1))
        for layer in (self.residual[-1], self.base_quality[-1], self.quality_delta[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features, points, base, tokens, candidates, valid, rotation=None, rotation_valid=None):
        n, count = valid.shape
        if n < 1 or count < 1:
            raise ValueError('Nonempty frame and candidate dimensions required')
        if features.shape != (n, 512) or points.shape != (n, 3) or base.shape != (n, 4):
            raise ValueError('Baseline rows must align')
        if candidates.shape != (n, count, 3) or tokens.shape != (n, count, 64) or valid.dtype != torch.bool:
            raise ValueError('Candidate rows must align')
        if not all(torch.isfinite(x).all() for x in (features, points, base, tokens, candidates)):
            raise ValueError('Nonfinite input')
        if rotation is None:
            rotation, rotation_valid = baseline_rotation(points.detach(), base.detach())
        elif rotation_valid is None:
            raise ValueError('An explicit rotation requires validity')
        delta = candidates - points[:, None]
        valid = valid & (delta.norm(dim=-1) <= self.max_shift_m)
        original = base[:, 3].topk(max(min(50, n), n // 2)).indices
        protected = torch.zeros(n, device=points.device, dtype=torch.bool)
        protected[original[:max(1, len(original) // 2)]] = True
        editable = valid & ~protected[:, None]
        features64 = self.feature(features.detach())
        base_logit = self.base_quality(features64).squeeze(-1)
        base_error = F.softplus(base_logit) + 1e-4
        all_predictions, all_quality, all_attention = [], [], []
        for start in range(0, n, self.chunk):
            stop = min(start + self.chunk, n)
            local_valid = valid[start:stop]
            local = tokens[start:stop]
            denominator = local_valid.sum(1, keepdim=True).clamp_min(1)
            mean = (local * local_valid[..., None]).sum(1) / denominator
            keys = F.normalize(self.key(local - mean[:, None]), dim=-1, eps=1e-6)
            query = F.normalize(self.query(features64[start:stop]), dim=-1, eps=1e-6)
            encoded_delta = fourier(delta[start:stop])
            scores = torch.einsum('nd,nmd->nm', query, keys) / self.temperature
            scores = scores + self.position_bound * self.position(encoded_delta).squeeze(-1).tanh()
            scores = scores.masked_fill(~local_valid, -1e4)
            attention = scores.softmax(-1) * local_valid
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
            joint = torch.cat((features64[start:stop, None].expand(-1, count, -1), local, encoded_delta), -1)
            residual = self.residual_bound_m / math.sqrt(3) * self.residual(joint).tanh()
            transported = base[start:stop, None, :3].detach() + delta[start:stop] @ rotation.detach().T
            prediction = transported + residual
            quality = F.softplus(base_logit[start:stop, None] + self.quality_delta(joint).squeeze(-1)) + 1e-4
            all_predictions.append(prediction)
            all_quality.append(quality)
            all_attention.append(attention)
        prediction = torch.cat(all_predictions)
        quality = torch.cat(all_quality)
        attention = torch.cat(all_attention)
        benefit = (base_error[:, None] - quality - self.margin_m).clamp_min(0)
        utility = attention * benefit * editable
        best = utility.argmax(-1)
        index = torch.arange(n, device=points.device)
        admitted = (utility[index, best] > 0) & bool(rotation_valid)
        anchor = torch.where(admitted[:, None], candidates[index, best], points)
        xyz = torch.where(admitted[:, None], prediction[index, best], base[:, :3])
        output = torch.cat((xyz, base[:, 3:4]), -1)
        chosen_quality = quality[index, best]
        selected, protected, swaps = protected_pool(base, anchor, output, admitted, base_error.detach(), chosen_quality.detach(), self.margin_m)
        return RepairResult(output, anchor, selected, admitted, protected, prediction, candidates, editable, attention, base_error, quality, rotation, bool(rotation_valid), swaps)


def repair_supervision(result, base, points, truth_rotation, truth_translation, margin=.05):
    target_base = points @ truth_rotation.T + truth_translation
    target_candidate = result.candidates @ truth_rotation.T + truth_translation
    base_raw = (base[:, :3].detach() - target_base).norm(dim=-1)
    candidate_raw = (result.candidate_prediction - target_candidate).norm(dim=-1)
    zero = result.candidate_prediction.sum() * 0
    fit = candidate_raw[result.valid].mean() if result.valid.any() else zero
    quality = F.smooth_l1_loss(result.base_error.log1p(), base_raw.detach().log1p())
    if result.valid.any():
        quality = quality + F.smooth_l1_loss(result.candidate_error[result.valid].log1p(), candidate_raw.detach()[result.valid].log1p())
    benefit = (base_raw[:, None] - candidate_raw.detach() - margin).clamp_min(0) * result.valid
    has_benefit = benefit.sum(-1) > 0
    target_attention = benefit / benefit.sum(-1, keepdim=True).clamp_min(1e-12)
    attention = -(target_attention * result.attention.clamp_min(1e-12).log()).sum(-1)
    attention = attention[has_benefit].mean() if has_benefit.any() else zero
    auxiliary = .1 * fit + .1 * quality + .05 * attention
    return auxiliary, {'candidate_fit': fit, 'quality': quality, 'benefit_attention': attention, 'supervised_candidates': int(result.valid.sum()), 'beneficial_points': int(has_benefit.sum())}
