import math

import torch
from torch import nn
from torch.nn import functional as F

from models.far_covolume import CoVolumeFAR, jacobian, tukey, update_pose


class CoVolumeStandardFAR(CoVolumeFAR):
    def __init__(self, point_chunk=256, iterations=3, search_spacings=(.4, .1),
                 tukey_thresholds=(1.2, .6), damping=1e-4, geo_sigma_bounds=(.15, .6),
                 geo_sigma_initial=.3, volume_ratio_bounds=(.05, 1.), volume_ratio_initial=.5):
        super().__init__(point_chunk, iterations, search_spacings, tukey_thresholds, damping)
        self.geo_sigma_bounds = tuple(geo_sigma_bounds)
        self.volume_ratio_bounds = tuple(volume_ratio_bounds)
        assert 0 < geo_sigma_bounds[0] < geo_sigma_initial < geo_sigma_bounds[1]
        assert 0 < volume_ratio_bounds[0] < volume_ratio_initial < volume_ratio_bounds[1]
        def logit(value, bounds):
            fraction = (value-bounds[0])/(bounds[1]-bounds[0])
            return math.log(fraction/(1-fraction))
        self.raw_geo_sigma = nn.Parameter(torch.full((len(search_spacings),), logit(geo_sigma_initial, geo_sigma_bounds)))
        self.raw_volume_ratio = nn.Parameter(torch.full((len(search_spacings),), logit(volume_ratio_initial, volume_ratio_bounds)))

    def geometry_scales(self, stage):
        low, high = self.geo_sigma_bounds
        return F.softplus(self.raw_structure[stage]), low+(high-low)*self.raw_geo_sigma[stage].sigmoid()

    def volume_scales(self, stage):
        low, high = self.volume_ratio_bounds
        ratio = low+(high-low)*self.raw_volume_ratio[stage].sigmoid()
        temperature = .03+9.97*self.raw_temperature.sigmoid()
        cap = ratio*2/temperature
        sigma = self.tukey_thresholds[stage]/(6*cap).sqrt()
        return ratio, cap, sigma

    def correlate_shared(self, observation, points, pose, source, target, structure, visible_ids, stage, appearance=None):
        positions = points@pose[:3, :3].T+pose[:3, 3]
        if appearance is None:
            appearance, offsets = self.appearance_logits(observation, positions, self.search_spacings[stage])
        else:
            offsets = self.template.to(positions)*self.search_spacings[stage]
        valid = visible_ids >= 0
        geometry = torch.zeros_like(appearance)
        if valid.any():
            ids = visible_ids[valid]
            assert torch.allclose(points[valid], source[ids], atol=1e-6, rtol=0)
            precision, _ = self.geometry_precision(pose, structure[ids], stage)
            residual = target[ids]-positions[valid]
            difference = offsets[None]-residual[:, None]
            square = torch.einsum('nki,nij,nkj->nk', difference, precision, difference)
            _, cap, _ = self.volume_scales(stage)
            logits = -cap*6/self.tukey_thresholds[stage]**2*tukey(square, self.tukey_thresholds[stage])
            geometry = geometry.index_copy(0, valid.nonzero().flatten(), logits)
        probability = (appearance+geometry).softmax(-1)
        mean = probability@offsets
        difference = offsets[None]-mean[:, None]
        covariance = torch.einsum('nk,nki,nkj->nij', probability, difference, difference)
        sigma = .005+.095*self.raw_sigma.sigmoid()
        covariance = covariance+sigma.square()*torch.eye(3, device=points.device, dtype=points.dtype)
        return mean, covariance, appearance, offsets

    def bounded_update(self, pose, h, b, positions, max_displacement):
        diagonal = h.diag().clamp_min(h.diag().mean().detach().clamp_min(1.)*1e-6)
        increment = torch.linalg.solve(h+self.damping*torch.diag(diagonal), b)
        j = jacobian(positions, pose[:3, 3])
        displacement = torch.einsum('nia,a->ni', j, increment).norm(dim=-1).max().clamp_min(1e-12)
        scale = (max_displacement/displacement).clamp(max=1.)
        increment = increment*scale
        if not torch.isfinite(increment).all():
            raise FloatingPointError('nonfinite bounded CoVolume update')
        return update_pose(pose, increment), dict(step_scale=float(scale.detach()),
            max_linear_point_step=float((displacement*scale).detach()), increment_norm=float(increment.detach().norm()))

    def warm_start(self, initial, source, target):
        pose = initial.double()
        source, target = source.double(), target.double()
        diagnostics = []
        for threshold in (1.2, .6):
            positions = source@pose[:3, :3].T+pose[:3, 3]
            residual = target-positions
            square = residual.square().sum(-1)
            weights = (1-square/threshold**2).clamp_min(0).square()
            inliers = int((weights.detach()>0).sum())
            if inliers < 6:
                diagnostics.append(dict(threshold=threshold, inliers=inliers, skipped=True))
                break
            j = jacobian(positions, pose[:3, 3])
            h = torch.einsum('nia,n,nib->ab', j, weights, j)
            b = torch.einsum('nia,n,ni->a', j, weights, residual)
            pose, info = self.bounded_update(pose, h, b, positions, threshold)
            diagnostics.append(dict(info, threshold=threshold, inliers=inliers))
        return pose.to(initial.dtype), diagnostics

    def solve_shared(self, points, pose, mean, covariance, source, target, structure, visible_ids, stage):
        current = pose.double()
        positions = points.double()@current[:3, :3].T+current[:3, 3]
        residual = mean.double()
        precision = torch.linalg.inv(covariance.double())
        distance = torch.einsum('ni,nij,nj->n', residual, precision, residual).clamp_min(1e-20).sqrt()
        information = precision*(2.5/distance).clamp(max=1.)[:, None, None]
        j = jacobian(positions, current[:3, 3])
        h = torch.einsum('nia,nij,njb->ab', j, information, j)
        b = torch.einsum('nia,nij,nj->a', j, information, residual)
        outside = torch.ones(len(source), device=pose.device, dtype=torch.bool)
        outside[visible_ids[visible_ids>=0]] = False
        inliers = 0
        support = positions
        if outside.any():
            geo_positions = source[outside].double()@current[:3, :3].T+current[:3, 3]
            geo_residual = target[outside].double()-geo_positions
            geo_precision, sigma = self.geometry_precision(current, structure[outside].double(), stage)
            square = torch.einsum('ni,nij,nj->n', geo_residual, geo_precision, geo_residual).clamp_min(0)
            weights = (1-square/self.tukey_thresholds[stage]**2).clamp_min(0).square()
            inliers = int((weights.detach()>0).sum())
            information = geo_precision*weights[:, None, None]/sigma.square()
            j = jacobian(geo_positions, current[:3, 3])
            h = h+torch.einsum('nia,nij,njb->ab', j, information, j)
            b = b+torch.einsum('nia,nij,nj->a', j, information, geo_residual)
            support = torch.cat((positions, geo_positions))
        current, info = self.bounded_update(current, h, b, support, 2*self.search_spacings[stage])
        return current.to(pose.dtype), dict(info, outside_inliers=inliers)

    def forward_batch(self, points, observations, initials, sources, targets, structures, visible_ids, truths=None):
        warm = [self.warm_start(pose, source, target) for pose, source, target in zip(initials, sources, targets)]
        poses = [row[0] for row in warm]
        losses = [[] for _ in points]
        stages = [[] for _ in points]
        lengths = [len(point) for point in points]
        for stage, spacing in enumerate(self.search_spacings):
            local_diagnostics = [[] for _ in points]
            local_losses = [[] for _ in points]
            for iteration in range(self.iterations):
                positions = [point@pose[:3, :3].T+pose[:3, 3] for point, pose in zip(points, poses)]
                logits, offsets = self.appearance_logits(torch.cat(observations), torch.cat(positions), spacing)
                appearance = logits.split(lengths)
                updated = []
                for index, (point, observed, pose, source, target, structure, visible) in enumerate(zip(points, observations, poses, sources, targets, structures, visible_ids)):
                    mean, covariance, local_logits, _ = self.correlate_shared(observed, point, pose, source, target, structure, visible, stage, appearance[index])
                    coverage = None
                    if truths is not None:
                        delta = point@truths[index][:3, :3].T+truths[index][:3, 3]-positions[index]
                        inside = (delta.detach().abs()<=2*spacing).all(-1)
                        labels = (-((delta.detach()[:, None]-offsets)**2).sum(-1)/(2*(spacing*.5)**2)).softmax(-1)
                        ce = -(labels[inside]*local_logits[inside].log_softmax(-1)).sum(-1).mean() if inside.any() else local_logits.sum()*0
                        local_losses[index].append(ce)
                        coverage = float(inside.float().mean().detach())
                    refined, info = self.solve_shared(point, pose, mean, covariance, source, target, structure, visible, stage)
                    local_diagnostics[index].append(dict(info, iteration=iteration, supervision_fraction=coverage))
                    updated.append(refined)
                poses = updated
            for index in range(len(points)):
                if truths is not None:
                    losses[index].append(torch.stack(local_losses[index]).mean())
                coverage = [row['supervision_fraction'] for row in local_diagnostics[index]]
                stages[index].append(dict(spacing=spacing, iterations=local_diagnostics[index],
                    supervision_fraction=sum(coverage)/len(coverage) if truths is not None else None,
                    first_supervision_fraction=coverage[0], last_supervision_fraction=coverage[-1]))
        return poses, [torch.stack(row).mean() if row else poses[i].sum()*0 for i, row in enumerate(losses)], dict(warm_start=[row[1] for row in warm], stages=stages)

    def forward(self, points, observed, initial, source, target, structure, visible_ids, truth=None):
        poses, losses, diagnostics = self.forward_batch([points], [observed], [initial], [source], [target], [structure], [visible_ids], [truth] if truth is not None else None)
        return poses[0], [losses[0]], dict(warm_start=diagnostics['warm_start'][0], stages=diagnostics['stages'][0],
            visible_fused=int((visible_ids>=0).sum()), geometry_outside=len(source)-len(torch.unique(visible_ids[visible_ids>=0])))


def registration_loss_batch(model, points, all_points, observed, truths, initials, sources, targets, structures, visible_ids):
    poses, correlation, diagnostics = model.forward_batch(points, observed, initials, sources, targets, structures, visible_ids, truths)
    pose_losses = []
    for raw, pose, truth in zip(all_points, poses, truths):
        distance = (raw@pose[:3, :3].T+pose[:3, 3]-(raw@truth[:3, :3].T+truth[:3, 3])).norm(dim=-1)
        pose_losses.append(F.huber_loss(distance, torch.zeros_like(distance), delta=.2))
    metrics = dict(correlation=float(torch.stack(correlation).mean().detach()), pose=float(torch.stack(pose_losses).mean().detach()), batch=len(points))
    for key in ('supervision_fraction', 'first_supervision_fraction', 'last_supervision_fraction'):
        metrics[key+'s'] = [sum(row[stage][key] for row in diagnostics['stages'])/len(points) for stage in range(len(model.search_spacings))]
    metrics['geometry_sigma_m'] = [float(model.geometry_scales(stage)[1].detach()) for stage in range(len(model.search_spacings))]
    metrics['volume_ratio'] = [float(model.volume_scales(stage)[0].detach()) for stage in range(len(model.search_spacings))]
    metrics['volume_sigma_m'] = [float(model.volume_scales(stage)[2].detach()) for stage in range(len(model.search_spacings))]
    return (torch.stack(correlation)+torch.stack(pose_losses)).mean(), metrics


def registration_loss_covolume(model, points, all_points, observed, truth, initial, source, target, structure, visible_ids):
    return registration_loss_batch(model, [points], [all_points], [observed], [truth], [initial], [source], [target], [structure], [visible_ids])
