import math

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import bearing_linearization, lidar_normal, normal_system, predictive_score, retract


class RayBundleCorrespondence(nn.Module):
    def __init__(self, descriptor_dim=128, rounds=3, angular_sigma=.005,
                 max_queries=64, max_candidates=4):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(descriptor_dim, descriptor_dim), nn.GELU(),
                                     nn.Linear(descriptor_dim, descriptor_dim))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.rounds = int(rounds)
        self.angular_sigma = float(angular_sigma)
        self.max_queries = int(max_queries)
        self.max_candidates = int(max_candidates)

    def encode(self, descriptors):
        x = F.normalize(descriptors, dim=-1)
        return F.normalize(x + self.adapter(x), dim=-1)

    def prior(self, query, reference, indices):
        q, r = self.encode(query), self.encode(reference)
        logits = 12 * (q[:, None] * r[indices.clamp_min(0)]).sum(-1)
        logits = logits.masked_fill(indices < 0, -torch.inf)
        null = logits.new_full((len(q), 1), 6.)
        return F.log_softmax(torch.cat((logits, null), -1), -1)

    def validate(self, frame, ray_map, candidates):
        forbidden = {'T_WB', 'ground_truth', 'gt_pose', 'target_pose'}
        if forbidden.intersection(frame):
            raise ValueError('query GT must be held outside the online frame')
        n, k = candidates.shape
        if not 0 < n <= self.max_queries or not 0 < k <= self.max_candidates:
            raise ValueError('query or candidate budget exceeded')
        if len(ray_map['mean']) == 0 or len(ray_map['mean']) > 20000:
            raise ValueError('invalid reference map size')
        if candidates.max() >= len(ray_map['mean']) or candidates.min() < -1:
            raise ValueError('invalid candidate identity')
        for key in ('source', 'target', 'weights', 'T0', 'origins', 'directions', 'descriptors', 'shared_covariance'):
            if not torch.isfinite(frame[key]).all():
                raise ValueError('nonfinite online input: ' + key)
        if frame['source'].shape != frame['target'].shape or len(frame['source']) > 10000:
            raise ValueError('invalid LiDAR correspondences')
        if len(frame['source']) < 6 or torch.any(frame['weights'] < 0):
            raise ValueError('insufficient or negative LiDAR evidence')
        if not math.isfinite(float(frame['independent_sigma'])) or float(frame['independent_sigma']) <= 0:
            raise ValueError('independent point noise must be positive and finite')
        if frame['origins'].shape != (n,3) or frame['directions'].shape != (n,3) or len(frame['descriptors']) != n:
            raise ValueError('ray and descriptor rows differ')
        if not torch.allclose(frame['directions'].norm(dim=-1), torch.ones(n, dtype=frame['directions'].dtype, device=candidates.device), atol=1e-6):
            raise ValueError('query directions must be unit vectors')
        for key in ('precision', 'mean', 'descriptors'):
            if not torch.isfinite(ray_map[key]).all():
                raise ValueError('nonfinite map input')
        if torch.linalg.eigvalsh(frame['shared_covariance']).min() <= 0 or torch.linalg.eigvalsh(ray_map['precision']).min() <= 0:
            raise ValueError('error models must be positive definite')

    def _hard_assignment(self, scores, indices, cameras):
        from scipy.optimize import linear_sum_assignment
        output = torch.zeros_like(scores)
        for camera in torch.unique(cameras):
            rows = torch.nonzero(cameras == camera, as_tuple=False).flatten()
            ids = torch.unique(indices[rows][indices[rows] >= 0]).tolist()
            lookup = {track: col for col, track in enumerate(ids)}
            cost = scores.new_full((len(rows), len(ids) + len(rows)), 1e12)
            for row, global_row in enumerate(rows.tolist()):
                for col, track in enumerate(indices[global_row].tolist()):
                    if track >= 0:
                        cost[row, lookup[track]] = -scores[global_row, col]
                cost[row, len(ids):] = -scores[global_row, -1]
            rr, cc = linear_sum_assignment(cost.detach().cpu().numpy())
            for row, col in zip(rr, cc):
                global_row = int(rows[row])
                selected = -1 if col >= len(ids) else int(torch.nonzero(indices[global_row] == ids[col], as_tuple=False)[0, 0])
                output[global_row, selected] = 1
        return output

    def forward(self, frame, ray_map, candidates, hard=None):
        self.validate(frame, ray_map, candidates)
        hard = not self.training if hard is None else bool(hard)
        active = torch.unique(candidates[candidates >= 0])
        if len(active) == 0:
            probabilities = frame['T0'].new_zeros(len(candidates), candidates.shape[1] + 1)
            probabilities[:, -1] = 1
            return dict(pose=frame['T0'], probabilities=probabilities, candidates=candidates,
                        active_ids=active, points=ray_map['mean'][active], scores=probabilities,
                        point_covariance=ray_map['precision'][active], initial_scores=probabilities,
                        selected=candidates.new_full((len(candidates),), -1))
        lookup = candidates.new_full((len(ray_map['mean']),), -1)
        lookup[active] = torch.arange(len(active), device=candidates.device)
        indices = torch.where(candidates >= 0, lookup[candidates.clamp_min(0)], -1)
        mean, precision = ray_map['mean'][active], ray_map['precision'][active]
        points, pose = mean.clone(), frame['T0'].clone()
        prior = self.prior(frame['descriptors'], ray_map['descriptors'][active], indices)
        alpha = pose.new_zeros(*indices.shape)
        origins = frame['origins'][:, None].expand(*indices.shape, 3)
        directions = frame['directions'][:, None].expand(*indices.shape, 3)
        groups = frame.get('group_ids', torch.arange(len(indices), device=indices.device))
        cameras = frame.get('camera_ids', torch.zeros(len(indices), device=indices.device, dtype=torch.long))
        initial_scores = None
        variance = self.angular_sigma ** 2
        for step in range(self.rounds + int(hard)):
            r, j, d, visible = bearing_linearization(pose, points[indices.clamp_min(0)], origins, directions)
            lh, lg = lidar_normal(pose, frame['source'], frame['target'], frame['weights'],
                                  float(frame['independent_sigma']), frame['shared_covariance'])
            scores = torch.empty_like(prior)
            for group in torch.unique(groups):
                members = groups == group
                other_alpha = alpha * (~members)[:, None]
                system = normal_system(lh, lg, points, mean, precision, indices, r, j, d, other_alpha, variance)
                value = predictive_score(system, indices[members], r[members], j[members], d[members], variance)
                scores[members, :-1] = prior[members, :-1] + value
            scores[:, -1] = prior[:, -1] - math.log(4 * math.pi)
            scores = torch.cat((scores[:, :-1].masked_fill((indices < 0) | ~visible, -torch.inf), scores[:, -1:]), -1)
            if initial_scores is None:
                initial_scores = scores
            probabilities = self._hard_assignment(scores, indices, cameras) if hard else F.softmax(scores, -1)
            alpha = probabilities[:, :-1]
            system = normal_system(lh, lg, points, mean, precision, indices, r, j, d, alpha, variance)
            delta, delta_points = system[:2]
            scale = torch.minimum(delta.new_tensor(1.), torch.minimum(1. / delta[:3].norm().clamp_min(1e-8), .15 / delta[3:].norm().clamp_min(1e-8)))
            pose = retract(pose, scale * delta)
            points = points + scale * delta_points
        chosen = probabilities.argmax(-1)
        selected = torch.where(chosen < indices.shape[1], candidates.gather(1, chosen.clamp_max(indices.shape[1]-1)[:, None]).squeeze(1), -1)
        point_covariance = system[3] + system[4].transpose(-1, -2) @ system[2] @ system[4]
        return dict(pose=pose, probabilities=F.softmax(scores, -1), candidates=candidates, active_ids=active,
                    points=points, point_covariance=point_covariance, scores=scores,
                    initial_scores=initial_scores, selected=selected)
