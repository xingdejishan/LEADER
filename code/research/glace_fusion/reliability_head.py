"""Per-correspondence reliability head for the supervised GLACE (improvement B).

A small MLP over the head-input features (encoder + global concatenation) and
the predicted camera-frame coordinate. Trained jointly with the coordinate
head on training-only LiDAR support: the label marks cells whose predicted
depth agrees with the LiDAR median ray depth within a factor
`depth_ratio_tol` (default 1.25). At inference it yields per-cell reliability
in [0, 1] that the fusion backend uses as the camera weight prior, replacing
the uniform 1/N_C.
"""
import math

import torch
from torch import nn


class ReliabilityHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.in_dim = int(in_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim + 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, features, pred_cam):
        """features: [N, in_dim]; pred_cam: [N, 3] camera-frame coordinates.
        Returns logits [N]."""
        if features.shape[1] != self.in_dim:
            raise ValueError(f'Feature dimension {features.shape[1]} != {self.in_dim}')
        return self.net(torch.cat((features, pred_cam), dim=1)).reshape(-1)

    @torch.no_grad()
    def reliability(self, features, pred_cam):
        return torch.sigmoid(self.forward(features, pred_cam))

    @staticmethod
    def consistency_labels(pred_cam, target_cam, valid, depth_ratio_tol=1.25):
        """1 where the predicted depth matches the LiDAR ray depth within
        depth_ratio_tol, on supported samples only; 0 elsewhere (the caller
        must mask unsupported samples out of the loss)."""
        pred_z = pred_cam[:, 2].clamp_min(1e-3)
        target_z = target_cam[:, 2].clamp_min(1e-3)
        log_ratio = torch.log(pred_z) - torch.log(target_z)
        agree = (log_ratio.abs() <= math.log(depth_ratio_tol)).float()
        return agree * valid.reshape(-1)

    def save_payload(self):
        return dict(in_dim=self.in_dim, state_dict=self.state_dict())
