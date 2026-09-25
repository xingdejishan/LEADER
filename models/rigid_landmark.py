import torch
from torch import nn

from models.landmark_memory import LandmarkMemory


def skew(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*vector.shape[:-1], 3, 3)


class RigidLandmarkFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = LandmarkMemory()
        self.weight = nn.Linear(64, 1)
        nn.init.zeros_(self.weight.weight)
        nn.init.zeros_(self.weight.bias)

    def forward(self, source, baseline, point_valid, memory_inputs):
        batch, points = source.shape[:2]
        delta, context = self.memory(*memory_inputs, return_context=True)
        delta = delta.reshape(batch, points, 3)
        weight = self.weight(context).sigmoid().reshape(batch, points) * point_valid
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
        local_delta = delta @ baseline[:, :3, :3]
        identity = torch.eye(3, device=source.device, dtype=source.dtype)
        jacobian = torch.cat((identity.expand(batch, points, 3, 3), -skew(source / 10.0)), dim=-1)
        normal = torch.einsum('bnci,bncj,bn->bij', jacobian, jacobian, weight)
        rhs = torch.einsum('bnci,bnc,bn->bi', jacobian, local_delta, weight)
        normal = normal + 1e-3 * torch.eye(6, device=source.device, dtype=source.dtype)[None]
        twist = torch.linalg.solve(normal, rhs[..., None]).squeeze(-1)
        translation = twist[:, :3].tanh()
        rotation = .1 * (twist[:, 3:] / 1.0).tanh()
        correction = torch.eye(4, device=source.device, dtype=source.dtype)[None].repeat(batch, 1, 1)
        correction[:, :3, :3] = torch.matrix_exp(skew(rotation))
        correction[:, :3, 3] = translation
        return baseline @ correction
