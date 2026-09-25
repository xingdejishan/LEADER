import torch
from torch import nn
from torch.nn import functional as F

from models.rigid_landmark import skew


class MultimodalGravityFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Linear(32, 16)
        self.point = nn.Sequential(nn.Linear(26, 32), nn.GELU(), nn.Linear(32, 17))
        self.fusion = nn.Sequential(nn.Linear(25, 32), nn.GELU(), nn.Linear(32, 5))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        with torch.no_grad():
            self.fusion[-1].bias[:2].fill_(-3.)

    def forward(self, batch):
        pose = batch['baseline']
        original = -pose[:, 2, :3].float()
        visual = F.normalize(batch['visual_up'].float(), dim=-1)
        normal = batch['normal'].float()
        alignment = (normal*original[:, None]).sum(-1, keepdim=True)
        normal = normal*torch.where(alignment >= 0, 1., -1.)
        features = torch.cat((self.image(batch['image'].float()), batch['source'].float()/20,
                              normal, original[:, None].expand_as(normal), alignment.abs()), -1)
        encoded = self.point(features)
        valid = batch['valid'] & (alignment[..., 0].abs() > .7)
        weight = torch.softmax(encoded[..., 0].masked_fill(~valid, -1e4), -1)*valid
        weight = weight/weight.sum(-1, keepdim=True).clamp_min(1e-8)
        normal_delta = ((normal-original[:, None])*weight[..., None]).sum(1)
        context = (encoded[..., 1:]*weight[..., None]).sum(1)
        inputs = torch.cat((context, visual-original, normal_delta, batch['uncertainty'].float(),
                            weight.sum(-1, keepdim=True)), -1)
        coefficients = self.fusion(inputs)
        visual_delta = .1*torch.tanh((visual-original)/.1)
        delta = coefficients[:, :1].sigmoid()*visual_delta + coefficients[:, 1:2].sigmoid()*normal_delta
        delta = delta + .02*torch.tanh(coefficients[:, 2:])
        delta = delta-(delta*original).sum(-1, keepdim=True)*original
        gravity = F.normalize(original+delta, dim=-1)
        cross = torch.cross(gravity, original, dim=-1)
        matrix = skew(cross)
        identity = torch.eye(3, device=pose.device)[None]
        correction = identity+matrix+matrix@matrix/(1+(gravity*original).sum(-1))[:, None, None].clamp_min(1e-6)
        output = pose.clone()
        output[:, :3, :3] = pose[:, :3, :3]@correction.to(pose.dtype)
        return output
