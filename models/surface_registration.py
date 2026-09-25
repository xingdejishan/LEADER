import torch
from torch import nn
from torch.nn import functional as F

from models.rigid_landmark import skew


class MultimodalSurfaceRotation(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Linear(32, 32, bias=False)
        nn.init.eye_(self.image.weight)
        self.match = nn.Sequential(nn.Linear(69, 32), nn.GELU(), nn.Linear(32, 1))
        self.confidence = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        for module in (self.match, self.confidence):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)

    def forward(self, batch):
        source = batch['source'].float()
        reference = batch['reference'].float()
        normal = batch['reference_normal'].float()
        valid = batch['valid'].bool()
        baseline = batch['baseline'].float()
        rotation = baseline[:, :3, :3]
        translation = baseline[:, :3, 3]
        q = F.normalize(self.image(batch['source_image'].float()), dim=-1)
        r = F.normalize(self.image(batch['reference_image'].float()), dim=-1)
        visual = torch.cat((q[:, :, None]*r, (q[:, :, None]-r).abs()), dim=-1)
        similarity = (q[:, :, None]*r).sum(-1)
        for _ in range(3):
            world = source @ rotation.transpose(1, 2) + translation[:, None]
            delta = reference-world[:, :, None]
            distance2 = delta.square().sum(-1)
            local_normal = torch.einsum('bnki,bij->bnkj', normal, rotation)
            local_delta = torch.einsum('bnki,bij->bnkj', delta, rotation)
            alignment = (batch['source_normal'][:, :, None]*local_normal).sum(-1).abs()
            edge = torch.cat((visual, local_delta/1.5, alignment[..., None],
                              distance2.clamp_min(1e-10).sqrt()[..., None]/1.5), dim=-1)
            logits = self.match(edge).squeeze(-1) + 2*similarity - distance2/(2*.5**2)
            attention = logits.masked_fill(~valid, -1e4).softmax(-1)*valid
            attention = attention/attention.sum(-1, keepdim=True).clamp_min(1e-8)
            context = (attention[..., None]*r).sum(-2)
            confidence = self.confidence(torch.cat((q, context), dim=-1)).sigmoid().squeeze(-1)
            residual = -(normal*delta).sum(-1)
            weight = attention*confidence[..., None]/(1+(residual/.2).square())
            weight = weight/weight.sum((1, 2), keepdim=True).clamp_min(1e-8)
            jacobian = torch.cross(source[:, :, None].expand_as(local_normal)/10, local_normal, dim=-1)
            hessian = torch.einsum('bnki,bnkj,bnk->bij', jacobian, jacobian, weight)
            rhs = torch.einsum('bnki,bnk,bnk->bi', jacobian, residual, weight)
            hessian = hessian+1e-3*torch.eye(3, device=source.device)[None]
            scaled = -torch.linalg.solve(hessian, rhs[..., None]).squeeze(-1)
            omega = .05*(scaled/.5).tanh()
            rotation = rotation @ torch.matrix_exp(skew(omega))
        output = baseline.clone()
        output[:, :3, :3] = rotation
        return output
