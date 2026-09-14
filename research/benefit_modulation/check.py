import torch
from model import BenefitModulation, auxiliary_losses, reliability_rank

torch.manual_seed(42)
head = BenefitModulation()
lidar = torch.randn(12, 512)
image = torch.randn(12, 128)
valid = torch.arange(12) % 3 != 0
rank = reliability_rank(torch.arange(12).float())
assert rank[0] == 0 and rank[-1] == 1
assert torch.equal(reliability_rank(torch.tensor([1., 1., 2.])), torch.tensor([.25, .25, 1.]))
out = head(lidar, image, rank, valid)
assert torch.equal(out['fused'], lidar)
assert sum(p.numel() for p in head.parameters()) == 37473
with torch.no_grad():
    head.head[-1].weight.normal_()
out = head(lidar, image, rank, valid)
assert torch.equal(out['fused'][~valid], lidar[~valid])
assert ((out['fused'] - lidar).abs() <= lidar.abs() * out['amplitude'][:, None] + 1e-6).all()
assert torch.equal(head(lidar, image, rank, valid, True)['gate'], torch.full((12,), .5))
decoder = torch.nn.Linear(512, 4).requires_grad_(False)
prediction = decoder(out['fused'])
attempt = decoder(out['attempt'])
target = torch.randn(12, 3)
base = (decoder(lidar)[:, :3] - target).norm(dim=-1).detach()
keep, gate, soft = auxiliary_losses(prediction, attempt, base, target, out['logits'], valid, rank >= .5)
assert not soft.requires_grad
(prediction.square().mean() + keep + .01 * gate).backward()
assert head.head[-1].weight.grad.abs().sum() > 0
assert all(p.grad is None for p in decoder.parameters())
print('PASS: identity, parameter count, mask, bounds, tied ranks, warmup, detached labels, frozen-decoder gradient flow')
