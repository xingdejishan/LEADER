import json
import torch
from torch.nn import functional as F
from fusion import ImageGate, project, sample_map, sample_visible


torch.manual_seed(2089)
eye = torch.eye(4)
k = torch.eye(3)
points = torch.tensor([[1., 1., 1.], [2., 2., 2.], [0., 0., -1.], [100., 0., 1.]])
uv, depth, valid = project(points, eye, k, (4, 4))
assert valid.tolist() == [True, True, False, False]
correction = eye.clone()
correction[:3, 3] = torch.tensor([3., 2., 4.])
shifted, _, shifted_valid = project(points + correction[:3, 3], eye, k, (4, 4), correction)
assert torch.equal(valid, shifted_valid) and torch.allclose(uv, shifted)
grid = torch.arange(16).float().reshape(1, 1, 4, 4)
assert sample_map(grid, torch.tensor([[1., 1.], [1.5, 1.]]), (4, 4))[:, 0].tolist() == [5., 5.5]
_, visible = sample_visible(grid, points, eye, k, torch.ones(4, 4), points, cell_size=1, tolerance=.1)
assert visible.tolist() == [True, False, False, False]
dense = torch.randn(1, 256, 8, 8)
weight, bias = torch.randn(128, 256, 1, 1), torch.randn(128)
queries = torch.rand(20, 2) * 6 + .5
left = sample_map(F.conv2d(dense, weight, bias), queries, (8, 8))
right = sample_map(dense, queries, (8, 8)) @ weight[:, :, 0, 0].T + bias
assert torch.allclose(left, right, atol=3e-5, rtol=1e-4)
gate = ImageGate()
lidar, image = torch.randn(20, 512), torch.randn(20, 128)
valid = torch.arange(20) % 2 == 0
assert torch.equal(gate(lidar, image, valid), lidar)
gate(lidar, image, valid).sum().backward()
assert gate.residual[-1].weight.grad.abs().sum() > 0
with torch.no_grad():
    gate.residual[-1].weight.normal_()
image[~valid] = float('nan')
output = gate(lidar, image, valid)
assert torch.equal(output[~valid], lidar[~valid]) and torch.isfinite(output).all()
print(json.dumps(dict(projection=True, correction=True, bilinear=True, occlusion=True,
                      dense_pca_equivalence=True, identity_initialization=True,
                      residual_gradient=True, missing_modality_parity=True,
                      parameters=sum(p.numel() for p in gate.parameters())), indent=2))
