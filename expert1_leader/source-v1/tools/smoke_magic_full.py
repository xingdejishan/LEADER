import json

import MinkowskiEngine as ME
import torch

from models.magic_fusion import polar_voxel_centers
from models.model_mink import LEADER


torch.manual_seed(9)
coordinates = torch.tensor(
    [[0, x, y, z] for x in range(-8, 24, 4)
     for y in range(8, 24, 4) for z in range(4, 20, 4)],
    dtype=torch.int32,
    device='cuda',
)
features = torch.randn(coordinates.shape[0], 3, device='cuda')
inputs = ME.SparseTensor(features, coordinates)
model = LEADER(in_channels=3, out_channels=4, feat_channels=512, magic=True).cuda()
model.eval()
encoded, stages = model.encoder(inputs, return_stages=True)
stride = torch.tensor(encoded.tensor_stride, device='cuda', dtype=torch.float32)
points = polar_voxel_centers(encoded.C, stride, 0.2, 1024)
sam = torch.randn(1, 256, 64, 64, device='cuda')
intrinsic = torch.tensor([[[120., 0., 512.], [0., 120., 400.], [0., 0., 1.]]], device='cuda')
identity = torch.eye(4, device='cuda')[None]
bounds = torch.tensor([[1024., 781.]], device='cuda')
model.magic_fusion.aggregate.output.weight.data.fill_(1e-5)
fused = model.magic_fusion(encoded.F, points, encoded.C, stride, sam, intrinsic,
                           identity, identity, bounds, stages=stages,
                           voxel_size=0.2, horizontal=1024)
prediction = model.decoder(fused)
loss = prediction.square().mean()
loss.backward()
finite = bool(torch.isfinite(prediction).all().item())
grad_norm = float(model.magic_fusion.stage_projections[0].weight.grad.norm().item())
if not finite or grad_norm <= 0:
    raise RuntimeError('Full MaGiC forward/backward failed')
print(json.dumps({
    'input_voxels': coordinates.shape[0],
    'stage_voxels': [stage.F.shape[0] for stage in stages],
    'output_voxels': encoded.F.shape[0],
    'output_finite': finite,
    'stage_projection_grad': grad_norm,
    'peak_allocated_mb': round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1),
}))
