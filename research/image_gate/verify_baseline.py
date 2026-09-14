import ast
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import numpy as np
import torch
import MinkowskiEngine as ME
from fusion import ImageGate
from run import REPO, read_scan

sys.path.insert(0, str(REPO))
from utils.pose_util import cartesian_to_polar_expansion

root = Path('/home/zhang/leader-image-gate')
rows = json.loads((root / 'manifest.json').read_text())
tree = ast.parse((REPO / 'data/NCLTVelodyne_datagenerator_mink.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NCLT_mink')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__getitem__')
scope = dict(np=np, ME=ME, cartesian_to_polar_expansion=cartesian_to_polar_expansion)
reader = ast.parse((REPO / 'data/robotcar_sdk/python/velodyne.py').read_text())
nodes = [n for n in reader.body if isinstance(n, ast.FunctionDef) and n.name in ['get_velo', 'data2xyzi']
         or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'velodatatype' for t in n.targets)]
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'official_NCLT_reader', 'exec'), scope)
exec(compile(ast.Module(body=[method], type_ignores=[]), 'official_NCLT_getitem', 'exec'), scope)
for row in [rows[0], rows[32], rows[-1]]:
    stub = SimpleNamespace(pcs=[row['scan']], poses=np.zeros((1, 3)), rots=np.eye(3)[None],
                           min_range=1., max_range=100., voxel_size=.2, horizontal_res=1024, level_correction=False)
    coords, feats, scan, _, _ = scope['__getitem__'](stub, 0)
    ours, intensity = read_scan(row['scan'])
    polar = cartesian_to_polar_expansion(ours, .2 * 1024)
    oc, of = ME.utils.sparse_quantize(coordinates=polar, features=np.column_stack([polar[:, 2], polar[:, 1], intensity]), quantization_size=.2)
    assert np.array_equal(scan, ours) and np.array_equal(coords, oc) and np.array_equal(feats, of)
torch.set_num_threads(4)
gate = ImageGate().cuda().eval()
gate.load_state_dict(torch.load(root / 'aligned.pt', map_location='cuda'))
with np.load(root / 'lidar' / (rows[-1]['frame_id'] + '.npz')) as data:
    features = torch.tensor(data['features'], device='cuda')
with np.load(root / 'visual' / (rows[-1]['frame_id'] + '.npz')) as data:
    image, valid = torch.tensor(data['image'], device='cuda'), torch.tensor(data['valid'], device='cuda')
with torch.inference_mode():
    for _ in range(20):
        gate(features, image, valid)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(100):
        output = gate(features, image, valid)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - begin) / 100
    relative = ((output - features)[valid].norm(dim=-1) / features[valid].norm(dim=-1).clamp_min(1e-8)).mean().item()
    assert torch.equal(output[~valid], features[~valid])
result = dict(official_loader_input_parity_frames=3, all_inputs_exact=True,
              missing_voxels_exact_after_training=True, gate_seconds_per_frame=seconds,
              benchmark_voxels=len(features), valid_residual_relative_norm=relative,
              parameters=sum(p.numel() for p in gate.parameters()))
(root / 'verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
