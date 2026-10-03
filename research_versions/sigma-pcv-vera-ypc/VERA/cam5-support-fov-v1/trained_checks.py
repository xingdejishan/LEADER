import json
import time

from common import PLAN, STORE, atomic, gpu_setup, guard, pace, sha

device = gpu_setup()
import numpy as np
import torch
from fov_visibility_v2 import validity
from runtime import Dataset, System, loss, state_dict_sha256

binding = json.loads((STORE / 'parent-binding.json').read_text())
assert sha(binding['checkpoint']) == binding['checkpoint_sha256']
model = System('vera', device).eval()
checkpoint = torch.load(binding['checkpoint'], map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
data = Dataset('train')
before = state_dict_sha256(model.state_dict())
assert before == binding['model_state_sha256']
rows = []
for index in (0, 300, 600, 900):
    started = time.monotonic()
    guard()
    batch, extra, record = data.load(index, True, device)
    with torch.no_grad():
        features, points = model.encode(batch, device)
    output, anchors, eta, active = model.from_encoded(features, points, extra, record, device)
    assert active.any()
    assert validity(anchors[active].detach().cpu().numpy(), extra['visibility'][0], PLAN['zbuffer_tolerance_m']).all()
    value, details = loss(model, output, anchors, eta, batch, device)
    value.backward()
    gradients = {name: None if parameter.grad is None else float(parameter.grad.abs().sum()) for name, parameter in model.vera.named_parameters()}
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.vera.parameters())
    assert gradients['key.weight'] > 0 and gradients['query.weight'] > 0 and gradients['visual.weight'] > 0
    scales = model.vera.last_visibility_scales.detach()
    rows.append({'key': record['key'], 'active': int(active.sum()), 'anchors_pass_fov_depth': int(active.sum()), 'backtracked': int((scales[active] < 1).sum()), 'zero_scale': int((scales[active] == 0).sum()), 'gradients_l1': gradients, 'loss': details})
    model.zero_grad(set_to_none=True)
    assert state_dict_sha256(model.state_dict()) == before
    pace(started)
atomic(STORE / 'trained-checks.json', {'exit': 0, 'binding_sha256': sha(STORE / 'parent-binding.json'), 'checkpoint_sha256': binding['checkpoint_sha256'], 'model_state_sha256': before, 'rows': rows, 'gpu_integration': True, 'training_gt_only': True, 'no_optimizer_step': True, 'cuda_reserved_peak_bytes': torch.cuda.max_memory_reserved(), 'scope': 'four Train frames actual GPU forward/backward at selected parent weights; no development accuracy or convergence claim'})
print(json.dumps({'trained_gpu_checks_passed': True, 'frames': len(rows)}))
