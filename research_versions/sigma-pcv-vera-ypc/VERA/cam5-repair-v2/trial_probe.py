import json
import time

from trial_common import PLAN, STORE, atomic, gpu_setup, guard, optimizer_for, pace, sha, verify_source
device = gpu_setup()
import numpy as np
import torch
from repair_runtime import RepairSystem, Dataset, reference
from compat_det import install
install()

torch.manual_seed(PLAN['seed'])
np.random.seed(PLAN['seed'])
system = RepairSystem(device).train()
baseline_sha = reference.state_dict_sha256(system.base.state_dict())
initial_sha = reference.state_dict_sha256(system.state_dict())
optimizer = optimizer_for(system)
dataset = Dataset('train')
rows = []
for index in (0, 300, 600, 900):
    start = time.monotonic()
    guard()
    batch, extra, record = dataset.load(index, True, device)
    features, points = system.encode(batch, device)
    result, base, geometry = system.from_encoded(features, points, extra)
    if index == 0:
        assert torch.equal(result.anchors, points) and torch.equal(result.output, base)
        assert torch.equal(result.selected, base[:, 3].topk(max(min(50, len(base)), len(base) // 2)).indices)
    value, evidence = system.loss(result, base, points, batch, device)
    assert torch.isfinite(value)
    optimizer.zero_grad(set_to_none=True)
    value.backward()
    assert all(torch.isfinite(p.grad).all() for p in system.parameters() if p.grad is not None)
    assert all(p.grad is None for p in system.base.parameters())
    gradients = {name: None if parameter.grad is None else float(parameter.grad.abs().sum()) for name, parameter in system.repair.named_parameters()}
    torch.nn.utils.clip_grad_norm_([p for p in system.parameters() if p.requires_grad], 10.)
    optimizer.step()
    rows.append({'key': record['key'], 'voxels': len(points), 'visible_candidates': int(geometry['valid'].sum()), 'supervised_candidates': evidence['supervised_candidates'], 'loss': float(value.detach()), 'admitted': int(result.admitted.sum()), 'swaps': result.swaps, 'seconds_compute': time.monotonic() - start, 'gradients': gradients})
    del value, result, base, geometry, features, points, extra, batch
    pace(start)
assert reference.state_dict_sha256(system.base.state_dict()) == baseline_sha
assert reference.state_dict_sha256(system.state_dict()) != initial_sha
system.eval()
dev = Dataset('dev')
start = time.monotonic()
with torch.no_grad():
    batch, extra, record = dev.load(0, True, device)
    assert 'T' not in batch
    result, _, _ = system(batch, extra, device)
    pose, count = system.pose(result, batch, PLAN['seed'] + record['dataset_index'], device)
    assert torch.isfinite(pose).all()
pace(start)
verify_source()
guard()
atomic(STORE / 'probe-complete.json', {'passed': True, 'train_frames': 4, 'optimizer_steps': 4, 'probe_weights_discarded': True, 'cold_baseline_exact': True, 'base_frozen_verified': True, 'rows': rows, 'dev_pose_smoke_key': record['key'], 'dev_pose': pose.detach().cpu().tolist(), 'selected': count, 'development_gt_reads': 0, 'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(), 'plan_sha256': sha(STORE / 'plan.json'), 'accuracy_claim': False})
print(json.dumps({'event': 'probe_complete', 'passed': True, 'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()}), flush=True)
