import json
import math
import os
import sys
import time

from trial_common import PLAN, STORE, atomic, gpu_setup, guard, load_modules, optimizer_for, pace, sha, verify_source
device = gpu_setup()
import numpy as np
import torch
from repair_runtime import RepairSystem, Dataset, reference

end = int(sys.argv[1])
assert end in PLAN['evaluate_epochs']
torch.manual_seed(PLAN['seed'])
np.random.seed(PLAN['seed'])
system = RepairSystem(device).train()
dataset = Dataset('train')
assert len(dataset) == 1200
optimizer = optimizer_for(system)
folder = STORE / 'training'
folder.mkdir(exist_ok=True)
latest = folder / 'latest.pt'
start_epoch, first_step, updates, history, totals = 1, 0, 0, [], {}
baseline_sha = reference.state_dict_sha256(system.base.state_dict())
if latest.exists():
    receipt = json.loads((folder / 'latest-receipt.json').read_text())
    assert sha(latest) == receipt['sha256']
    payload = torch.load(latest, map_location='cpu')
    load_modules(system, payload)
    assert payload['baseline_state_sha256'] == baseline_sha
    optimizer.load_state_dict(payload['optimizer'])
    torch.set_rng_state(payload['torch_rng'])
    torch.cuda.set_rng_state(payload['cuda_rng'])
    start_epoch, first_step, updates, history, totals = payload['resume_epoch'], payload['next_step'], payload['updates'], payload['history'], payload['partial_totals']
    del payload
else:
    atomic(folder / 'initialization.json', {'baseline_state_sha256': baseline_sha, 'new_module_initial_state_sha256': reference.state_dict_sha256({**{'tokens.' + k: v for k, v in system.tokens.state_dict().items()}, **{'repair.' + k: v for k, v in system.repair.state_dict().items()}}), 'probe_weights_reused': False, 'baseline_frozen': True, 'plan_sha256': sha(STORE / 'plan.json')})


def save(epoch, step, epoch_complete=None):
    guard()
    payload = {'tokens': {k: v.detach().cpu() for k, v in system.tokens.state_dict().items()}, 'repair': {k: v.detach().cpu() for k, v in system.repair.state_dict().items()}, 'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(), 'resume_epoch': epoch, 'next_step': step, 'updates': updates, 'history': history, 'partial_totals': totals, 'epoch': epoch_complete, 'plan_sha256': sha(STORE / 'plan.json'), 'baseline_checkpoint_sha256': PLAN['checkpoint_sha256'], 'baseline_state_sha256': baseline_sha}
    temporary = folder / 'latest.pt.tmp'
    torch.save(payload, temporary)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, latest)
    atomic(folder / 'latest-receipt.json', {'sha256': sha(latest), 'resume_epoch': epoch, 'next_step': step, 'updates': updates, 'time': time.time()})
    if epoch_complete in PLAN['evaluate_epochs']:
        target = folder / ('epoch-%02d.pt' % epoch_complete)
        assert not target.exists()
        os.link(latest, target)
        atomic(folder / ('epoch-%02d-receipt.json' % epoch_complete), {'checkpoint_sha256': sha(target), 'epoch': epoch_complete, 'updates': updates, 'history': history})


for epoch in range(start_epoch, end + 1):
    system.train()
    order = np.random.default_rng(PLAN['seed'] + 1000 + epoch).permutation(len(dataset))
    scale = .2 + .8 * .5 * (1 + math.cos(math.pi * (epoch - 1) / max(1, PLAN['epochs'] - 1)))
    for group in optimizer.param_groups:
        group['lr'] = (PLAN['rgb_lr'] if group['name'] == 'rgb' else PLAN['lr']) * scale
    optimizer.zero_grad(set_to_none=True)
    first = first_step if epoch == start_epoch else 0
    if first == 0:
        totals = {}
    assert first % PLAN['accumulation'] == 0
    begin = time.monotonic()
    for step in range(first, len(order)):
        start = time.monotonic()
        guard()
        batch, extra, record = dataset.load(int(order[step]), True, device)
        features, points = system.encode(batch, device)
        result, base, geometry = system.from_encoded(features, points, extra)
        value, evidence = system.loss(result, base, points, batch, device)
        assert torch.isfinite(value)
        (value / PLAN['accumulation']).backward()
        details = {key: float(number.detach()) if torch.is_tensor(number) else float(number) for key, number in evidence.items()}
        valid_points = result.valid.any(-1)
        entropy = -(result.attention.clamp_min(1e-12).log() * result.attention).sum(-1)
        denominator = result.valid.sum(-1).clamp_min(2).float().log()
        details.update({'loss': float(value.detach()), 'admitted': int(result.admitted.sum()), 'admitted_in_pool': int(result.admitted[result.selected].sum()), 'swaps': result.swaps, 'voxels': len(points), 'normalized_attention_entropy': float((entropy[valid_points] / denominator[valid_points]).mean()) if valid_points.any() else 0.})
        for key, number in details.items():
            totals[key] = totals.get(key, 0.) + number
        if (step + 1) % PLAN['accumulation'] == 0:
            assert all(torch.isfinite(p.grad).all() for p in system.parameters() if p.grad is not None)
            assert all(p.grad is None for p in system.base.parameters())
            if updates % 50 == 0:
                atomic(folder / 'gradient-evidence.json', {'epoch': epoch, 'updates_before_step': updates, 'parameters': {name: None if parameter.grad is None else float(parameter.grad.abs().sum()) for name, parameter in system.named_parameters() if parameter.requires_grad}, 'base_has_grad': False})
            torch.nn.utils.clip_grad_norm_([p for p in system.parameters() if p.requires_grad], 10.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if updates % PLAN['checkpoint_every_updates'] == 0:
                save(epoch, step + 1)
        if (step + 1) % 20 == 0:
            atomic(folder / 'progress.json', {'epoch': epoch, 'frames': step + 1, 'total': len(order), 'updates': updates, 'time': time.time(), 'metrics_current': details})
            print(json.dumps({'event': 'train', 'epoch': epoch, 'frames': step + 1, 'loss': details['loss']}), flush=True)
        del value, result, base, geometry, features, points, batch, extra
        pace(start)
    assert reference.state_dict_sha256(system.base.state_dict()) == baseline_sha
    history.append({'epoch': epoch, 'frames': len(dataset), 'updates': updates, 'metrics': {key: number / len(dataset) for key, number in totals.items()}, 'seconds_current_attempt': time.monotonic() - begin, 'baseline_frozen_verified': True})
    atomic(folder / 'history.json', history)
    totals = {}
    save(epoch + 1, 0, epoch)
    print(json.dumps({'event': 'epoch_complete', **history[-1]}), flush=True)
verify_source()
atomic(folder / ('train-to-%02d-complete.json' % end), {'epoch': end, 'updates': updates, 'checkpoint_sha256': sha(folder / ('epoch-%02d.pt' % end)), 'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(), 'plan_sha256': sha(STORE / 'plan.json')})
