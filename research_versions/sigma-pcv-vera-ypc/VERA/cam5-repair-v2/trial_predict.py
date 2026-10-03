import json
import os
import resource
import sys
import time

from trial_common import PLAN, STORE, atomic, gpu_setup, guard, load_modules, pace, sha, verify_source
device = gpu_setup()
import numpy as np
import torch
from repair_runtime import RepairSystem, Dataset, reference
from compat_det import install
install()

epoch = int(sys.argv[1])
assert epoch in PLAN['evaluate_epochs']
torch.manual_seed(PLAN['seed'])
np.random.seed(PLAN['seed'])
system = RepairSystem(device).eval()
checkpoint = STORE / 'training' / ('epoch-%02d.pt' % epoch)
receipt = json.loads(checkpoint.with_name('epoch-%02d-receipt.json' % epoch).read_text())
assert sha(checkpoint) == receipt['checkpoint_sha256']
payload = torch.load(checkpoint, map_location='cpu')
assert payload['epoch'] == epoch
load_modules(system, payload)
del payload
before = reference.state_dict_sha256(system.state_dict())
dataset = Dataset('dev')
assert len(dataset) == 1200
folder = STORE / 'eval' / ('e%02d' % epoch)
folder.mkdir(parents=True, exist_ok=True)
path = folder / 'predictions.jsonl'
binding = {'checkpoint_sha256': sha(checkpoint), 'plan_sha256': sha(STORE / 'plan.json'), 'inputs_sha256': sha(STORE / 'inputs.json')}
marker = folder / 'prediction-start.json'
if marker.exists():
    assert json.loads(marker.read_text()) == binding
else:
    atomic(marker, binding)
prior = [json.loads(row) for row in path.read_text().splitlines()] if path.exists() else []
assert [row['key'] for row in prior] == [row['key'] for row in dataset.inputs[:len(prior)]]
failures = sum(row['status'] != 'success' for row in prior)
begin = time.monotonic()
with path.open('a') as stream, torch.no_grad():
    for index in range(len(prior), len(dataset)):
        start = time.monotonic()
        guard()
        record = dataset.inputs[index]
        row = {'key': record['key'], 'index': index, 'status': 'success'}
        try:
            batch, extra, _ = dataset.load(index, True, device)
            assert 'T' not in batch
            result, _, _ = system(batch, extra, device)
            seed = PLAN['seed'] + record['dataset_index']
            pose, count = system.pose(result, batch, seed, device)
            row.update({'pose': pose.detach().cpu().double().tolist(), 'valid_corr': len(result.anchors), 'selected_corr': count, 'admitted': int(result.admitted.sum()), 'admitted_in_pool': int(result.admitted[result.selected].sum()), 'pool_swaps': result.swaps, 'rotation_valid': result.rotation_valid, 'seed': seed})
        except (RuntimeError, ValueError, AssertionError) as error:
            if any(word in str(error).lower() for word in ('out of memory', 'nvrtc', 'gpu-architecture', 'memory floor', 'rss ceiling', 'disk free')):
                raise
            row.update({'status': 'failure', 'pose': np.eye(4).tolist(), 'error': str(error)})
            failures += 1
        row['seconds'] = time.monotonic() - start
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        if (index + 1) % 20 == 0:
            os.fsync(stream.fileno())
            atomic(folder / 'progress.json', {'frames': index + 1, 'failures': failures, 'time': time.time()})
        if 'result' in locals():
            del result
        if 'extra' in locals():
            del extra
        if 'batch' in locals():
            del batch
        pace(start)
assert reference.state_dict_sha256(system.state_dict()) == before
verify_source()
atomic(folder / 'prediction-freeze.json', {'frames': len(dataset), 'failures': failures, 'prediction_sha256': sha(path), 'checkpoint_sha256': sha(checkpoint), 'model_state_sha256': before, 'plan_sha256': sha(STORE / 'plan.json'), 'input_sha256': PLAN['inputs_sha256'], 'seconds_current_attempt': time.monotonic() - begin, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(), 'development_gt_reads': 0})
print(json.dumps({'event': 'prediction_complete', 'epoch': epoch, 'frames': len(dataset), 'failures': failures}), flush=True)
