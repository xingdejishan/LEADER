import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time


ROOT = Path('/home/zhangcy/projects/leader-vera-main-20261002/cam5-repair-v2')
STORE = ROOT / 'storage/trial-20261003-v1'
REFERENCE_ROOT = ROOT.parent / 'cam5-support-fov-v1'
spec = importlib.util.spec_from_file_location('repair_trial_reference_common', REFERENCE_ROOT / 'common.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
PLAN = json.loads((STORE / 'plan.json').read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            digest.update(data)
    return digest.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def output_bytes():
    return sum((Path(folder) / name).stat().st_size for folder, _, files in os.walk(ROOT / 'storage', followlinks=False) for name in files if not (Path(folder) / name).is_symlink())


def guard():
    assert os.getuid() == 1003 and sys.prefix == '/home/zhangcy/.conda/envs/leader'
    available = next(int(row.split()[1]) * 1024 for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:'))
    assert available > PLAN['ram_floor_gib'] * 1024 ** 3, 'system available RAM floor'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < PLAN['rss_limit_gib'] * 1024 ** 3, 'process RSS ceiling'
    assert shutil.disk_usage(STORE).free > PLAN['disk_floor_gib'] * 1024 ** 3, 'disk free floor'


def verify_source():
    reference.verify_source()
    manifest = json.loads((STORE / 'deployment.json').read_text())
    assert sha(STORE / 'plan.json') == manifest['plan_sha256']
    assert sha(STORE / 'inputs.json') == PLAN['inputs_sha256']
    for name, digest in manifest['files'].items():
        assert sha(ROOT / name) == digest, name
    baseline = PLAN['leader_finetune_baseline']
    for key, digest_key in (('prediction', 'prediction_sha256'), ('freeze_path', 'freeze_sha256'), ('evaluation_path', 'evaluation_sha256')):
        assert sha(baseline[key]) == baseline[digest_key]


def gpu_setup():
    import torch
    assert int(os.environ['CUDA_VISIBLE_DEVICES']) in PLAN['allowed_gpus']
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(PLAN['allocator_fraction'])
    guard()
    verify_source()
    reference.prohibit_dev_gt()
    return torch.device('cuda:0')


def pace(start):
    import torch
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    time.sleep(max(0., elapsed * (1 / PLAN['shared_work_fraction'] - 1)))


def optimizer_for(system):
    import torch
    return torch.optim.Adam([
        {'params': system.tokens.stem.parameters(), 'lr': PLAN['rgb_lr'], 'name': 'rgb'},
        {'params': [p for n, p in system.tokens.named_parameters() if not n.startswith('stem.')], 'lr': PLAN['lr'], 'name': 'visual_projection'},
        {'params': system.repair.parameters(), 'lr': PLAN['lr'], 'name': 'endpoint_quality'}
    ])


def load_modules(system, payload):
    assert payload['plan_sha256'] == sha(STORE / 'plan.json')
    assert payload['baseline_checkpoint_sha256'] == PLAN['checkpoint_sha256']
    system.tokens.load_state_dict(payload['tokens'], strict=True)
    system.repair.load_state_dict(payload['repair'], strict=True)
