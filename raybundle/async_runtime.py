import json
from pathlib import Path
import time

import numpy as np
import torch

from .mapping import sha256
from .training_runtime import atomic_json


def read_base(base):
    base=Path(base)
    binding=json.loads((base/'binding.json').read_text())
    settings=json.loads((base/'settings.json').read_text())
    prepared=json.loads((base/'prepared.json').read_text())
    for name in ('settings','prepared','baseline'):
        if sha256(base/(name+'.json'))!=binding[name+'_sha256']:
            raise ValueError('frozen base binding changed: '+name)
    source=Path(__file__).resolve().parents[1]
    for name,digest in settings['source_sha256'].items():
        if sha256(source/name)!=digest:
            raise ValueError('original runtime code changed: '+name)
    for name,path in prepared['maps'].items():
        if sha256(path)!=binding['map_sha256'][name]:
            raise ValueError('reference map changed')
    if sha256(base/'noise.json')!=prepared['noise_sha256'] or sha256(base/'training_labels.json')!=prepared['label_sha256']:
        raise ValueError('training noise or labels changed')
    return settings,prepared,binding


def restore_training(checkpoint,model,optimizer,rng):
    model.load_state_dict(checkpoint['model'],strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    rng.set_state(checkpoint['numpy_rng'])
    torch.set_rng_state(checkpoint['torch_rng'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])


def publish_job(output,epoch,path,digest,stats):
    output=Path(output)
    if sha256(path)!=digest:
        raise ValueError('checkpoint changed before queue publication')
    destination=output/'queue'/('{:02d}.json'.format(epoch))
    if destination.exists():
        raise FileExistsError(destination)
    atomic_json(destination,dict(epoch=epoch,checkpoint=str(path),checkpoint_sha256=digest,training=stats))


def verified_job(path):
    job=json.loads(Path(path).read_text())
    if sha256(job['checkpoint'])!=job['checkpoint_sha256']:
        raise ValueError('queued checkpoint SHA mismatch')
    return job


def pending_jobs(output):
    output=Path(output)
    return [p for p in sorted((output/'queue').glob('*.json')) if not (output/'results'/p.name).exists()]


def check_worker(output,name):
    descriptor=Path(output)/(name+'_worker.json')
    if descriptor.exists():
        record=json.loads(descriptor.read_text())
        exit_path=Path(record['guard_run'])/'exit.json'
        if exit_path.exists():
            status=json.loads(exit_path.read_text())
            if status['code']!=0 or status['failure']:
                raise RuntimeError(name+' worker stopped: '+json.dumps(status))


def wait_for_capacity(output,budget,capacity=2):
    check_worker(output,'validation')
    while len(pending_jobs(output))>=capacity:
        budget.check()
        check_worker(output,'validation')
        if (Path(output)/'validation_failed.json').exists():
            raise RuntimeError('validation worker failed; checkpoint retained')
        time.sleep(5)
