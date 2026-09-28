import json
import os
from pathlib import Path
import resource
import shutil
import time

import torch

from .mapping import sha256


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    os.replace(temporary,path)


def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value,dict):
        return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return type(value)(cpu_state(v) for v in value)
    return value


def move(value, device):
    return {k:v.to(device) for k,v in value.items()}


class TrainingBudget:
    def __init__(self, output):
        import sys
        self.output=Path(output).resolve()
        root=Path('/home/zhangcy/projects/leader-ray-bundle-20260928').resolve()
        if sys.prefix!='/home/zhangcy/.conda/envs/leader' or root not in self.output.parents or root.stat().st_uid!=os.getuid():
            raise RuntimeError('training must use the existing environment and private project')
        os.umask(0o077)
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.35)
        self.check()

    def check(self):
        if shutil.disk_usage(self.output.parent).free < 3*2**30:
            raise RuntimeError('disk floor reached')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 10*2**30:
            raise RuntimeError('training RSS budget reached')
        memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(memory['MemAvailable'].split()[0])*1024 < 32*2**30:
            raise RuntimeError('system memory floor reached')


def save_checkpoint(path, value, budget):
    budget.check()
    path=Path(path)
    temporary=path.with_suffix('.tmp')
    torch.save(cpu_state(value),temporary)
    os.replace(temporary,path)
    return sha256(path)


def progress(output, **fields):
    record=dict(time=time.time(),**fields)
    atomic_json(Path(output)/'progress.json',record)
    print(json.dumps(record,allow_nan=False),flush=True)
