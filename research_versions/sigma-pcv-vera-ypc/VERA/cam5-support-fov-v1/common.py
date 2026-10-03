import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys

ROOT=Path('/home/zhangcy/projects/leader-vera-main-20261002/cam5-support-fov-v1')
STORE=ROOT/'storage'
PLAN=json.loads((STORE/'plan.json').read_text())
sys.path.insert(0,str(ROOT/'source'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def atomic(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    os.replace(temp,path)

def guard():
    assert os.getuid()==1003 and sys.prefix=='/home/zhangcy/.conda/envs/leader'
    memory={k:int(v.split()[0])*1024 for k,v in (s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines()) if k in ('MemAvailable',)}
    assert memory['MemAvailable']>PLAN['ram_floor_gib']*1024**3,'system available RAM floor'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<PLAN['rss_limit_gib']*1024**3,'process RSS ceiling'
    assert shutil.disk_usage(STORE).free>PLAN['disk_floor_gib']*1024**3,'disk free floor'

def verify_source():
    deployment=json.loads((STORE/'deployment.json').read_text())
    assert sha(STORE/'plan.json')==deployment['plan_sha256'],'frozen plan changed'
    for name,digest in PLAN['source_sha256'].items():assert sha(ROOT/'source'/name)==digest,name
    for name,digest in deployment['files'].items():assert sha(ROOT/name)==digest,name
    assert sha(PLAN['checkpoint'])==PLAN['checkpoint_sha256']

def prohibit_dev_gt():
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=os.fsdecode(args[0]).lower()
            if 'evaluator_only' in path or '/test_gt/' in path or ('groundtruth' in path and any(d in path for d in PLAN['dev_dates'])):
                raise PermissionError('development GT denied in worker')
    sys.addaudithook(audit)

def gpu_setup():
    assert not PLAN.get('preparation_only',True),'training protocol not frozen'
    import torch
    torch.set_num_threads(1)
    assert torch.cuda.device_count()==1
    torch.cuda.set_per_process_memory_fraction(PLAN['allocator_fraction'])
    guard();verify_source();prohibit_dev_gt()
    return torch.device('cuda:0')

def pace(start):
    import time
    import torch
    torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    time.sleep(max(0.,elapsed*(1/PLAN['shared_work_fraction']-1)))
