import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

try:
    import resource
except ImportError:
    resource=None

ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime'
V5=ROOT.parent/'v5'
SETTINGS=Path(os.environ.get('V7_PATHS',str(ROOT/'local/paths.json')))
STORE=None
PARENT=RUNTIME


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(name,value):
    path=STORE/name
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)


def budget():
    assert sys.platform.startswith('linux'),'V7 GPU execution requires Linux'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<12*2**30
    assert shutil.disk_usage(STORE).free>8*2**30
    available=int(next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')).split()[1])*1024
    assert available>32*2**30
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    memory=[int(row.split(',')[1]) for row in rows.splitlines() if int(row.split(',')[0])==os.getpid()]
    assert len(memory)<=1 and (not memory or memory[0]<=23500)


def validate():
    global STORE
    settings=read(SETTINGS)
    STORE=Path(settings['output']).resolve()
    assert STORE.stat().st_uid==os.getuid() and not STORE.is_symlink()
    plan=read(STORE/'plan.json')
    assert all(sha(ROOT/name)==digest for name,digest in plan['sources'].items())
    assert sha(plan['initial_checkpoint'])==plan['initial_checkpoint_sha256']
    assert all(sha(settings[name])==plan[name+'_sha256'] for name in ['train_manifest','test_manifest'])
    budget()
    return plan


def cuda(frames):
    import torch
    return [{name:value.to('cuda:0',non_blocking=True) if torch.is_tensor(value) else value for name,value in frame.items()} for frame in frames]


def imports():
    sys.path[:0]=[str(RUNTIME),str(V5/'runtime'),str(V5/'source')]
    import numpy as np
    import torch
    import adaptive_solver
    import selfverify
    import reference_solver
    import mixed_field
    from fused_field import enable as field_enable
    from batched_solver import enable as batch_enable
    from models.far_covolume_standard import CoVolumeStandardFAR
    from tools.magic_nclt_runtime import state_dict_sha256
    torch.set_num_threads(1)
    os.nice(max(0,15-os.nice(0)))
    settings=read(SETTINGS)
    config=read(ROOT/'model.json')
    for name,digest in config['source_sha256'].items():
        assert sha(V5/'source'/name)==digest,name
    config['gt_root']=settings['evaluator_gt']
    def deny_test(config):
        forbidden=Path(config['gt_root']).resolve()
        denied=[]
        def deny(event,args):
            if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
                path=Path(os.fsdecode(args[0])).resolve()
                if path==forbidden or forbidden in path.parents:
                    denied.append(str(path))
                    raise PermissionError('Evaluator GT prohibited during training and prediction')
        sys.addaudithook(deny)
        return denied
    common=SimpleNamespace(torch=torch,np=np,cuda=cuda,selfverify=selfverify,state_dict_sha256=state_dict_sha256,CoVolumeStandardFAR=CoVolumeStandardFAR,field_enable=field_enable,batch_enable=batch_enable,reference_solver=reference_solver,mixed_field=mixed_field,deny_test=deny_test)
    return common,{},config,adaptive_solver,FrameLoader


def model(c,config,state,training):
    instance=c.CoVolumeStandardFAR(**config['far_covolume'])
    instance.load_state_dict(state,strict=True)
    instance=instance.cuda().train(training).requires_grad_(training)
    c.field_enable(instance)
    c.batch_enable(instance)
    c.reference_solver.enable(instance)
    c.mixed_field.enable(instance)
    instance.point_chunk=512
    instance.iteration_spacings=((.4,.4,.4),(.1,.1,.1))
    return instance


class FrameLoader:
    def __init__(self,config,split='train'):
        settings=read(SETTINGS)
        self.path=Path(settings[split+'_manifest']).resolve()
        self.manifest=read(self.path)
        self.split=split
        assert self.manifest['split']==split
        assert self.manifest['frozen_LEADER_sha256']==config['base_checkpoint_sha256']
        self.records=[(row['key'].split('/')[0],index,row['key']) for index,row in enumerate(self.manifest['records'])]
        assert len(self.records)==len({row[2] for row in self.records})=={'train':107339,'test':104732}[split]
        self.datasets={date:None for date in sorted({row[0] for row in self.records})}
        assert set(self.datasets)==set(read(ROOT/'protocol.json')[{'train':'train_dates','test':'test_dates'}[split]])

    def frame(self,index,pin=False):
        import torch
        row=self.manifest['records'][int(index)]
        path=(self.path.parent/row['file']).resolve()
        assert sha(path)==row['sha256']
        frame=torch.load(path,map_location='cpu')
        required={'points','pixels','visible_ids','structure','initial','source','target','rgb','sam','hw','key'}
        assert required<=frame.keys() and frame['key']==row['key']
        if self.split=='train':
            assert {'truth','raw'}<=frame.keys()
        else:
            assert not {'truth','raw','perturbation'}&frame.keys()
            assert 'input_sha256' in frame
        return {name:value.pin_memory() if pin and torch.is_tensor(value) else value for name,value in frame.items()}

    def close(self):
        pass
