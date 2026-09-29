import argparse
import json
from pathlib import Path
import shutil
import time

import torch

from .async_runtime import verified_job,read_base
from .io import load_arrays
from .mapping import sha256
from .train import evaluate
from .training_runtime import TrainingBudget,atomic_json,move,progress


def execute(task,job_path):
    base,output=Path(task['base']),Path(task['output'])
    work=output/'validation'
    work.mkdir(exist_ok=True)
    budget=TrainingBudget(work)
    job=verified_job(job_path)
    if (output/'results'/Path(job_path).name).exists():
        raise FileExistsError('job already evaluated')
    settings,prepared,parent_binding=read_base(base)
    baseline=json.loads((base/'baseline.json').read_text())
    binding=json.loads(Path(task['binding']).read_text())
    if task['model']=='repaired':
        from .repaired import RepairedRayBundle
        model=RepairedRayBundle().double().cuda()
        current=json.loads((output/'settings.json').read_text())
        if sha256(output/'settings.json')!=binding['settings_sha256']:
            raise ValueError('repair settings changed')
    else:
        from .model import RayBundleCorrespondence
        model=RayBundleCorrespondence().double().cuda()
        current=settings
    source=Path(task['source'])
    for name,digest in current['source_sha256'].items():
        if sha256(source/name)!=digest:
            raise ValueError('frozen model source changed: '+name)
    checkpoint=torch.load(job['checkpoint'],map_location='cpu')
    if checkpoint['binding']!=binding or checkpoint['epoch']!=job['epoch']:
        raise ValueError('job binding or epoch differs')
    model.load_state_dict(checkpoint['model'],strict=True)
    data,meta=load_arrays(prepared['maps']['full'])
    if meta['sha256']!=binding['map_sha256']['full']:
        raise ValueError('development reference map changed')
    ray_map=move(data,'cuda')
    from data.leader_nclt_manifest import ManifestLEADERDataset,DEV_DATES
    dataset=ManifestLEADERDataset([Path(settings['data_root'])/d/'manifest.json' for d in DEV_DATES],settings['data_root'],'dev')
    noise=json.loads((base/'noise.json').read_text())
    args=argparse.Namespace(output=str(work),data_root=settings['data_root'],gt_root=task['gt_root'])
    progress(work,phase='validation-started',epoch=job['epoch'],variant=task['name'])
    result=evaluate(model,ray_map,meta,dataset,prepared['frames']['dev'],noise,work,job['epoch'],job['checkpoint_sha256'],args,budget)
    if sha256(job['checkpoint'])!=job['checkpoint_sha256']:
        raise ValueError('checkpoint changed while evaluating')
    best=json.loads((output/'selection.json').read_text())
    values=dict(mpe=result['mpe_mean_m'],moe=result['moe_mean_deg'],
        combined=.5*(result['mpe_mean_m']/baseline['mpe_mean_m']+result['moe_mean_deg']/baseline['moe_mean_deg']))
    for name,value in values.items():
        if result['failures']==0 and (best[name] is None or value<best[name]['value']):
            best[name]=dict(epoch=job['epoch'],value=value,checkpoint_sha256=job['checkpoint_sha256'],
                mpe=result['mpe_mean_m'],moe=result['moe_mean_deg'])
            temporary=output/('best_'+name+'.tmp')
            shutil.copyfile(job['checkpoint'],temporary)
            temporary.replace(output/('best_'+name+'.pt'))
    atomic_json(output/'selection.json',best)
    report=dict(epoch=job['epoch'],training=job['training'],checkpoint_sha256=job['checkpoint_sha256'],result=result,best=best,
        execution=dict(variant=task['name'],source=task['source'],worker_sha256=sha256(__file__),finished=time.time()))
    with (work/'history.jsonl').open('a') as stream:
        stream.write(json.dumps(report,allow_nan=False)+'\n')
    atomic_json(output/'results'/Path(job_path).name,report)
    progress(work,phase='validation-completed',epoch=job['epoch'],mpe=result['mpe_mean_m'],moe=result['moe_mean_deg'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',required=True)
    parser.add_argument('--job',required=True)
    args=parser.parse_args()
    execute(json.loads(Path(args.task).read_text()),args.job)
