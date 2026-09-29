import argparse
import json
from pathlib import Path
import shutil
import time

import torch

from .async_runtime import read_base,verified_job,pending_jobs,check_worker
from .io import load_arrays
from .mapping import sha256
from .model import RayBundleCorrespondence
from .train import evaluate
from .training_runtime import TrainingBudget,atomic_json,move,progress


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--gt-root',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    work=output/'validation'
    work.mkdir(exist_ok=False)
    budget=TrainingBudget(work)
    settings,prepared,binding=read_base(args.base)
    from data.leader_nclt_manifest import ManifestLEADERDataset,DEV_DATES
    dataset=ManifestLEADERDataset([Path(settings['data_root'])/d/'manifest.json' for d in DEV_DATES],settings['data_root'],'dev')
    data,map_meta=load_arrays(prepared['maps']['full'])
    ray_map=move(data,'cuda')
    model=RayBundleCorrespondence().double().cuda()
    noise=json.loads((Path(args.base)/'noise.json').read_text())
    baseline=json.loads((Path(args.base)/'baseline.json').read_text())
    eval_args=argparse.Namespace(output=str(work),data_root=settings['data_root'],gt_root=args.gt_root)
    best=None
    progress(work,phase='ready-for-frozen-checkpoints',map_sha256=map_meta['sha256'])
    while True:
        budget.check()
        if best is None and (output/'migration.json').exists():
            migration=json.loads((output/'migration.json').read_text())
            best=json.loads((Path(args.base)/'selection.json').read_text())
            for name,record in best.items():
                if record is not None:
                    original=Path(args.base)/('best_'+name+'.pt')
                    if sha256(original)!=record['checkpoint_sha256']:
                        raise ValueError('previous best checkpoint changed')
                    shutil.copyfile(original,output/('best_'+name+'.pt'))
            atomic_json(output/'selection.json',best)
        jobs=pending_jobs(output)
        if not jobs:
            if (output/'training_done.json').exists():
                atomic_json(output/'completion.json',dict(status='training-and-validation-completed',epochs=settings['epochs'],best=best))
                return
            check_worker(output,'training')
            time.sleep(5)
            continue
        if best is None:
            raise ValueError('validation job appeared before migration state')
        job=verified_job(jobs[0])
        checkpoint=torch.load(job['checkpoint'],map_location='cpu')
        if checkpoint['epoch']!=job['epoch'] or checkpoint['binding']!=binding:
            raise ValueError('validation checkpoint metadata differs')
        model.load_state_dict(checkpoint['model'],strict=True)
        result=evaluate(model,ray_map,map_meta,dataset,prepared['frames']['dev'],noise,work,job['epoch'],job['checkpoint_sha256'],eval_args,budget)
        if sha256(job['checkpoint'])!=job['checkpoint_sha256']:
            raise ValueError('immutable checkpoint changed during validation')
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
        report=dict(epoch=job['epoch'],training=job['training'],checkpoint_sha256=job['checkpoint_sha256'],result=result,best=best)
        with (work/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(report,allow_nan=False)+'\n')
        atomic_json(output/'results'/jobs[0].name,report)
        progress(work,phase='validation-completed',epoch=job['epoch'],mpe=result['mpe_mean_m'],moe=result['moe_mean_deg'])


if __name__=='__main__':
    main()
