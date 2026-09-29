import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .async_runtime import read_base,restore_training,publish_job,wait_for_capacity
from .cli import training_loss
from .io import load_arrays,candidate_indices
from .mapping import sha256
from .model import RayBundleCorrespondence
from .training_runtime import TrainingBudget,atomic_json,move,progress,save_checkpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    work=output/'training'
    work.mkdir(exist_ok=False)
    budget=TrainingBudget(work)
    settings,prepared,binding=read_base(args.base)
    migration=json.loads((output/'migration.json').read_text())
    resume=Path(migration['resume_checkpoint'])
    if sha256(resume)!=migration['resume_sha256']:
        raise ValueError('resume checkpoint changed')
    state=torch.load(resume,map_location='cpu')
    if state['binding']!=binding or state['epoch']!=migration['completed_epoch'] or state['total_updates']!=state['epoch']*300:
        raise ValueError('resume state or update count differs')
    maps,metadata={},{}
    for name in ('even','odd'):
        value,meta=load_arrays(prepared['maps'][name])
        maps[name]=move(value,'cuda')
        metadata[name]=meta
    labels=json.loads((Path(args.base)/'training_labels.json').read_text())
    noise=json.loads((Path(args.base)/'noise.json').read_text())
    model=RayBundleCorrespondence().double().cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'])
    rng=np.random.RandomState(20)
    restore_training(state,model,optimizer,rng)
    atomic_json(work/'resumed.json',dict(epoch=state['epoch'],total_updates=state['total_updates'],
                resume_sha256=migration['resume_sha256'],optimizer_restored=True,rng_restored=True,binding=binding))
    for epoch in range(state['epoch']+1,settings['epochs']+1):
        wait_for_capacity(output,budget)
        model.train()
        optimizer.zero_grad()
        total_loss=0.
        updates=0
        start=time.monotonic()
        for position,index in enumerate(rng.permutation(1200)):
            budget.check()
            frame,meta=load_arrays(prepared['frames']['train'][index],online=True)
            frame['independent_sigma']=torch.tensor(noise['independent_sigma'],dtype=torch.float64)
            frame['shared_covariance']=torch.tensor(noise['shared_covariance'],dtype=torch.float64)
            frame=move(frame,'cuda')
            name='odd' if index%2==0 else 'even'
            ids=candidate_indices(model,frame,maps[name],meta,metadata[name])
            truth=torch.tensor(labels[meta['frame_key']]['pose'],dtype=torch.float64,device='cuda')
            loss=training_loss(model,frame,maps[name],ids,truth)
            (loss/4).backward()
            total_loss+=float(loss.detach())
            if (position+1)%4==0:
                gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                if not torch.isfinite(gradient):
                    raise FloatingPointError('nonfinite training gradient')
                optimizer.step()
                optimizer.zero_grad()
                updates+=1
                if updates==1 or updates%25==0:
                    progress(work,phase='training',epoch=epoch,frames=position+1,total=1200,updates=updates,
                             loss=total_loss/(position+1),gradient=float(gradient),gpu_reserved_mib=torch.cuda.memory_reserved()/2**20)
        checkpoint=output/'checkpoints'/('{:02d}.pt'.format(epoch))
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,total_updates=epoch*300,
                   numpy_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                   map_sha256=binding['map_sha256']['full'],binding=binding,async_migration_sha256=sha256(output/'migration.json'))
        digest=save_checkpoint(checkpoint,state,budget)
        stats=dict(epoch=epoch,train_frames=1200,updates=updates,train_loss=total_loss/1200,seconds=time.monotonic()-start)
        atomic_json(work/'latest_state.json',dict(epoch=epoch,path=str(checkpoint),sha256=digest,total_updates=epoch*300))
        publish_job(output,epoch,checkpoint,digest,stats)
        with (work/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(stats,checkpoint_sha256=digest,validation='queued'))+'\n')
        progress(work,phase='epoch-queued-for-validation',epoch=epoch)
    atomic_json(output/'training_done.json',dict(epochs=settings['epochs'],status='training-completed-validation-may-remain'))


if __name__=='__main__':
    main()
