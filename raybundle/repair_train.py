import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .async_runtime import read_base,publish_job,wait_for_capacity
from .io import load_arrays
from .mapping import sha256
from .repaired import RepairedRayBundle,repaired_loss
from .smoke import state_sha
from .training_runtime import TrainingBudget,atomic_json,move,progress,save_checkpoint


def main():
    parser=argparse.ArgumentParser()
    for name in ('base','output','identities'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    output=Path(args.output)
    work=output/'training'
    work.mkdir(exist_ok=False)
    budget=TrainingBudget(work)
    original,prepared,parent_binding=read_base(args.base)
    source=Path(__file__).resolve().parents[1]
    settings=dict(original,variant='ray-bundle-repair-v1',initialization='fresh seed20 adapter; no original trained weights',
        parent_binding=parent_binding,repairs=['geometry-before-topk-and-cavity-refresh','three-soft-one-hard',
        'per-camera-soft-capacity','residual-centered-lidar-anchor-and-exact-null','multi-view-identity-supervision'],
        identity_loss_weight=.1,identity_labels_sha256=sha256(args.identities),
        source_sha256={str(p.relative_to(source)):sha256(p) for p in (source/'raybundle').glob('*.py')},
        limits='finite Gaussian reference landmarks; fixed reference poses; Cam5 only')
    atomic_json(output/'settings.json',settings)
    binding=dict(parent=parent_binding,settings_sha256=sha256(output/'settings.json'),
        identities_sha256=sha256(args.identities),map_sha256=parent_binding['map_sha256'])
    atomic_json(output/'binding.json',binding)
    identities=json.loads(Path(args.identities).read_text())
    labels=json.loads((Path(args.base)/'training_labels.json').read_text())
    noise=json.loads((Path(args.base)/'noise.json').read_text())
    maps={}
    for name in ('even','odd'):
        data,_=load_arrays(prepared['maps'][name])
        maps[name]=move(data,'cuda')
    torch.manual_seed(20)
    rng=np.random.RandomState(20)
    model=RepairedRayBundle().double().cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=original['learning_rate'])
    atomic_json(work/'initialization.json',dict(model_state_sha256=state_sha(model),seed=20,updates=0,
        parent_training_weights_loaded=False,binding=binding))
    for epoch in range(1,original['epochs']+1):
        wait_for_capacity(output,budget)
        model.train()
        optimizer.zero_grad()
        total_loss=0.
        started=time.monotonic()
        for position,index in enumerate(rng.permutation(1200)):
            budget.check()
            frame,meta=load_arrays(prepared['frames']['train'][index],online=True)
            frame['independent_sigma']=torch.tensor(noise['independent_sigma'],dtype=torch.float64)
            frame['shared_covariance']=torch.tensor(noise['shared_covariance'],dtype=torch.float64)
            frame=move(frame,'cuda')
            name='odd' if index%2==0 else 'even'
            identity=identities[meta['frame_key']]
            if identity['map']!=name:
                raise ValueError('identity map differs from held-out reference split')
            confirmed=torch.tensor(identity['identities'],dtype=torch.long,device='cuda')
            truth=torch.tensor(labels[meta['frame_key']]['pose'],dtype=torch.float64,device='cuda')
            loss=repaired_loss(model,frame,maps[name],truth,confirmed)
            (loss/4).backward()
            total_loss+=float(loss.detach())
            if (position+1)%4==0:
                gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                if not bool(torch.isfinite(gradient)):
                    raise FloatingPointError('nonfinite repaired gradient')
                optimizer.step()
                optimizer.zero_grad()
                updates=(position+1)//4
                if updates==1 or updates%25==0:
                    progress(work,phase='training',epoch=epoch,frames=position+1,total=1200,updates=updates,
                        loss=total_loss/(position+1),gradient=float(gradient),gpu_reserved_mib=torch.cuda.memory_reserved()/2**20)
        checkpoint=output/'checkpoints'/('{:02d}.pt'.format(epoch))
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,total_updates=epoch*300,
            numpy_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            map_sha256=binding['map_sha256']['full'],binding=binding)
        digest=save_checkpoint(checkpoint,state,budget)
        stats=dict(epoch=epoch,train_frames=1200,updates=300,train_loss=total_loss/1200,seconds=time.monotonic()-started)
        atomic_json(work/'latest_state.json',dict(epoch=epoch,path=str(checkpoint),sha256=digest,total_updates=epoch*300))
        publish_job(output,epoch,checkpoint,digest,stats)
        with (work/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(stats,checkpoint_sha256=digest,validation='queued'))+'\n')
        progress(work,phase='epoch-queued-for-validation',epoch=epoch)
    atomic_json(output/'training_done.json',dict(epochs=original['epochs'],status='training-completed-validation-may-remain'))


if __name__=='__main__':
    main()
