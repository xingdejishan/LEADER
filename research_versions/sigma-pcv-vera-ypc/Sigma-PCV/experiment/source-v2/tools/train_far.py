import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from models.far import ForwardAppearanceRegistration,registration_loss
from tools.far_runtime import budget,setup,make_dataset,load_frame,observation,verify_sources,save_checkpoint,report
from tools.magic_nclt_runtime import safe_json_dump,sha256_file,state_dict_sha256


def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(cpu(v) for v in value)
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    output=Path(args.output)
    output.mkdir(exist_ok=not args.smoke)
    assert not (output/'initialization.json').exists()
    setup(output)
    verify_sources(config)
    device=torch.device('cuda:0')
    dataset=make_dataset(config,'train')
    torch.manual_seed(config['seed'])
    rng=np.random.RandomState(config['seed'])
    generator=torch.Generator().manual_seed(config['seed'])
    model=ForwardAppearanceRegistration().to(device)
    model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    center=torch.tensor(torch.load(config['base_checkpoint'],map_location='cpu')['center_t'],device=device,dtype=torch.float32)
    safe_json_dump(output/'initialization.json',dict(state_sha256=state_dict_sha256(model.state_dict()),parameters=sum(p.numel() for p in model.parameters()),seed=config['seed']))
    if args.smoke:
        order=[0,300,600,900]*2
        steps=2
    else:
        order=None
        steps=None
        dataset.verify_all_input_hashes()
        dataset.validate_all_sam_embeddings()
        (output/'checkpoints').mkdir(exist_ok=True)
        (output/'queue').mkdir(exist_ok=True)
        (output/'results').mkdir(exist_ok=True)
        safe_json_dump(output/'settings.json',config)
    for epoch in range(1,2 if args.smoke else config['epochs']+1):
        if not args.smoke:
            while len([p for p in (output/'queue').glob('*.json') if not (output/'results'/p.name).exists()])>=2:
                if (output/'validation_failed.json').exists():raise RuntimeError('validation failed')
                report(output,phase='waiting-for-validation',epoch=epoch)
                budget(output)
                time.sleep(5)
        started=time.monotonic()
        indices=order if args.smoke else rng.permutation(1200).tolist()
        optimizer.zero_grad(set_to_none=True)
        total_loss=0
        count=0
        component_norms={}
        for position,index in enumerate(indices):
            budget(output)
            if not args.smoke and (output/'validation_failed.json').exists():raise RuntimeError('validation failed')
            sample,points,pixels,rgb=load_frame(dataset,index,device)
            if len(points)<6:raise ValueError('insufficient visible points in training frame')
            observed=observation(model,sample,pixels,rgb,device)
            truth=torch.as_tensor(sample['T_WB'],device=device).clone()
            truth[:3,3]-=center
            all_points=torch.from_numpy(sample['points']).to(device)
            loss,metrics=registration_loss(model,points,all_points,observed,truth,generator)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
            (loss/config['accumulation']).backward()
            total_loss+=float(loss.detach())
            count+=1
            if (position+1)%config['accumulation']==0:
                gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),10)
                if not torch.isfinite(gradient):raise FloatingPointError('nonfinite gradient')
                if args.smoke:
                    for name in ('detail','descriptor','field'):
                        component_norms[name]=sum(float(p.grad.norm()) for p in getattr(model,name).parameters() if p.grad is not None)
                    component_norms['temperature']=float(model.raw_temperature.grad.abs())
                    component_norms['sigma']=float(model.raw_sigma.grad.abs())
                    assert all(value>0 and np.isfinite(value) for value in component_norms.values()),component_norms
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates=(position+1)//config['accumulation']
                if updates==1 or updates%25==0 or args.smoke:
                    report(output,phase='smoke' if args.smoke else 'training',epoch=epoch,frames=position+1,
                        updates=updates,total=1200,loss=total_loss/count,visible_points=len(points),
                        gradient=float(gradient),components=metrics,component_gradients=component_norms,
                        gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        if args.smoke:
            model.eval()
            with torch.no_grad():
                observed=observation(model,sample,pixels,rgb,device)
                predicted,_,_=model(points,observed,truth)
                assert torch.isfinite(predicted).all()
            safe_json_dump(output/'smoke_passed.json',dict(steps=steps,frames=8,discarded_weights=True,
                components=component_norms,seconds=time.monotonic()-started,gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
            return
        checkpoint=output/'checkpoints'/('{:02d}.pt'.format(epoch))
        digest=save_checkpoint(checkpoint,dict(model=cpu(model.state_dict()),epoch=epoch,config=config,config_sha256=sha256_file(args.config)))
        complete=dict(model=cpu(model.state_dict()),optimizer=cpu(optimizer.state_dict()),epoch=epoch,
            numpy_rng=rng.get_state(),generator_rng=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),config=config)
        latest_sha=save_checkpoint(output/'latest.pt',complete)
        job=dict(epoch=epoch,checkpoint=str(checkpoint),checkpoint_sha256=digest,settings_sha256=sha256_file(output/'settings.json'))
        safe_json_dump(output/'latest_state.json',dict(epoch=epoch,checkpoint_sha256=digest,latest_sha256=latest_sha,total_updates=epoch*300))
        safe_json_dump(output/'queue'/('{:02d}.json'.format(epoch)),job)
        stats=dict(epoch=epoch,frames=count,loss=total_loss/count,seconds=time.monotonic()-started,checkpoint_sha256=digest)
        with (output/'history.jsonl').open('a') as stream:stream.write(json.dumps(stats)+'\n')
        report(output,phase='epoch-queued-for-validation',**stats)
    safe_json_dump(output/'training_done.json',dict(epochs=config['epochs']))


if __name__=='__main__':main()
