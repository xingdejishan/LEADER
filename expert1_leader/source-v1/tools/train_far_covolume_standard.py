import argparse,json,math,time
from pathlib import Path

import numpy as np
import torch

from models.far_covolume_standard import CoVolumeStandardFAR as CoVolumeFAR,registration_loss_batch
from tools.far_runtime import verify_sources,save_checkpoint
from tools.far_covolume_standard_runtime import make_dataset,load_frame,InitialCache
from tools.far_covolume_budget import setup,budget
from tools.magic_nclt_runtime import safe_json_dump,sha256_file,state_dict_sha256
from tools.predict_far_covolume_standard import audited_probe
from tools.train_far import cpu


def cosine_lr(step,total,initial,minimum):
    if total<2 or not 0<=step<total:raise ValueError('invalid cosine schedule position')
    return minimum+(initial-minimum)*.5*(1+math.cos(math.pi*step/(total-1)))


def restore(output,model,optimizer,rng,config):
    saved=torch.load(output/'latest.pt',map_location='cpu')
    metadata=json.loads((output/'latest_state.json').read_text())
    assert sha256_file(output/'latest.pt')==metadata['latest_sha256']
    assert saved['config']==config and saved['epoch']==metadata['epoch']
    model.load_state_dict(saved['model'],strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    rng.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    assert all(torch.equal(value.cpu(),saved['model'][name]) for name,value in model.state_dict().items())
    current=optimizer.state_dict()
    assert current['param_groups']==saved['optimizer']['param_groups']
    for key,values in saved['optimizer']['state'].items():
        for name,value in values.items():
            actual=current['state'][key][name]
            assert torch.equal(actual.cpu(),value) if torch.is_tensor(value) else actual==value
    a,b=rng.get_state(),saved['numpy_rng']
    assert a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]
    assert torch.equal(torch.get_rng_state(),saved['torch_rng'])
    assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),saved['cuda_rng']))
    safe_json_dump(output/'resume.json',dict(epoch=saved['epoch'],latest_sha256=metadata['latest_sha256'],model_optimizer_rng_restored=True))
    return saved['epoch']+1


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--one-epoch',action='store_true')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=not args.smoke)
    if args.smoke:assert not (output/'initialization.json').exists() and not (output/'latest.pt').exists()
    setup(output)
    verify_sources(config)
    device=torch.device('cuda:0')
    dataset=make_dataset(config,'train')
    center=torch.as_tensor(torch.load(config['base_checkpoint'],map_location='cpu')['center_t'],device=device,dtype=torch.float32).reshape(3)
    cache=InitialCache(config,dataset,center)
    torch.manual_seed(config['seed'])
    rng=np.random.RandomState(config['seed'])
    model=CoVolumeFAR(**config['far_covolume']).to(device).train()
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    start_epoch=1
    if (output/'latest.pt').exists():
        start_epoch=restore(output,model,optimizer,rng,config)
    else:
        assert config['initialization']=='independent_random_FAR' and not optimizer.state
        assert not any(name.startswith('parent_') for name in config)
        initial_hash=state_dict_sha256(model.state_dict())
        initial_record=dict(state_sha256=initial_hash,seed=config['seed'],initialization='independent_random_FAR',pretrained_FAR_weights_loaded=False,formal_weights_from_smoke=False,optimizer='new_Adam',parameters=sum(p.numel() for p in model.parameters()))
        if (output/'initialization.json').exists():assert json.loads((output/'initialization.json').read_text())==initial_record
        else:safe_json_dump(output/'initialization.json',initial_record)
    if not args.smoke:
        dataset.verify_all_input_hashes()
        dataset.validate_all_sam_embeddings()
    for name in ('checkpoints','queue','results'):(output/name).mkdir(exist_ok=True)
    if (output/'settings.json').exists():assert json.loads((output/'settings.json').read_text())==config
    else:safe_json_dump(output/'settings.json',config)
    batch_size=config['batch_size']
    assert batch_size==8 and config['accumulation']==1 and len(dataset)==1200
    updates_per_epoch=len(dataset)//batch_size
    assert updates_per_epoch==config['updates_per_epoch'] and len(dataset)%batch_size==0
    total_updates=config['epochs']*updates_per_epoch
    end_epoch=2 if args.smoke else min(start_epoch+1,config['epochs']+1) if args.one_epoch else config['epochs']+1
    for epoch in range(start_epoch,end_epoch):
        while not args.smoke and len([job for job in (output/'queue').glob('*.json') if not (output/'results'/job.name).exists()])>=2:
            if (output/'validation_failed.json').exists():raise RuntimeError('covolume validation failed')
            budget(output)
            time.sleep(5)
        indices=[0,1,300,301,600,601,900,901]*2 if args.smoke else rng.permutation(len(dataset)).tolist()
        started=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        total_loss=0.
        coverage=np.zeros(len(config['far_covolume']['search_spacings']))
        for offset in range(0,len(indices),batch_size):
            budget(output)
            if (output/'validation_failed.json').exists():raise RuntimeError('covolume validation failed')
            frames=[]
            for index in indices[offset:offset+batch_size]:
                initial,source,target=cache.frame(dataset.records[index].frame_key,device)
                sample,points,pixels,rgb,visible_ids,structure=load_frame(dataset,index,device,source,config['point_max'])
                if len(points)<6:raise ValueError('too few visible training points')
                truth=torch.as_tensor(sample['T_WB'],device=device).clone()
                truth[:3,3]-=center
                frames.append(dict(points=points,raw=torch.from_numpy(sample['points']).to(device),pixels=pixels,rgb=rgb,
                    sam=sample['sam_features'].to(device=device,dtype=torch.float32),hw=sample['sam_metadata']['resized_size_hw'],
                    initial=initial,source=source,target=target,structure=structure,visible=visible_ids,truth=truth))
            assert len(frames)==batch_size
            def values(name):return [frame[name] for frame in frames]
            observed=model.observation_batch(values('sam'),values('rgb'),values('pixels'),values('hw'))
            loss,metrics=registration_loss_batch(model,values('points'),values('raw'),observed,values('truth'),values('initial'),values('source'),values('target'),values('structure'),values('visible'))
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite covolume loss')
            loss.backward()
            total_loss+=float(loss.detach())*batch_size
            coverage+=np.asarray(metrics['supervision_fractions'])*batch_size
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            if not torch.isfinite(gradient):raise FloatingPointError('nonfinite covolume gradient')
            components={}
            if args.smoke:
                for name in ('detail','descriptor','field'):
                    components[name]=sum(float(parameter.grad.norm()) for parameter in getattr(model,name).parameters() if parameter.grad is not None)
                for name in ('raw_temperature','raw_sigma','raw_structure','raw_geo_sigma','raw_volume_ratio'):
                    components[name]=float(getattr(model,name).grad.norm())
                assert all(value>0 and np.isfinite(value) for value in components.values())
            updates=offset//batch_size+1
            step=(epoch-1)*updates_per_epoch+updates-1
            lr=cosine_lr(step,total_updates,config['learning_rate'],config['minimum_learning_rate'])
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            count=offset+batch_size
            if updates==1 or updates%25==0 or args.smoke:
                progress=dict(phase='smoke' if args.smoke else 'training',epoch=epoch,frames=count,total=len(dataset),updates=updates,
                    batch_size=batch_size,accumulation=1,updates_per_epoch=updates_per_epoch,loss=total_loss/count,learning_rate=lr,gradient=float(gradient),mean_supervision_fractions=(coverage/count).tolist(),
                    components=metrics,component_gradients=components,gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
                safe_json_dump(output/'progress.json',progress)
                print(json.dumps(progress),flush=True)
            del loss,observed,frames
        if args.smoke:
            latest_sha=save_checkpoint(output/'latest.pt',dict(model=cpu(model.state_dict()),optimizer=cpu(optimizer.state_dict()),epoch=0,
                numpy_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),config=config,smoke_only=True))
            safe_json_dump(output/'latest_state.json',dict(epoch=0,latest_sha256=latest_sha,smoke_only=True,actual_updates=2))
            freeze=audited_probe(config,output/'dev_probe',model)
            safe_json_dump(output/'smoke_passed.json',dict(frames=len(indices),updates=2,batch_size=batch_size,accumulation=1,component_gradients=components,gt_free_dev_probe_frames=4,
                external_SC2_calls=freeze['audited_external_SC2_calls'],forbidden_cascade_verified=freeze['legacy_full_pool_and_second_backend_forbidden'],
                gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,discarded_weights=True,seconds=time.monotonic()-started))
            return
        checkpoint=output/'checkpoints'/('%02d.pt'%epoch)
        digest=save_checkpoint(checkpoint,dict(model=cpu(model.state_dict()),epoch=epoch,config=config,config_sha256=sha256_file(args.config)))
        complete=dict(model=cpu(model.state_dict()),optimizer=cpu(optimizer.state_dict()),epoch=epoch,numpy_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),config=config,
            scheduler=dict(name='cosine_per_update',completed_updates=epoch*updates_per_epoch,total_updates=total_updates,last_lr=lr))
        latest_sha=save_checkpoint(output/'latest.pt',complete)
        safe_json_dump(output/'latest_state.json',dict(epoch=epoch,checkpoint_sha256=digest,latest_sha256=latest_sha,total_updates=epoch*updates_per_epoch))
        job=dict(epoch=epoch,checkpoint=str(checkpoint),checkpoint_sha256=digest,settings_sha256=sha256_file(output/'settings.json'))
        safe_json_dump(output/'queue'/('%02d.json'%epoch),job)
        stats=dict(epoch=epoch,frames=1200,loss=total_loss/1200,last_lr=lr,mean_supervision_fractions=(coverage/1200).tolist(),
            seconds=time.monotonic()-started,checkpoint_sha256=digest)
        with (output/'history.jsonl').open('a') as stream:stream.write(json.dumps(stats)+'\n')
        safe_json_dump(output/'progress.json',dict(stats,phase='epoch-queued-for-validation'))
    if not args.smoke and json.loads((output/'latest_state.json').read_text())['epoch']==config['epochs']:
        safe_json_dump(output/'training_done.json',dict(epochs=config['epochs']))


if __name__=='__main__':main()
