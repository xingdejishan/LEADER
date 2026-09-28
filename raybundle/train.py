import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from .calibration import fit_noise
from .cli import training_loss
from .io import candidate_indices,load_arrays,save_arrays
from .leader import load_leader,encode_frame,L0_SHA
from .mapping import read_records,build_map,extract,sha256
from .model import RayBundleCorrespondence
from .smoke import state_sha
from .training_runtime import TrainingBudget,atomic_json,cpu_state,move,progress,save_checkpoint


def evaluate(adapter, ray_map, map_meta, dataset, paths, noise, output, epoch, checkpoint_sha, args, budget):
    output=Path(output)/'epochs'/('{:02d}'.format(epoch))
    output.mkdir(parents=True,exist_ok=False)
    predictions=output/'predictions.jsonl'
    if adapter is not None:
        adapter.eval()
    started=time.monotonic()
    with predictions.open('x') as stream, torch.no_grad():
        for index,path in enumerate(paths):
            budget.check()
            frame,metadata=load_arrays(path,online=True)
            record=dataset.records[index]
            row=dict(frame_key=record.frame_key,date=record.date,timestamp_us=record.timestamp_us,status='success')
            try:
                if adapter is None:
                    pose=frame['T0']
                else:
                    frame['independent_sigma']=torch.tensor(noise['independent_sigma'],dtype=torch.float64)
                    frame['shared_covariance']=torch.tensor(noise['shared_covariance'],dtype=torch.float64)
                    frame=move(frame,'cuda')
                    candidates=candidate_indices(adapter,frame,ray_map,metadata,map_meta)
                    result=adapter(frame,ray_map,candidates,hard=True)
                    pose=result['pose']
                    row['visual_correspondences']=int((result['selected']>=0).sum())
                row['T_WB_pred']=pose.detach().cpu().tolist()
            except (ValueError,RuntimeError,FloatingPointError) as error:
                if 'out of memory' in str(error).lower() or 'nvrtc' in str(error).lower():
                    raise
                row.update(status='failed',error=str(error),T_WB_pred=np.eye(4).tolist())
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            if (index+1)%100==0:
                progress(args.output,phase='validation',epoch=epoch,frames=index+1,total=len(paths))
        stream.flush()
        os.fsync(stream.fileno())
    freeze=output/'prediction_freeze.json'
    atomic_json(freeze,dict(epoch=epoch,frames=len(paths),ground_truth_read_online=False,checkpoint_sha256=checkpoint_sha,
                           prediction_sha256=sha256(predictions),validation_manifest_sha256=dataset.manifest_hashes,
                           map_sha256=map_meta['sha256'] if map_meta else None,runner_sha256=sha256(__file__)))
    subprocess.run([sys.executable,'-B','-m','raybundle.evaluate','--data-root',args.data_root,'--gt-root',args.gt_root,
                    '--predictions',str(predictions),'--freeze',str(freeze),'--output',str(output/'evaluation.json')],
                   check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1'))
    result=json.loads((output/'evaluation.json').read_text())
    result.update(evaluation_sha256=sha256(output/'evaluation.json'),seconds=time.monotonic()-started)
    return result


def prepare(args, output, budget):
    from data.leader_nclt_manifest import ManifestLEADERDataset,TRAIN_DATES,DEV_DATES
    datasets={split:ManifestLEADERDataset([Path(args.data_root)/date/'manifest.json' for date in dates],args.data_root,split)
              for split,dates in (('train',TRAIN_DATES),('dev',DEV_DATES))}
    model,center=load_leader(args.checkpoint,'cuda')
    original=state_sha(model)
    initial_noise=dict(independent_sigma=.1,shared_covariance=np.diag([.2**2]*3+[np.deg2rad(1.)**2]*3).tolist())
    feature_dir=output/'features'
    feature_dir.mkdir()
    paths={}
    labels={}
    for split,dataset in datasets.items():
        folder=output/'frames'/split
        folder.mkdir(parents=True)
        paths[split]=[]
        for index,record in enumerate(dataset.records):
            budget.check()
            sample=dataset[index]
            if split=='dev' and 'T_WB' in sample:
                raise ValueError('dev input contains GT')
            features=extract(record.image_path,max_features=512,expected_sha=record.image_sha256)
            if split=='train':
                save_arrays(feature_dir/(record.image_sha256+'.npz'),dict(pixels=features[0],descriptors=features[1]),
                            dict(image_sha256=record.image_sha256))
                labels[record.frame_key]=dict(pose=np.asarray(record.T_WB).tolist())
            frame=encode_frame(sample,record,model,center,initial_noise,features=features)
            path=folder/('{:04d}.npz'.format(index))
            save_arrays(path,frame,dict(frame_key=record.frame_key,image_sha256=record.image_sha256,
                       scan_sha256=record.scan_sha256,checkpoint_sha256=L0_SHA,noise_model='placeholder; overwritten by frozen train calibration'))
            paths[split].append(path)
            if (index+1)%50==0:
                progress(output,phase='cache-'+split,frames=index+1,total=len(dataset))
    if state_sha(model)!=original or any(p.grad is not None for p in model.parameters()):
        raise ValueError('frozen LEADER changed')
    del model,center,frame,sample
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(output/'training_labels.json',labels)
    noise=fit_noise(paths['train'],labels)
    noise['input_sha256']=[sha256(p) for p in paths['train']]
    atomic_json(output/'noise.json',noise)
    records=read_records(args.data_root,[Path(args.data_root)/date/'manifest.json' for date in TRAIN_DATES],mapping=True)
    maps={}
    map_dir=output/'maps'
    map_dir.mkdir()
    for name,selected in (('even',records[::2]),('odd',records[1::2]),('full',records)):
        progress(output,phase='map-'+name,reference_frames=len(selected))
        def report(stage,done,total):
            budget.check()
            progress(output,phase='map-'+name,stage=stage,done=done,total=total)
        data,metadata=build_map(selected,feature_cache=feature_dir,progress_callback=report)
        path=map_dir/(name+'.npz')
        save_arrays(path,data,metadata)
        maps[name]=path
        del data,metadata
        gc.collect()
    atomic_json(output/'prepared.json',dict(frames={k:[str(p) for p in v] for k,v in paths.items()},maps={k:str(v) for k,v in maps.items()},
                 train_manifest_sha256=datasets['train'].manifest_hashes,dev_manifest_sha256=datasets['dev'].manifest_hashes,
                 label_sha256=sha256(output/'training_labels.json'),noise_sha256=sha256(output/'noise.json'),leader_unchanged=True))
    return datasets,paths,labels,noise,maps


def main():
    parser=argparse.ArgumentParser()
    for name in ('data-root','gt-root','checkpoint','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--epochs',type=int,default=40)
    args=parser.parse_args()
    output=Path(args.output)
    budget=TrainingBudget(output)
    if (output/'settings.json').exists():
        raise FileExistsError('existing formal run; refusing duplicate execution')
    source=Path(__file__).resolve().parents[1]
    files={str(p.relative_to(source)):sha256(p) for folder in ('raybundle','models','data','utils') for p in (source/folder).rglob('*.py')}
    settings=dict(schema='leader-ray-bundle-train-v1',seed=20,epochs=args.epochs,train_frames=1200,dev_frames=1200,
                  physical_batch=1,gradient_accumulation=4,updates_per_epoch=300,learning_rate=1e-4,optimizer='Adam',
                  learning_rate_schedule='constant',loss='translation squared + 25 rotation Frobenius squared',
                  frozen_leader_sha256=L0_SHA,query_features=64,candidates=4,rounds=3,precision='float64',
                  train_maps='even query uses odd references; odd query uses even references; 600 references each',
                  dev_map='all 1200 training references; no dev reference image',map_track_cap=20000,
                  map_truncation='seed20 shuffled components to avoid date-order truncation',noise='fit once on 1200 train frames only',
                  selection='MPE, MOE and mean normalized to same-code L0; dev subset only',
                  resources=dict(gpu_memory_fraction=.35,rss_soft_gib=10,rss_guard_gib=12,disk_floor_gib=3),
                  data_root=args.data_root,source_sha256=files,initialization='fresh adapter; no functional-smoke weights')
    atomic_json(output/'settings.json',settings)
    datasets,paths,labels,noise,map_paths=prepare(args,output,budget)
    baseline=evaluate(None,None,None,datasets['dev'],paths['dev'],None,output,0,L0_SHA,args,budget)
    if baseline['frames']!=1200 or baseline['failures']:
        raise ValueError('same-code L0 baseline failed')
    atomic_json(output/'baseline.json',baseline)
    map_data={}
    map_meta={}
    for name,path in map_paths.items():
        data,metadata=load_arrays(path)
        map_data[name]=move(data,'cuda')
        map_meta[name]=metadata
    binding=dict(settings_sha256=sha256(output/'settings.json'),prepared_sha256=sha256(output/'prepared.json'),
                 map_sha256={k:v['sha256'] for k,v in map_meta.items()},baseline_sha256=sha256(output/'baseline.json'))
    atomic_json(output/'binding.json',binding)
    torch.manual_seed(20)
    rng=np.random.RandomState(20)
    adapter=RayBundleCorrespondence().double().cuda()
    optimizer=torch.optim.Adam(adapter.parameters(),lr=1e-4)
    best={name:dict(value=float('inf'),epoch=None) for name in ('mpe','moe','combined')}
    best_public={name:None for name in best}
    for epoch in range(1,args.epochs+1):
        adapter.train()
        optimizer.zero_grad()
        total_loss=0.
        valid_updates=0
        started=time.monotonic()
        for position,index in enumerate(rng.permutation(1200)):
            budget.check()
            frame,metadata=load_arrays(paths['train'][index],online=True)
            frame['independent_sigma']=torch.tensor(noise['independent_sigma'],dtype=torch.float64)
            frame['shared_covariance']=torch.tensor(noise['shared_covariance'],dtype=torch.float64)
            frame=move(frame,'cuda')
            name='odd' if index%2==0 else 'even'
            candidates=candidate_indices(adapter,frame,map_data[name],metadata,map_meta[name])
            truth=torch.tensor(labels[metadata['frame_key']]['pose'],dtype=torch.float64,device='cuda')
            loss=training_loss(adapter,frame,map_data[name],candidates,truth)
            (loss/4).backward()
            total_loss+=float(loss.detach())
            if (position+1)%4==0:
                gradient=torch.nn.utils.clip_grad_norm_(adapter.parameters(),10.)
                if not torch.isfinite(gradient):
                    raise FloatingPointError('nonfinite training gradient')
                optimizer.step()
                optimizer.zero_grad()
                valid_updates+=1
                if valid_updates==1 or valid_updates%25==0:
                    progress(output,phase='training',epoch=epoch,frames=position+1,total=1200,updates=valid_updates,
                             loss=total_loss/(position+1),gradient=float(gradient),gpu_reserved_mib=torch.cuda.memory_reserved()/2**20)
        checkpoint=output/'epochs'/('{:02d}.pt'.format(epoch))
        candidate=dict(model=adapter.state_dict(),epoch=epoch,map_sha256=map_meta['full']['sha256'],binding=binding)
        checkpoint_sha=save_checkpoint(checkpoint,candidate,budget)
        result=evaluate(adapter,map_data['full'],map_meta['full'],datasets['dev'],paths['dev'],noise,output,epoch,checkpoint_sha,args,budget)
        values=dict(mpe=result['mpe_mean_m'],moe=result['moe_mean_deg'],
                    combined=.5*(result['mpe_mean_m']/baseline['mpe_mean_m']+result['moe_mean_deg']/baseline['moe_mean_deg']))
        for name,value in values.items():
            if value<best[name]['value'] and result['failures']==0:
                best[name]=dict(value=value,epoch=epoch,checkpoint_sha256=checkpoint_sha)
                best_public[name]=dict(best[name],mpe=result['mpe_mean_m'],moe=result['moe_mean_deg'])
                shutil.copyfile(checkpoint,output/('best_'+name+'.pt'))
        latest=dict(candidate,optimizer=optimizer.state_dict(),best=best_public,numpy_rng=rng.get_state(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),total_updates=epoch*300)
        save_checkpoint(output/'latest.pt',latest,budget)
        record=dict(epoch=epoch,train_frames=1200,updates=valid_updates,train_loss=total_loss/1200,
                    checkpoint_sha256=checkpoint_sha,result=result,best=best_public,seconds=time.monotonic()-started)
        with (output/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        atomic_json(output/'selection.json',best_public)
        progress(output,phase='epoch-complete',epoch=epoch,mpe=result['mpe_mean_m'],moe=result['moe_mean_deg'],failures=result['failures'])
    atomic_json(output/'completion.json',dict(epochs=args.epochs,best=best_public,status='completed',binding=binding))


if __name__=='__main__':
    main()
