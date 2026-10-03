import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from models.sigma_pcv import prior_loss,posterior_nll
from models.far_covolume_standard import registration_loss_covolume
from tools.sigma_runtime import initialize,train_mode,network_row,support_set,image_observations,posterior,backend_target,sc2,cpu,move,save,load,progress,budget,make_dataset
from tools.magic_nclt_runtime import sha256_file,state_dict_sha256,safe_json_dump


def stage1_loss(row,far,pcv):
    L,mean,cov,supported=posterior(row,far,pcv)
    prior=prior_loss(row['mu'],L,row['reliability'],row['truth'],pcv.sigma_min)
    nll=posterior_nll(mean,cov,row['truth'],supported)
    ce,coverage=pcv.ce(far.field,row['mu'],L,row['descriptors'],row['neighbours'],row['support_valid'],row['truth'])
    loss=prior+nll+ce
    return loss,dict(prior=float(prior.detach()),posterior_nll=float(nll.detach()),ce=float(ce.detach()),coverage=coverage,
        supported=int(supported.sum()),points=len(mean),mean_correction_m=float((mean-row['mu']).detach().norm(dim=-1).mean()))


def stage2_loss(row,far,pcv):
    L,mean,cov,supported=posterior(row,far,pcv)
    target=backend_target(row,L,mean,cov)
    loss,metrics=registration_loss_covolume(far,row['points'],row['raw'],row['observed'],row['pose_truth'],row['initial'],row['source'][row['selected']],target,row['structure'],row['visible'])
    return loss,dict(metrics,supported=int(supported.sum()))


def components(leader,far,pcv):
    result={}
    for name,model in [('rpge',leader.encoder),('decoder',leader.decoder),('covariance',pcv.branch),('field',far.field),('descriptor',far.descriptor),('detail',far.detail)]:
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in gradients),name
        result[name]=sum(float(g.norm()) for g in gradients)
    return result


def optimizer_for(leader,far,pcv,config,stage):
    groups=[]
    if stage==1:
        groups.append(dict(params=list(leader.parameters()),lr=config['leader_lr']))
    groups.extend([dict(params=[p for p in far.parameters() if p.requires_grad],lr=config['far_lr']),dict(params=[p for p in pcv.parameters() if p.requires_grad],lr=config['pcv_lr'])])
    return torch.optim.Adam(groups)


def build_row(sample,record,leader,far,device,center,backend=False,truth=False):
    row=network_row(sample,leader,device,center,truth)
    row.update(support_set(sample,row['source']))
    row.update(image_observations(sample,record,row,far,device,backend))
    if backend and truth:
        row['raw']=torch.as_tensor(sample['points'],device=device)
    return row


def analytic_checks(pcv,far,device):
    mu=torch.randn(4,3,device=device,requires_grad=True)
    L=torch.eye(3,device=device).repeat(4,1,1).mul(.4).requires_grad_()
    d=F.normalize(torch.randn(4,32,device=device),dim=-1).requires_grad_()
    neighbours=torch.arange(4,device=device)[:,None].expand(4,8)
    none=torch.zeros(4,8,device=device,dtype=torch.bool)
    mean,cov,supported=pcv.posterior(far.field,mu,L,d,neighbours,none)
    assert torch.equal(mean,mu) and torch.equal(cov,L@L.transpose(-1,-2)) and not supported.any()
    class Constant(torch.nn.Module):
        pass
    old=pcv.appearance
    def flat(field,positions,factors,desc,ids,valid):
        delta=torch.einsum('nij,kj->nki',factors,pcv.grid)
        return torch.zeros(len(factors),125,device=device),delta
    pcv.appearance=flat
    mean,cov,_=pcv.posterior(far.field,mu,L,d,neighbours,~none)
    expected=(pcv.grid[:,0].square()*(-.5*pcv.grid.square().sum(-1)).softmax(-1)).sum()+1/12
    assert (mean-mu).abs().max()<1e-6
    assert torch.allclose(cov,expected*(L@L.transpose(-1,-2)),atol=1e-6)
    pcv.appearance=old
    far.zero_grad(set_to_none=True)
    loss,_=pcv.ce(far.field,mu,L,d,neighbours,~none,mu.detach()+.1)
    loss.backward()
    assert mu.grad is None and L.grad is None and d.grad is not None
    assert far.field.tables.grad is not None and far.field.tables.grad.norm()>0
    far.zero_grad(set_to_none=True)
    pcv.zero_grad(set_to_none=True)
    return dict(no_support_exact=True,flat_likelihood_covariance_ratio=float(expected),ce_coordinate_stop_gradient=True)


def smoke(config,store):
    leader,far,pcv,center,device=initialize(config)
    original_hash=state_dict_sha256(leader.state_dict())
    train_mode(leader,far,pcv,1)
    checks=analytic_checks(pcv,far,device)
    optimizer=optimizer_for(leader,far,pcv,config,1)
    dataset=make_dataset(config['parent_config'],'train')
    dev=make_dataset(config['parent_config'],'dev')
    assert not set(dataset.keys)&set(dev.keys)
    metrics=[]
    for index in (0,300):
        budget(store)
        optimizer.zero_grad(set_to_none=True)
        sample=dataset[index]
        row=build_row(sample,dataset.records[index],leader,far,device,center,truth=True)
        loss,details=stage1_loss(row,far,pcv)
        assert torch.isfinite(loss)
        loss.backward()
        gradients=components(leader,far,pcv)
        assert all(v>0 for v in gradients.values()),gradients
        norm=torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']],1.)
        assert torch.isfinite(norm)
        optimizer.step()
        metrics.append(dict(index=index,loss=float(loss.detach()),gradients=gradients,details=details))
        progress(store,phase='smoke-stage1',**metrics[-1])
        del row,loss
    assert state_dict_sha256(leader.state_dict())!=original_hash
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    train_mode(leader,far,pcv,2)
    row=build_row(sample,dataset.records[300],leader,far,device,center,backend=True,truth=True)
    with torch.no_grad():
        _,mean,_,_=posterior(row,far,pcv)
        row['initial']=sc2(config,row,mean,300)
    loss,details=stage2_loss(row,far,pcv)
    assert torch.isfinite(loss)
    loss.backward()
    gradient=components(leader,far,pcv)
    assert gradient['covariance']>0 and gradient['field']>0 and gradient['rpge']==gradient['decoder']==0,gradient
    progress(store,phase='smoke-stage2',loss=float(loss.detach()),gradients=gradient)
    del row,loss,leader,far,pcv
    torch.cuda.empty_cache()
    leader,far,pcv,center,device=initialize(config)
    leader.requires_grad_(False).eval()
    far.requires_grad_(False).eval()
    pcv.requires_grad_(False).eval()
    baseline={r['frame_key']:r for r in (json.loads(line) for line in Path(config['baseline_predictions']).read_text().splitlines())}
    parity=[]
    with torch.no_grad():
        for index in (0,300,600,900):
            sample=dev[index]
            assert 'T_WB' not in sample
            row=build_row(sample,dev.records[index],leader,far,device,center,backend=True)
            initial=sc2(config,row,row['mu'],index)
            target=row['mu'][row['selected']]
            pose,_,_=far(row['points'],row['observed'],initial,row['source'][row['selected']],target,row['structure'],row['visible'])
            pose[:3,3]+=center
            difference=float((pose-torch.as_tensor(baseline[sample['frame_key']]['T_WB_pred'],device=device)).abs().max())
            assert difference<2e-5,difference
            L,mean,cov,supported=posterior(row,far,pcv)
            assert torch.isfinite(mean).all() and torch.isfinite(cov).all()
            assert torch.linalg.eigvalsh(cov.double()).min()>0
            parity.append(dict(frame_key=sample['frame_key'],baseline_pose_max=difference,points=len(mean),supported=int(supported.sum())))
    record=dict(passed=True,checks=checks,stage1=metrics,stage2_gradients=gradient,baseline_parity=parity,discarded_smoke_weights=True,
        max_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,train_keys=dataset.keys,dev_keys=dev.keys)
    safe_json_dump(store/'smoke_passed.json',record)
    progress(store,phase='smoke_passed',passed=True,max_reserved_mib=record['max_reserved_mib'])


def train(config,store,stage):
    leader,far,pcv,center,device=initialize(config)
    if stage==2:
        load(store/('stage1-%02d.pt'%config['stage1_epochs']),leader,far,pcv,config)
    train_mode(leader,far,pcv,stage)
    frozen_hash=state_dict_sha256(leader.state_dict()) if stage==2 else None
    optimizer=optimizer_for(leader,far,pcv,config,stage)
    dataset=make_dataset(config['parent_config'],'train')
    if stage==1:
        dataset.verify_all_input_hashes()
        dataset.validate_all_sam_embeddings()
    else:
        manifest=json.loads((store/'initial_manifest.json').read_text())
        assert manifest['config_sha256']==config['self_sha256'] and manifest['frame_keys']==dataset.keys
    rng=np.random.RandomState(config['seed']+stage)
    epochs=config['stage%d_epochs'%stage]
    history=[]
    accumulation=config['accumulation']
    parameters=[p for g in optimizer.param_groups for p in g['params']]
    for epoch in range(1,epochs+1):
        started=time.monotonic()
        totals=[]
        optimizer.zero_grad(set_to_none=True)
        for count,index in enumerate(rng.permutation(len(dataset)),1):
            budget(store)
            sample=dataset[int(index)]
            if stage==1:
                row=build_row(sample,dataset.records[index],leader,far,device,center,truth=True)
                loss,metrics=stage1_loss(row,far,pcv)
            else:
                entry=manifest['frames'][int(index)]
                path=store/'initial_cache'/entry['file']
                assert sha256_file(path)==entry['sha256']
                row=move(torch.load(path,map_location='cpu'),device)
                assert row['frame_key']==sample['frame_key']
                row.update(image_observations(sample,dataset.records[index],row,far,device,True))
                truth=torch.as_tensor(sample['T_WB'],device=device,dtype=torch.float32).clone()
                truth[:3,3]-=center
                row.update(pose_truth=truth,raw=torch.as_tensor(sample['points'],device=device))
                loss,metrics=stage2_loss(row,far,pcv)
            assert torch.isfinite(loss),(stage,epoch,count,metrics)
            (loss/accumulation).backward()
            totals.append(float(loss.detach()))
            if count%accumulation==0:
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                assert torch.isfinite(norm),(stage,epoch,count)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if count==accumulation or count%50==0:
                progress(store,phase='stage%d_train'%stage,epoch=epoch,frames=count,total=len(dataset),updates=count//accumulation,
                    loss=float(np.mean(totals)),last_loss=totals[-1],gradient=float(norm),metrics=metrics,seconds=time.monotonic()-started,
                    max_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
            del loss,row
        if stage==2:
            assert state_dict_sha256(leader.state_dict())==frozen_hash
        path=store/('stage%d-%02d.pt'%(stage,epoch))
        digest=save(path,leader,far,pcv,config,stage=stage,epoch=epoch)
        save(store/('stage%d-resume.pt'%stage),leader,far,pcv,config,stage=stage,epoch=epoch,optimizer=cpu(optimizer.state_dict()),numpy_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        history.append(dict(epoch=epoch,frames=len(dataset),updates=len(dataset)//accumulation,loss=float(np.mean(totals)),seconds=time.monotonic()-started,checkpoint_sha256=digest))
        safe_json_dump(store/('stage%d_history.json'%stage),history)
        progress(store,phase='stage%d_epoch_complete'%stage,**history[-1])
    safe_json_dump(store/('stage%d_done.json'%stage),dict(history=history,final_checkpoint_sha256=digest,leader_frozen=stage==2))


@torch.no_grad()
def cache_initial(config,store):
    leader,far,pcv,center,device=initialize(config)
    path=store/('stage1-%02d.pt'%config['stage1_epochs'])
    load(path,leader,far,pcv,config)
    leader.requires_grad_(False).eval()
    far.requires_grad_(False).eval()
    pcv.requires_grad_(False).eval()
    dataset=make_dataset(config['parent_config'],'train')
    directory=store/'initial_cache'
    directory.mkdir(exist_ok=False)
    entries=[]
    for index in range(len(dataset)):
        budget(store)
        sample=dataset[index]
        online={k:v for k,v in sample.items() if k!='T_WB'}
        row=build_row(online,dataset.records[index],leader,far,device,center,backend=True)
        L,mean,cov,supported=posterior(row,far,pcv)
        row['initial']=sc2(config,row,mean,index)
        for key in ('descriptors','observed','points','structure','visible','selected'):
            row.pop(key)
        saved=directory/('%04d.pt'%index)
        torch.save(cpu(row),saved)
        entries.append(dict(index=index,frame_key=row['frame_key'],file=saved.name,sha256=sha256_file(saved)))
        if (index+1)%50==0:
            progress(store,phase='cache_initial',frames=index+1,total=len(dataset))
    safe_json_dump(store/'initial_manifest.json',dict(frames=entries,frame_keys=dataset.keys,manifest_hashes=dataset.manifest_hashes,
        config_sha256=config['self_sha256'],stage1_checkpoint_sha256=sha256_file(path),ground_truth_used_for_prediction=False))


@torch.no_grad()
def predict(config,store):
    leader,far,pcv,center,device=initialize(config)
    checkpoint=store/('stage2-%02d.pt'%config['stage2_epochs'])
    load(checkpoint,leader,far,pcv,config)
    leader.requires_grad_(False).eval()
    far.requires_grad_(False).eval()
    pcv.requires_grad_(False).eval()
    dataset=make_dataset(config['parent_config'],'dev')
    dataset.verify_all_input_hashes()
    dataset.validate_all_sam_embeddings()
    train_keys=set(json.loads((store/'initial_manifest.json').read_text())['frame_keys'])
    assert not train_keys&set(dataset.keys)
    rows=[]
    diagnostics=[]
    started=time.monotonic()
    with (store/'predictions.tmp').open('x') as stream:
        for count,index in enumerate(config['dev_indices'],1):
            budget(store)
            sample=dataset[index]
            assert 'T_WB' not in sample
            row=build_row(sample,dataset.records[index],leader,far,device,center,backend=True)
            L,mean,cov,supported=posterior(row,far,pcv)
            initial=sc2(config,row,mean,index)
            target=backend_target(row,L,mean,cov)
            pose,_,info=far(row['points'],row['observed'],initial,row['source'][row['selected']],target,row['structure'],row['visible'])
            assert torch.isfinite(pose).all()
            pose[:3,3]+=center
            initial[:3,3]+=center
            record=dict(frame_key=sample['frame_key'],date=sample['date'],timestamp_us=int(sample['timestamp_us']),status='success',
                T_WB_pred=pose.cpu().double().tolist(),T_WB_initial=initial.cpu().double().tolist(),
                pcv_supported=int(supported.sum()),pcv_correction_mean_m=float((mean-row['mu']).norm(dim=-1).mean()))
            stream.write(json.dumps(record,allow_nan=False)+'\n')
            rows.append(record['frame_key'])
            diagnostics.append(dict(frame_key=record['frame_key'],**info))
            if count%20==0:
                stream.flush()
                progress(store,phase='predict',frames=count,total=len(config['dev_indices']),seconds=time.monotonic()-started)
    (store/'predictions.tmp').replace(store/'predictions.jsonl')
    safe_json_dump(store/'solver_diagnostics.json',diagnostics)
    safe_json_dump(store/'prediction_freeze.json',dict(frame_keys=rows,frames=len(rows),manifest_hashes=dataset.manifest_hashes,
        config_sha256=config['self_sha256'],checkpoint_sha256=sha256_file(checkpoint),prediction_sha256=sha256_file(store/'predictions.jsonl'),
        baseline_checkpoint_sha256=config['far_checkpoint_sha256'],parent_source_sha256=config['parent_config']['source_sha256'],
        new_source_sha256=config['new_source_sha256'],ground_truth_read_online=False,SC2_calls=len(rows),seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--phase',required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    config['self_sha256']=sha256_file(args.config)
    store=Path(config['store'])
    phases={'smoke':smoke,'stage1':lambda c,s:train(c,s,1),'cache_initial':cache_initial,'stage2':lambda c,s:train(c,s,2),'predict':predict}
    phases[args.phase](config,store)


if __name__=='__main__':
    main()
