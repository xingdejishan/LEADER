import json
import resource
import sys
import time

from common import PLAN, ROOT, STORE, atomic, gpu_setup, guard,sha,verify_source
device=gpu_setup()
import numpy as np
import torch
from runtime import Dataset,System,loss,state_dict_sha256

arm=sys.argv[1];assert arm in ('lidar_finetune','vera')
assert (STORE/'checks.json').exists()
torch.manual_seed(PLAN['seed']);np.random.seed(PLAN['seed'])
system=System(arm,device);dataset=Dataset('train')
out=STORE/arm;out.mkdir(exist_ok=False)
frozen={k:v for k,v in system.leader.encoder.state_dict().items() if not k.startswith('decoders.')}
frozen_sha=state_dict_sha256(frozen);encoder_start=state_dict_sha256(system.leader.encoder.state_dict());decoder_start=state_dict_sha256(system.leader.decoder.state_dict())
groups=[{'params':system.leader.decoder.parameters(),'lr':PLAN['lr'],'name':'mmregressor'},{'params':system.leader.encoder.decoders.parameters(),'lr':PLAN['rpge_decoder_lr'],'name':'rpge_decoder'}]
if system.vera is not None:groups.append({'params':system.vera.parameters(),'lr':PLAN['lr'],'name':'vera'})
optimizer=torch.optim.Adam(groups)
history=[];begin=time.perf_counter();last_state=None
for epoch in range(1,PLAN['epochs']+1):
    system.mode(epoch);order=np.random.default_rng(PLAN['seed']+epoch).permutation(len(dataset))
    optimizer.zero_grad(set_to_none=True);metrics=[];epoch_start=time.perf_counter()
    for step,index in enumerate(order):
        guard()
        batch,extra,record=dataset.load(int(index),arm=='vera',device)
        output,points,eta,active=system(batch,extra,record,device)
        value,details=loss(system,output,points,eta,batch,device)
        assert torch.isfinite(value)
        (value/PLAN['accumulation']).backward()
        if (step+1)%PLAN['accumulation']==0 or step+1==len(order):
            assert all(torch.isfinite(p.grad).all() for p in system.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_([p for p in system.parameters() if p.requires_grad],10.)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
        details.update({'loss':float(value.detach()),'active':0 if active is None else int(active.sum()),'anchor_mean_m':0 if eta is None else float(torch.linalg.vector_norm(points-system.last_source_points,dim=-1).mean().detach())})
        metrics.append(details)
        if (step+1)%20==0:print(json.dumps({'event':'train','arm':arm,'epoch':epoch,'frames':step+1,'total':len(order),'loss':details['loss']}),flush=True)
    current_frozen={k:v for k,v in system.leader.encoder.state_dict().items() if not k.startswith('decoders.')}
    assert state_dict_sha256(current_frozen)==frozen_sha
    if epoch<=PLAN['stage1_epochs']:assert state_dict_sha256(system.leader.encoder.state_dict())==encoder_start
    history.append({'epoch':epoch,'seconds':time.perf_counter()-epoch_start,'frames':len(order),'steps':len(order)//PLAN['accumulation'],'metrics':{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]},'encoder_frozen_verified':epoch<=PLAN['stage1_epochs'],'encoder_stem_and_downsample_frozen_verified':True})
    atomic(out/'history.json',history)
    if system.vera is not None:
        gradients={name:None if p.grad is None else float(p.grad.abs().sum()) for name,p in system.vera.named_parameters()}
    print(json.dumps({'event':'epoch_complete','arm':arm,**history[-1]}),flush=True)
state={k:v.detach().cpu() for k,v in system.state_dict().items()}
state_sha=state_dict_sha256(state)
assert state_dict_sha256(system.leader.decoder.state_dict())!=decoder_start
torch.save({'model':state,'arm':arm,'epoch':PLAN['epochs'],'plan_sha256':sha(STORE/'plan.json'),'state_sha256':state_sha},out/'final.pt')
verify_source()
atomic(out/'training-complete.json',{'epochs':PLAN['epochs'],'frames_per_epoch':len(dataset),'optimizer_steps':PLAN['epochs']*len(dataset)//PLAN['accumulation'],'checkpoint_sha256':sha(out/'final.pt'),'state_sha256':state_sha,'seconds':time.perf_counter()-begin,'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cuda_reserved_peak_bytes':torch.cuda.max_memory_reserved(),'frozen_encoder_prefix_sha256':frozen_sha})
print(json.dumps({'event':'training_complete','arm':arm,'sha256':sha(out/'final.pt')}),flush=True)
