import json
import math
import os
from pathlib import Path
import sys
import time

from common import PLAN,ROOT,STORE,atomic,gpu_setup,guard,pace,sha,verify_source
device=gpu_setup()
import numpy as np
import torch
from runtime import Dataset,System,loss,state_dict_sha256

end=int(sys.argv[1]);assert end in PLAN['evaluate_epochs']
torch.manual_seed(PLAN['seed']);np.random.seed(PLAN['seed'])
system=System('vera',device);dataset=Dataset('train');out=STORE/'vera';out.mkdir(exist_ok=True)
groups=[{'params':system.leader.decoder.parameters(),'lr':PLAN['lr'],'name':'mmregressor'},{'params':system.leader.encoder.decoders.parameters(),'lr':PLAN['rpge_decoder_lr'],'name':'rpge_decoder'},{'params':system.vera.parameters(),'lr':PLAN['lr'],'name':'vera'}]
optimizer=torch.optim.Adam(groups)
latest=out/'latest.pt';history=[];partial=[];next_step=0;start_epoch=1;updates=0
binding=json.loads((STORE/'parent-binding.json').read_text())
assert binding['parent_plan_sha256']==PLAN['parent_plan_sha256']
parent=Path(binding['checkpoint']);assert sha(parent)==binding['checkpoint_sha256']
value=torch.load(latest if latest.exists() else parent,map_location='cpu')
system.load_state_dict(value['model'],strict=True)
optimizer.load_state_dict(value['optimizer'])
torch.set_rng_state(value['torch_rng']);torch.cuda.set_rng_state(value['cuda_rng'])
if latest.exists():
    assert value['plan_sha256']==sha(STORE/'plan.json') and value['parent_checkpoint_sha256']==sha(parent)
    assert value['parent_binding_sha256']==sha(STORE/'parent-binding.json')
    start_epoch=value['resume_epoch'];next_step=value['next_step'];history=value['history'];partial=value['partial'];updates=value['updates']
else:
    assert value['epoch']==binding['selected_epoch'] and value['next_step']==0 and not value['partial']
    assert value['plan_sha256']==binding['parent_plan_sha256']
    updates=value['updates']
del value
frozen_sha=state_dict_sha256({k:v for k,v in system.leader.encoder.state_dict().items() if not k.startswith('decoders.')})
parent_sha=sha(parent);plan_sha=sha(STORE/'plan.json');begin=time.monotonic()
atomic(out/'initialization.json',{'parent_checkpoint_sha256':parent_sha,'parent_epoch':binding['selected_epoch'],'parent_binding_sha256':sha(STORE/'parent-binding.json'),'epoch_semantics':PLAN['epoch_semantics'],'optimizer_transition':PLAN['optimizer_transition'],'active_source_uses_measured_support':True,'architecture_zero_feature_identity_retained':True,'trained_initialization_zero_residual':False,'initial_model_state_sha256':binding['model_state_sha256'],'start_epoch':start_epoch,'resume_step':next_step,'plan_sha256':plan_sha})
def save(resume_epoch,step,metrics,epoch=None):
    guard()
    state={k:v.detach().cpu() for k,v in system.state_dict().items()}
    payload={'model':state,'optimizer':optimizer.state_dict(),'resume_epoch':resume_epoch,'next_step':step,'epoch':epoch,'partial':metrics,'history':history,'updates':updates,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'parent_checkpoint_sha256':parent_sha,'parent_binding_sha256':sha(STORE/'parent-binding.json'),'parent_epoch':binding['selected_epoch'],'phase':PLAN['phase'],'plan_sha256':plan_sha,'frozen_prefix_sha256':frozen_sha}
    temp=out/'latest.pt.tmp';torch.save(payload,temp)
    with temp.open('rb') as f:os.fsync(f.fileno())
    os.replace(temp,latest)
    atomic(out/'latest-receipt.json',{'path':str(latest),'sha256':sha(latest),'resume_epoch':resume_epoch,'next_step':step,'updates':updates,'time':time.time()})
    if epoch in PLAN['evaluate_epochs']:
        target=out/('epoch-%02d.pt'%epoch)
        if not target.exists():os.link(latest,target)
        atomic(out/('epoch-%02d-receipt.json'%epoch),{'checkpoint_sha256':sha(target),'epoch':epoch,'plan_sha256':plan_sha,'parent_checkpoint_sha256':parent_sha,'history':history,'updates':updates})
for epoch in range(start_epoch,end+1):
    system.mode(PLAN['stage1_epochs']+1);order=np.random.default_rng(PLAN['seed']+1000+epoch).permutation(len(dataset))
    progress=(epoch-1)/max(1,PLAN['epochs']-1)
    scale=PLAN['optimizer_lr_min_fraction']+(1-PLAN['optimizer_lr_min_fraction'])*.5*(1+math.cos(math.pi*progress))
    for group in optimizer.param_groups:group['lr']=(PLAN['rpge_decoder_lr'] if group['name']=='rpge_decoder' else PLAN['lr'])*scale
    optimizer.zero_grad(set_to_none=True);metrics=partial if epoch==start_epoch else [];epoch_begin=time.monotonic()
    first=next_step if epoch==start_epoch else 0
    assert first%PLAN['accumulation']==0
    for step in range(first,len(order)):
        frame_start=time.monotonic();guard()
        batch,extra,record=dataset.load(int(order[step]),True,device)
        output,points,eta,active=system(batch,extra,record,device);value,details=loss(system,output,points,eta,batch,device)
        assert torch.isfinite(value);(value/PLAN['accumulation']).backward()
        details.update({'loss':float(value.detach()),'active':int(active.sum()),'voxels':len(points),'active_eta_mean':float(eta[active].norm(dim=-1).mean().detach()) if active.any() else 0.,'total_shift_active_m':float((points-system.last_source_points)[active].norm(dim=-1).mean().detach()) if active.any() else 0.})
        scales=system.vera.last_visibility_scales.detach()
        details.update({'visibility_backtrack_active_fraction':float((scales[active]<1).float().mean()) if active.any() else 0.,'visibility_zero_active_fraction':float((scales[active]==0).float().mean()) if active.any() else 0.})
        metrics.append(details)
        if (step+1)%PLAN['accumulation']==0:
            assert all(torch.isfinite(p.grad).all() for p in system.parameters() if p.grad is not None)
            if updates%50==0:
                atomic(out/'gradient-evidence.json',{'epoch':epoch,'updates':updates,'groups':{g['name']:sum(float(p.grad.abs().sum()) for p in g['params'] if p.grad is not None) for g in optimizer.param_groups},'vera_parameters':{n:None if p.grad is None else float(p.grad.abs().sum()) for n,p in system.vera.named_parameters()},'active_last_frame':int(active.sum())})
            torch.nn.utils.clip_grad_norm_([p for p in system.parameters() if p.requires_grad],10.)
            optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
            if updates%PLAN['checkpoint_every_updates']==0:save(epoch,step+1,metrics)
        if (step+1)%20==0:
            atomic(out/'progress.json',{'epoch':epoch,'frames':step+1,'total':len(order),'updates':updates,'time':time.time(),'loss':details['loss']})
            print(json.dumps({'event':'train','epoch':epoch,'frames':step+1,'loss':details['loss']}),flush=True)
        pace(frame_start)
    assert state_dict_sha256({k:v for k,v in system.leader.encoder.state_dict().items() if not k.startswith('decoders.')})==frozen_sha
    row={'epoch':epoch,'frames':len(order),'updates':updates,'seconds_current_attempt':time.monotonic()-epoch_begin,'metrics':{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]},'learning_rates':{g['name']:g['lr'] for g in optimizer.param_groups},'frozen_encoder_verified':True}
    history.append(row);atomic(out/'history.json',history);save(epoch+1,0,[],epoch);print(json.dumps({'event':'epoch_complete',**row}),flush=True)
verify_source();atomic(out/('train-to-%02d-complete.json'%end),{'epoch':end,'checkpoint_sha256':sha(out/('epoch-%02d.pt'%end)),'plan_sha256':plan_sha,'seconds':time.monotonic()-begin,'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()})
