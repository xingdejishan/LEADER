import json
import resource
import time

from common import PLAN, ROOT, STORE, atomic, gpu_setup, guard,sha
device=gpu_setup()
import numpy as np
import torch
from scipy.spatial import cKDTree
from runtime import Dataset,System,loss,state_dict_sha256

torch.manual_seed(PLAN['seed']);np.random.seed(PLAN['seed'])
data=Dataset('train');dev=Dataset('dev')
base=System('l0',device).eval();new=System('vera',device).eval()
rows=[]
with torch.no_grad():
    for i in (0,300,600,900):
        batch,extra,record=data.load(i,True,device)
        features,p=new.encode(batch,device)
        old=base.leader.decoder(features)
        output,anchor,eta,active=new.from_encoded(features,p,extra,record,device)
        distance,indices=cKDTree(p.cpu().numpy()).query(anchor.cpu().numpy(),workers=1)
        assert len(set(indices.tolist()))==len(indices)
        delta=(output-old[torch.as_tensor(indices,device=device)]).abs().max().item();shift=float(distance.max())
        atomic(STORE/'parity-in-progress.json',{'key':record['key'],'row_order_equal':bool(np.array_equal(indices,np.arange(len(indices)))),'output_max_abs':delta,'source_nearest_max_abs':shift})
        assert delta<=1e-6 and shift<=1e-6,(delta,shift)
        assert active.sum()>0,'No geometrically supported visual anchors'
        rows.append({'key':record['key'],'output_max_abs':delta,'anchor_max_abs':shift,'points':len(p),'active':int(active.sum())})
        batch.pop('T')
        again=new.from_encoded(features,p,extra,record,device)[0]
        assert torch.equal(output,again)
        guard()
new.mode(1)
batch,extra,record=data.load(0,True,device)
output,anchor,eta,active=new(batch,extra,record,device)
value,details=loss(new,output,anchor,eta,batch,device);value.backward()
query_gradient=new.vera.query.weight.grad.abs().sum().item()
assert np.isfinite(query_gradient) and query_gradient>0
assert new.vera.visual.weight.grad.abs().sum()>0
assert new.vera.displacement.weight.grad.abs().sum()>0
assert all(p.grad is None for p in new.leader.encoder.parameters())
assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)
with np.load(STORE/'geometry'/(record['key'].replace('/','_')+'.npz')) as saved:
    g={k:saved[k] for k in saved.files}
basis=torch.as_tensor(g['basis'],device=device);frame=torch.as_tensor(g['frame'],device=device)
e=torch.randn(len(basis),2,device=device,requires_grad=True)
point=torch.as_tensor(g['points'],device=device);a=point+torch.einsum('nij,nj->ni',basis,e)
pose=batch['T'].to(device)[0]
label=a@pose[:3,:3].T+pose[:3,3]-new.center
direction=torch.randn_like(label);(label*direction).sum().backward()
expected=torch.einsum('nij,nj->ni',basis.transpose(1,2),direction@pose[:3,:3])
target_gradient_max=(e.grad-expected).abs().max().item();assert target_gradient_max<2e-6
candidates=point[:,None]+torch.einsum('nij,mj->nmi',basis,torch.as_tensor(g['eta'],device=device))
weights=torch.rand(len(point),49,device=device);weights/=weights.sum(-1,keepdim=True)
a=torch.einsum('nm,nmi->ni',weights,candidates)
l1=a@pose[:3,:3].T+pose[:3,3]-new.center
l2=torch.einsum('nm,nmi->ni',weights,candidates@pose[:3,:3].T+pose[:3,3]-new.center)
affine_error=(l1-l2).abs().max().item();assert affine_error<2e-4
ev=torch.tensor([.7,-.4],device=device).expand(len(point),-1)
d=torch.einsum('nij,nj->ni',basis,ev)
cross=torch.cross(basis[:,:,0],basis[:,:,1],dim=-1)
normal=cross/cross.norm(dim=-1,keepdim=True).clamp_min(1e-10)
normal_error=(normal*d).sum(-1).abs().max().item();assert normal_error<1e-6
assert np.array_equal(g['valid'],g['valid'][:,::-1])
atomic(STORE/'checks.json',{'baseline_parity':rows,'parity_scope':'VERA/MMRegressor conditional on the identical single original-main RPGE encoding; not an end-to-end repeatability claim','original_main_alignment':'unchanged as explicitly requested','original_main_repeatability_issue_preserved':True,'query_gradient_l1':query_gradient,'target_gradient_max_abs':target_gradient_max,'affine_target_max_abs':affine_error,'normal_displacement_max_abs':normal_error,'symmetric_visible_sets':True,'training_gt_only':True,'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
print(json.dumps({'event':'checks_complete','rows':rows,'query_gradient':query_gradient}),flush=True)
