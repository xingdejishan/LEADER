import contextlib
import json
import math
import sys
import time
from common import PLAN,STORE,atomic,gpu_setup,guard,pace,sha
device=gpu_setup()
import numpy as np
from scipy.spatial import cKDTree
import cv2
import torch
from runtime import Dataset,System,loss,trr_loss
from fov_visibility_v2 import validity,project,sample_coverage

epoch=int(sys.argv[1]);torch.manual_seed(PLAN['seed']);np.random.seed(PLAN['seed'])
model=System('vera',device);checkpoint=STORE/'vera'/('epoch-%02d.pt'%epoch)
receipt=json.loads((STORE/'vera'/('epoch-%02d-receipt.json'%epoch)).read_text())
assert sha(checkpoint)==receipt['checkpoint_sha256']
value=torch.load(checkpoint,map_location='cpu');model.load_state_dict(value['model'],strict=True);del value
model.eval();data=Dataset('train');rows=[]

def aligned_geometry(record,points):
    with np.load(STORE/'geometry'/(record['key'].replace('/','_')+'.npz'),allow_pickle=False) as saved:
        geo={k:saved[k] for k in saved.files}
    current=points.detach().cpu().numpy()
    if not np.array_equal(geo['points'],current):
        distances,indices=cKDTree(geo['points']).query(current,workers=1)
        assert distances.max()<1e-8 and len(set(indices.tolist()))==len(indices)
        geo={k:v[indices] if k!='eta' else v for k,v in geo.items()}
    assert np.array_equal(geo['points'],current)
    return geo

@contextlib.contextmanager
def zero_branches(names):
    handles=[getattr(model.vera,name).register_forward_hook(lambda m,args,result:torch.zeros_like(result)) for name in names]
    try:yield
    finally:
        for h in handles:h.remove()

def attention_capture(geo):
    chunks=[]
    for ci in range(len(data.inputs[0]['cameras'])):
        ids=np.flatnonzero(geo['active']&(geo['camera']==ci))
        chunks.extend(ids[s:s+PLAN['chunk']] for s in range(0,len(ids),PLAN['chunk']))
    state={'next':0,'rows':[]}
    def key_hook(m,args,result):state['key']=result.detach()
    def position_hook(m,args,result):state['position']=result.detach()
    def query_hook(m,args,result):
        idx=chunks[state['next']];state['next']+=1
        q=result.detach();key=state['key'];position=state['position']
        mask=torch.as_tensor(geo['valid'][idx],device=device)
        image_scores=torch.einsum('nd,nmd->nm',q,key)/8
        position_scores=torch.einsum('nd,nmd->nm',q,position)/8
        total=image_scores+position_scores
        probability=total.masked_fill(~mask,-torch.inf).softmax(-1)
        entropy=-(probability*probability.clamp_min(1e-30).log()).sum(-1)/mask.sum(-1).float().log()
        def span(score):
            return score.masked_fill(~mask,-torch.inf).max(-1).values-score.masked_fill(~mask,torch.inf).min(-1).values
        state['rows'].append(torch.stack((entropy,probability.max(-1).values,span(total),span(image_scores),span(position_scores),q.norm(dim=-1)),dim=-1).cpu().numpy())
    handles=[model.vera.key.register_forward_hook(key_hook),model.vera.position.register_forward_hook(position_hook),model.vera.query.register_forward_hook(query_hook)]
    return state,handles,chunks

def intervention(features,points,extra,record,names):
    with zero_branches(names):
        out,anchor,eta,active=model.from_encoded(features,points,extra,record,device)
    return out,anchor,eta,active

def correspondence_statistics(output,anchor,active,basis,batch):
    pose=batch['T'].to(device)[0]@torch.linalg.inv(batch['T_corr'].to(device)[0])
    target=anchor@pose[:3,:3].T+pose[:3,3]-model.center
    residual=output[:,:3]-target;raw=residual.norm(dim=-1)
    reliability=output[:,3]
    low=(reliability.clamp(-10*math.pi,10*math.pi).atan()*(math.log(10)/math.pi)).exp()
    high=(reliability.atan()*(math.log(10)/math.pi)).exp()
    denominator=output.new_zeros((1,))
    denominator.scatter_add_(0,torch.zeros(len(output),dtype=torch.long,device=output.device),torch.minimum(low,high))
    weights=torch.maximum(low,high)/denominator[0].clamp_min(1e-6)
    count=max(min(50,len(output)),int(.5*len(output)))
    selected=torch.zeros(len(output),dtype=torch.bool,device=output.device)
    selected[reliability.topk(count).indices]=True
    cross=torch.cross(basis[:,:,0],basis[:,:,1],dim=-1);length=cross.norm(dim=-1)
    normal=cross/length[:,None].clamp_min(1e-10)
    normal_valid=active&(length>1e-8)
    body_residual=residual@pose[:3,:3]
    normal_error=(body_residual*normal).sum(-1)
    tangent_error=(body_residual-normal*normal_error[:,None]).norm(dim=-1)
    def average(values,mask):return float(values[mask].mean()) if mask.any() else 0.
    return {'selected_count':count,'selected_active_count':int((selected&active).sum()),'selected_active_fraction':float((selected&active).sum())/count,'fraction_active_selected':float((selected&active).sum())/int(active.sum()) if active.any() else 0.,'active_trr_weight_mass_fraction':float(weights[active].sum()/weights.sum()),'active_trr_contribution':float((weights[active]*raw[active]).sum()),'inactive_trr_contribution':float((weights[~active]*raw[~active]).sum()),'active_raw_mean_m':average(raw,active),'inactive_raw_mean_m':average(raw,~active),'selected_active_raw_mean_m':average(raw,selected&active),'selected_inactive_raw_mean_m':average(raw,selected&~active),'active_u_mean':average(reliability,active),'inactive_u_mean':average(reliability,~active),'active_non_degenerate_normal_count':int(normal_valid.sum()),'active_normal_error_mean_m':average(normal_error.abs(),normal_valid),'active_tangent_error_mean_m':average(tangent_error,normal_valid)}

def visibility_checker(extra,geo):
    def measure(anchor):
        points=anchor.detach().cpu().numpy();total=int(geo['active'].sum());passed=0;outside=0;missing=0;depth_mismatch=0;fov_invalid=0
        for ci,context in enumerate(extra['visibility']):
            idx=np.flatnonzero(geo['active']&(geo['camera']==ci))
            uv,z,inside=project(points[idx],context['camera'])
            coverage_ok=sample_coverage(context['coverage'],uv)>=context['threshold']
            fov_invalid+=int((inside&~coverage_ok).sum());outside+=int((~inside).sum())
            xy=np.rint(np.clip(uv,[0,0],[807,615])).astype(int);depth=context['depth'][xy[:,1],xy[:,0]]
            finite=np.isfinite(depth);close=np.abs(z-depth)<=PLAN['zbuffer_tolerance_m']
            missing+=int((inside&coverage_ok&~finite).sum());depth_mismatch+=int((inside&coverage_ok&finite&~close).sum())
            passed+=int(validity(points[idx],context,PLAN['zbuffer_tolerance_m']).sum())
        assert passed+outside+fov_invalid+missing+depth_mismatch==total
        return {'active':total,'passed_same_candidate_depth_test':passed,'passed_official_fov_and_depth':passed,'outside_or_nonpositive':outside,'outside_official_fov':fov_invalid,'missing_raster_depth':missing,'depth_mismatch':depth_mismatch,'fraction_passed':passed/total if total else 0.}
    return measure

def head_transport_statistics(output, anchor, active, plain, source, batch):
    pose = batch['T'].to(device)[0] @ torch.linalg.inv(batch['T_corr'].to(device)[0])
    base_target = source @ pose[:3, :3].T + pose[:3, 3] - model.center
    target = anchor @ pose[:3, :3].T + pose[:3, 3] - model.center
    required = (anchor - source) @ pose[:3, :3].T
    response = output[:, :3] - plain[:, :3]
    base_raw = (plain[:, :3] - base_target).norm(dim=-1)
    raw = (output[:, :3] - target).norm(dim=-1)
    transported_residual = plain[:, :3] + required - target
    base_residual = plain[:, :3] - base_target
    affine_error = float((required - (target - base_target)).abs().max())
    residual_error = float((transported_residual - base_residual).abs().max())
    bound = float(8 * torch.finfo(output.dtype).eps * (base_target.abs().max() + target.abs().max() + required.abs().max() + plain[:, :3].abs().max() + 1))
    assert affine_error <= bound and residual_error <= bound
    with torch.enable_grad():
        predictions = output[:, :3].detach().clone().requires_grad_(True)
        reliability = output[:, 3].detach().clone().requires_grad_(True)
        value = trr_loss(target.detach(), predictions, reliability, torch.zeros(len(output), device=device, dtype=torch.long))[0]
        mu_gradient, u_gradient = torch.autograd.grad(value, (predictions, reliability))
    low = (reliability.detach().clamp(-10 * math.pi, 10 * math.pi).atan() * (math.log(10) / math.pi)).exp()
    high = (reliability.detach().atan() * (math.log(10) / math.pi)).exp()
    weights = torch.maximum(low, high) / torch.minimum(low, high).sum().clamp_min(1e-6)
    mu_norm = mu_gradient.norm(dim=-1)
    nonzero = raw > 1e-6
    if nonzero.any():
        assert float((mu_norm[nonzero] - weights[nonzero]).abs().max()) < 1e-6
    errors = raw.detach().cpu().double().numpy()
    lo = math.exp(math.atan(-10 * math.pi) * math.log(10) / math.pi)
    hi = math.exp(math.atan(10 * math.pi) * math.log(10) / math.pi)
    ratio = float(errors.mean())
    for iteration in range(50):
        good = errors < ratio
        ideal_weights = np.where(good, hi, lo)
        updated = float((errors * ideal_weights).sum() / ideal_weights.sum())
        if abs(updated - ratio) < 1e-12:
            ratio = updated
            break
        ratio = updated
    assert abs(float((ideal_weights * (errors - ratio)).sum())) < 1e-8
    good = torch.as_tensor(errors < ratio, device=device)
    active_numpy = active.detach().cpu().numpy()
    ideal_weights = ideal_weights / ideal_weights.sum()
    selected = torch.zeros(len(output), device=device, dtype=torch.bool)
    selected[output[:, 3].topk(max(min(50, len(output)), int(.5 * len(output)))).indices] = True
    clipped = output[:, 3] < -10 * math.pi
    def mean(value, mask):
        return float(value[mask].mean()) if mask.any() else 0.
    return {'active_base_raw_original_target_m': mean(base_raw, active), 'active_full_raw_new_target_m': mean(raw, active), 'active_gt_transport_oracle_raw_new_target_m': mean(transported_residual.norm(dim=-1), active), 'active_required_target_motion_m': mean(required.norm(dim=-1), active), 'active_head_response_motion_m': mean(response.norm(dim=-1), active), 'active_target_transport_mismatch_m': mean((response - required).norm(dim=-1), active), 'target_affine_max_error_m': affine_error, 'gt_transport_residual_max_error_m': residual_error, 'target_float_rounding_bound_m': bound, 'trr_output_mu_gradient_active_mass_fraction': float(mu_norm[active].sum() / mu_norm.sum().clamp_min(1e-24)), 'trr_output_u_gradient_active_l1_fraction': float(u_gradient[active].abs().sum() / u_gradient.abs().sum().clamp_min(1e-24)), 'active_u_below_negative_clip_count': int((active & clipped).sum()), 'active_u_below_negative_clip_gradient_mean': mean(u_gradient, active & clipped), 'bounded_reliability_oracle_fixed_mu_loss': ratio, 'bounded_reliability_oracle_active_weight_mass_fraction': float(ideal_weights[active_numpy].sum()), 'active_raw_below_oracle_threshold_count': int((active & good).sum()), 'active_raw_below_oracle_threshold_selected_count': int((active & good & selected).sum()), 'active_raw_below_oracle_threshold_u_mean': mean(output[:, 3], active & good), 'inactive_removed_module_output_max_delta': float((output[~active] - plain[~active]).abs().max()) if (~active).any() else 0.}

with torch.no_grad():
    for date_offset in (0,300,600,900):
        for offset in (0,99,199,299):
            start=time.monotonic();guard();index=date_offset+offset
            batch,extra,record=data.load(index,True,device);features,points=model.encode(batch,device)
            model.from_encoded(features,points,extra,record,device)
            geo=aligned_geometry(record,points)
            state,handles,chunks=attention_capture(geo)
            try:output,anchor,eta,active=model.from_encoded(features,points,extra,record,device)
            finally:
                for h in handles:h.remove()
            assert state['next']==len(chunks)
            _,full_details=loss(model,output,anchor,eta,batch,device)
            visibility_scales=model.vera.last_visibility_scales.detach().clone()
            plain=model.leader.decoder(features);_,plain_details=loss(model,plain,points,None,batch,device)
            _,wrong,_=data.load(date_offset+(offset+150)%300,True,device)
            wrong=dict(extra,images=wrong['images'],sam=wrong['sam'])
            swapped,swapped_anchor,swapped_eta,_=model.from_encoded(features,points,wrong,record,device)
            _,wrong_details=loss(model,swapped,swapped_anchor,swapped_eta,batch,device)
            support=torch.as_tensor(geo['support_points'],device=device)
            basis=torch.as_tensor(geo['basis'],device=device)
            motion=anchor-support
            normal=torch.cross(basis[:,:,0],basis[:,:,1],dim=-1)
            normal=normal/normal.norm(dim=-1,keepdim=True).clamp_min(1e-10)
            row={'key':record['key'],'voxels':len(points),'active':int(active.sum()),'full_trr':full_details['trr'],'disabled_module_trr':plain_details['trr'],'wrong_image_trr':wrong_details['trr'],'image_output_mean_abs_delta':float((swapped-output).abs().mean()),'image_anchor_mean_delta_m':float((swapped_anchor-anchor).norm(dim=-1).mean()),'learned_motion_active_mean_m':float(motion[active].norm(dim=-1).mean()) if active.any() else 0.,'active_eta_mean':float(eta[active].norm(dim=-1).mean()) if active.any() else 0.,'learned_motion_active_normal_max_m':float((motion[active]*normal[active]).sum(-1).abs().max()) if active.any() else 0.}
            for name,val in head_transport_statistics(output,anchor,active,plain,points,batch).items():row['head_'+name]=val
            row['full_visibility_backtrack_active_fraction']=float((visibility_scales[active]<1).float().mean()) if active.any() else 0.
            row['full_visibility_zero_active_fraction']=float((visibility_scales[active]==0).float().mean()) if active.any() else 0.
            for label,result,endpoint in [('full',output,anchor),('wrong_image',swapped,swapped_anchor)]:
                for name,val in correspondence_statistics(result,endpoint,active,basis,batch).items():row[label+'_'+name]=val
            measure=visibility_checker(extra,geo)
            full_visibility=measure(anchor);wrong_visibility=measure(swapped_anchor)
            for label,result in [('full',full_visibility),('wrong_image',wrong_visibility)]:
                for name,val in result.items():row[label+'_anchor_visibility_'+name]=val
            if state['rows']:
                values=np.concatenate(state['rows'])
                assert np.isfinite(values).all()
                for col,name in enumerate(['normalized_entropy','max_probability','full_logit_span','image_logit_span','position_logit_span','query_l2_norm']):
                    row['attention_'+name+'_median']=float(np.median(values[:,col]))
                row['attention_fraction_max_probability_gt_0_99']=float(np.mean(values[:,1]>.99))
            for label,names in [('geometry_only',('key','visual')),('image_only_attention',('position',)),('fixed_support',('query',))]:
                changed,changed_anchor,changed_eta,_=intervention(features,points,extra,record,names)
                _,details=loss(model,changed,changed_anchor,changed_eta,batch,device)
                row[label+'_trr']=details['trr']
                row[label+'_output_mean_abs_delta']=float((changed-output).abs().mean())
                row[label+'_active_anchor_mean_delta_m']=float((changed_anchor-anchor)[active].norm(dim=-1).mean()) if active.any() else 0.
                for name,val in correspondence_statistics(changed,changed_anchor,active,basis,batch).items():row[label+'_'+name]=val
                for name,val in measure(changed_anchor).items():row[label+'_anchor_visibility_'+name]=val
                if label=='fixed_support':
                    row['fixed_support_eta_max']=float(changed_eta.abs().max())
                    assert row['fixed_support_eta_max']<2e-6
            rows.append(row);pace(start)
out=STORE/'diagnostics';out.mkdir(exist_ok=True)
gradient_path=STORE/'vera/gradient-evidence.json'
gradient_snapshot=json.loads(gradient_path.read_text()) if gradient_path.exists() else None
assert gradient_snapshot is not None and gradient_snapshot['epoch']==epoch
atomic(out/('e%02d.json'%epoch),{'diagnostic_version':6,'epoch':epoch,'checkpoint_sha256':sha(checkpoint),'diagnostic_source_sha256':sha(__file__),'train_frames':16,'scope':'fixed training frames, same encoder output, branch interventions are inference perturbations, not separately trained ablations or independent evaluation','geometry_only_definition':'zero image key and visual feature residual; position attention and learned displacement remain','image_only_attention_definition':'zero positional key; image key and visual feature residual remain','fixed_support_definition':'zero query gives symmetric uniform attention and zero learned tangent displacement; image feature and fixed support displacement remain','visibility_scope':'actual learned/intervened anchors use official FOV and candidate depth admissibility with piecewise backtracking; not certified physical scene visibility; zero scale may cut displacement gradient','head_transport_scope':'Same actual GPU encoding and trained parameters; removed-module base is OOD, not LiDAR control retraining. True-rotation transport and fixed-mu optimal reliability are Train-only offline oracle diagnostics, not deployable predictions or pose accuracy. Output mu/u gradients only, no parameter update; negative clipped u is penalized rather than unlimited confidence escape.','correspondence_scope':'u ranking matches unchanged backend top-half/min50 pool; single-frame TRR weights match frozen loss; active normal/tangent residuals refer to fitted PCA plane, not certified scene surface','training_gradient_snapshot':gradient_snapshot,'training_gradient_snapshot_sha256':sha(gradient_path),'geometry_rows_aligned_to_current_encoder_points':True,'rows':rows,'means':{k:float(np.mean([r[k] for r in rows if k in r])) for k in rows[0] if k!='key'}})
print(json.dumps({'diagnostic_complete':True,'diagnostic_version':6,'epoch':epoch,'frames':16}))
