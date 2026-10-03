import types
import torch
from torch.nn import functional as F
from models.far_covolume import jacobian,tukey
from models.far import skew,so3_exp

def pad(values):
    return torch.nn.utils.rnn.pad_sequence(values,batch_first=True)

def transform(points,pose):
    return points@pose[:,:3,:3].transpose(-1,-2)+pose[:,None,:3,3]

def jac(points,pivot):
    identity=torch.eye(3,device=points.device,dtype=points.dtype).expand(*points.shape[:-1],3,3)
    return torch.cat((-skew(points-pivot[:,None]),identity),-1)

def equations(positions,pivot,information,residual):
    j=jac(positions,pivot)
    h=torch.einsum('bnia,bnij,bnjc->bac',j,information,j)
    b=torch.einsum('bnia,bnij,bnj->ba',j,information,residual)
    return h,b

def bounded(model,pose,h,b,support,valid,max_displacement):
    diagonal=h.diagonal(dim1=-2,dim2=-1)
    floor=diagonal.mean(-1).detach().clamp_min(1.)*1e-6
    diagonal=torch.maximum(diagonal,floor[:,None])
    increment=torch.linalg.solve(h+model.damping*torch.diag_embed(diagonal),b)
    j=jac(support,pose[:,:3,3])
    displacement=torch.einsum('bnia,ba->bni',j,increment).norm(dim=-1)
    displacement=torch.where(valid,displacement,torch.zeros_like(displacement)).max(-1).values.clamp_min(1e-12)
    scale=(max_displacement/displacement).clamp(max=1.)
    increment=increment*scale[:,None]
    rotation=so3_exp(increment[:,:3])@pose[:,:3,:3]
    translation=pose[:,:3,3]+increment[:,3:]
    top=torch.cat((rotation,translation[:,:,None]),-1)
    result=torch.cat((top,pose[:,3:4]),-2)
    info=torch.stack((scale,displacement*scale,increment.norm(dim=-1)),-1).detach()
    return result,info

def forward_batch(self,points,observations,initials,sources,targets,structures,visible_ids,truths=None):
    lengths=[len(x) for x in points]
    source_lengths=[len(x) for x in sources]
    device=points[0].device
    point=pad(points)
    source=pad(sources)
    target=pad(targets)
    structure=pad(structures)
    pose=torch.stack(initials).double()
    pv=torch.arange(point.shape[1],device=device)[None]<torch.tensor(lengths,device=device)[:,None]
    sv=torch.arange(source.shape[1],device=device)[None]<torch.tensor(source_lengths,device=device)[:,None]
    visible=pad([F.pad(x,(0,point.shape[1]-len(x)),value=-1) for x in visible_ids])
    geo_valid=(visible>=0)&pv
    indices=visible.clamp_min(0)
    selected_target=target.gather(1,indices[:,:,None].expand(-1,-1,3))
    selected_structure=structure.gather(1,indices[:,:,None,None].expand(-1,-1,3,3))
    outside=torch.ones_like(sv)
    for index,ids in enumerate(visible_ids):outside[index,ids[ids>=0]]=False
    outside=outside&sv
    observed=torch.cat(observations)
    s64,t64,struct64=source.double(),target.double(),structure.double()
    warm=[]
    active=torch.ones(len(points),device=device,dtype=torch.bool)
    for threshold in (1.2,.6):
        position=transform(s64,pose)
        residual=t64-position
        square=residual.square().sum(-1)
        weights=(1-square/threshold**2).clamp_min(0).square()*sv
        inliers=(weights.detach()>0).sum(-1)
        active=active&(inliers>=6)
        j=jac(position,pose[:,:3,3])
        h=torch.einsum('bnia,bn,bnic->bac',j,weights,j)
        b=torch.einsum('bnia,bn,bni->ba',j,weights,residual)
        safe_h=torch.where(active[:,None,None],h,torch.eye(6,device=device,dtype=h.dtype)[None])
        safe_b=torch.where(active[:,None],b,torch.zeros_like(b))
        updated,info=bounded(self,pose,safe_h,safe_b,position,sv,threshold)
        pose=torch.where(active[:,None,None],updated,pose)
        warm.append((info,inliers,active.clone()))
    pose=pose.to(initials[0].dtype)
    losses=[]
    stage_diagnostics=[]
    sigma=.005+.095*self.raw_sigma.sigmoid()
    eye3=torch.eye(3,device=device,dtype=point.dtype)
    truth=torch.stack(truths) if truths is not None else None
    for stage,spacing in enumerate(self.search_spacings):
        offsets=self.template.to(point)*spacing
        local_losses=[]
        info_rows=[]
        for iteration in range(self.iterations):
            position=transform(point,pose)
            packed=torch.cat([x[:length] for x,length in zip(position,lengths)])
            logits,_=self.appearance_logits(observed,packed,spacing)
            appearance=pad(logits.split(lengths))
            strength,_=self.geometry_scales(stage)
            local=eye3+strength*selected_structure
            rotation=pose[:,:3,:3]
            covariance=rotation[:,None]@local@rotation.transpose(-1,-2)[:,None]
            precision=torch.linalg.inv(covariance)
            residual=selected_target-position
            difference=offsets[None,None]-residual[:,:,None]
            square=torch.einsum('bnki,bnij,bnkj->bnk',difference,precision,difference)
            _,cap,_=self.volume_scales(stage)
            geometry=-cap*6/self.tukey_thresholds[stage]**2*tukey(square,self.tukey_thresholds[stage])
            geometry=geometry*geo_valid[:,:,None]
            probability=(appearance+geometry).softmax(-1)
            mean=probability@offsets
            difference=offsets[None,None]-mean[:,:,None]
            covariance=torch.einsum('bnk,bnki,bnkj->bnij',probability,difference,difference)+sigma.square()*eye3
            if truth is not None:
                delta=(transform(point,truth)-position).detach()
                inside=(delta.abs()<=2*spacing).all(-1)&pv
                labels=(-((delta[:,:,None]-offsets)**2).sum(-1)/(2*(spacing*.5)**2)).softmax(-1)
                ce=-(labels*appearance.log_softmax(-1)).sum(-1)
                local_losses.append((ce*inside).sum(-1)/inside.sum(-1).clamp_min(1))
                coverage=inside.float().sum(-1)/pv.sum(-1)
            else:coverage=point.new_full((len(points),),-1.)
            current=pose.double()
            position64=transform(point.double(),current)
            precision=torch.linalg.inv(covariance.double())
            residual64=mean.double()
            distance=torch.einsum('bni,bnij,bnj->bn',residual64,precision,residual64).clamp_min(1e-20).sqrt()
            information=precision*(2.5/distance).clamp(max=1.)[:,:,None,None]*pv[:,:,None,None]
            h,b=equations(position64,current[:,:3,3],information,residual64)
            geo_position=transform(s64,current)
            geo_residual=t64-geo_position
            strength,geo_sigma=self.geometry_scales(stage)
            local=torch.eye(3,device=device,dtype=torch.float64)+strength*struct64
            rotation=current[:,:3,:3]
            precision=torch.linalg.inv(rotation[:,None]@local@rotation.transpose(-1,-2)[:,None])
            square=torch.einsum('bni,bnij,bnj->bn',geo_residual,precision,geo_residual).clamp_min(0)
            weights=(1-square/self.tukey_thresholds[stage]**2).clamp_min(0).square()*outside
            information=precision*weights[:,:,None,None]/geo_sigma.square()
            gh,gb=equations(geo_position,current[:,:3,3],information,geo_residual)
            current,info=bounded(self,current,h+gh,b+gb,torch.cat((position64,geo_position),1),torch.cat((pv,outside),1),2*spacing)
            pose=current.to(pose.dtype)
            info_rows.append(torch.cat((info,(weights.detach()>0).sum(-1)[:,None],coverage.detach()[:,None]),-1))
        if truths is not None:losses.append(torch.stack(local_losses).mean(0))
        stage_diagnostics.append(torch.stack(info_rows))
    if not torch.isfinite(pose).all():raise FloatingPointError('nonfinite batched CoVolume update')
    raw_warm=[(info.cpu().tolist(),inliers.cpu().tolist(),active.cpu().tolist()) for info,inliers,active in warm]
    warm_diagnostics=[]
    for index in range(len(points)):
        rows=[]
        for stage,(info,inliers,active) in enumerate(raw_warm):
            if not active[index]:
                rows.append({'threshold':(1.2,.6)[stage],'inliers':inliers[index],'skipped':True})
                break
            rows.append({'step_scale':info[index][0],'max_linear_point_step':info[index][1],'increment_norm':info[index][2],'threshold':(1.2,.6)[stage],'inliers':inliers[index]})
        warm_diagnostics.append(rows)
    raw_stages=torch.stack(stage_diagnostics).cpu().tolist()
    stages=[]
    for index in range(len(points)):
        rows=[]
        for stage,spacing in enumerate(self.search_spacings):
            iterations=[]
            for iteration in range(self.iterations):
                values=raw_stages[stage][iteration][index]
                iterations.append({'step_scale':values[0],'max_linear_point_step':values[1],'increment_norm':values[2],'outside_inliers':int(values[3]),'iteration':iteration,'supervision_fraction':values[4] if truths is not None else None})
            coverage=[x['supervision_fraction'] for x in iterations]
            rows.append({'spacing':spacing,'iterations':iterations,'supervision_fraction':sum(coverage)/len(coverage) if truths is not None else None,'first_supervision_fraction':coverage[0],'last_supervision_fraction':coverage[-1]})
        stages.append(rows)
    correlation=torch.stack(losses).mean(0) if truths is not None else pose.flatten(1).sum(-1)*0
    return list(pose.unbind(0)),list(correlation.unbind(0)),{'warm_start':warm_diagnostics,'stages':stages}

def enable(model):
    model.forward_batch=types.MethodType(forward_batch,model)
    return model
