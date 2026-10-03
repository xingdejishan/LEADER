import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.far import ForwardAppearanceRegistration,skew,so3_exp


def tukey(square,threshold):
    fraction=(square/(threshold*threshold)).clamp(0,1)
    return threshold*threshold/6*(1-(1-fraction).pow(3))


def jacobian(positions,pivot):
    return torch.cat((-skew(positions-pivot),torch.eye(3,device=positions.device,dtype=positions.dtype).expand(len(positions),3,3)),-1)


def update_pose(pose,increment):
    top=torch.cat((so3_exp(increment[:3])@pose[:3,:3],(pose[:3,3]+increment[3:])[:,None]),-1)
    return torch.cat((top,pose.new_tensor([[0.,0.,0.,1.]])),0)


class CoVolumeFAR(ForwardAppearanceRegistration):
    def __init__(self,point_chunk=256,iterations=3,search_spacings=(.4,.1),tukey_thresholds=(1.2,.6),damping=1e-6):
        super().__init__(point_chunk=point_chunk)
        if len(search_spacings)!=len(tukey_thresholds) or iterations not in (2,3):raise ValueError('invalid shared-volume stages')
        self.iterations=iterations
        self.search_spacings=tuple(search_spacings)
        self.tukey_thresholds=tuple(tukey_thresholds)
        self.damping=damping
        self.raw_structure=nn.Parameter(torch.full((len(search_spacings),),math.log(math.expm1(4.))))
        self.raw_geo_sigma=nn.Parameter(torch.tensor([math.log(math.expm1(value-.01)) for value in search_spacings]))

    def geometry_scales(self,stage):
        return F.softplus(self.raw_structure[stage]),.01+F.softplus(self.raw_geo_sigma[stage])

    def geometry_precision(self,pose,structure,stage):
        strength,sigma=self.geometry_scales(stage)
        local=torch.eye(3,device=pose.device,dtype=pose.dtype)+strength*structure
        rotation=pose[:3,:3]
        covariance=rotation[None]@local@rotation.T[None]
        return torch.linalg.inv(covariance),sigma

    def appearance_logits(self,observation,positions,spacing):
        offsets=self.template.to(positions)*spacing
        temperature=.03+9.97*self.raw_temperature.sigmoid()
        logits=[]
        for start in range(0,len(positions),self.point_chunk):
            query=positions[start:start+self.point_chunk,None]+offsets
            flat=query.reshape(-1,3)
            features=checkpoint(self.field,flat.requires_grad_(True)) if self.training and torch.is_grad_enabled() else self.field(flat)
            features=features.reshape(len(query),len(offsets),32)
            logits.append((features*observation[start:start+len(query),None]).sum(-1)/temperature)
        return torch.cat(logits),offsets

    def correlate_shared(self,observation,points,pose,source,target,structure,visible_ids,stage,appearance=None):
        positions=points@pose[:3,:3].T+pose[:3,3]
        offsets=self.template.to(positions)*self.search_spacings[stage]
        if appearance is None:appearance,offsets=self.appearance_logits(observation,positions,self.search_spacings[stage])
        valid=visible_ids>=0
        geometry=torch.zeros_like(appearance)
        if valid.any():
            ids=visible_ids[valid]
            assert torch.allclose(points[valid],source[ids],atol=1e-6,rtol=0),'visible geometry must retain its exact source identity'
            precision,sigma=self.geometry_precision(pose,structure[ids],stage)
            residual=target[ids]-positions[valid]
            difference=offsets[None]-residual[:,None]
            square=torch.einsum('nki,nij,nkj->nk',difference,precision,difference)
            geometry=geometry.index_copy(0,valid.nonzero().flatten(),-tukey(square,self.tukey_thresholds[stage])/sigma.square())
        probability=(appearance+geometry).softmax(-1)
        mean=probability@offsets
        difference=offsets[None]-mean[:,None]
        covariance=torch.einsum('nk,nki,nkj->nij',probability,difference,difference)
        sigma=.005+.095*self.raw_sigma.sigmoid()
        covariance=covariance+sigma.square()*torch.eye(3,device=points.device,dtype=points.dtype)
        return mean,covariance,appearance,offsets

    def solve_shared(self,points,pose,mean,covariance,source,target,structure,visible_ids,stage):
        dtype=pose.dtype
        current=pose.double()
        observed=points.double()@current[:3,:3].T+current[:3,3]+mean.double()
        visual_precision=torch.linalg.inv(covariance.double())
        outside=torch.ones(len(source),device=pose.device,dtype=torch.bool)
        outside[visible_ids[visible_ids>=0]]=False
        geo_source=source[outside].double()
        geo_target=target[outside].double()
        local_structure=structure[outside].double()
        diagnostics=[]
        for _ in range(self.iterations):
            positions=points.double()@current[:3,:3].T+current[:3,3]
            residual=observed-positions
            distance=torch.einsum('ni,nij,nj->n',residual,visual_precision,residual).clamp_min(1e-20).sqrt()
            information=visual_precision*(2.5/distance).clamp(max=1.)[:,None,None]
            j=jacobian(positions,current[:3,3])
            h=torch.einsum('nia,nij,njb->ab',j,information,j)
            b=torch.einsum('nia,nij,nj->a',j,information,residual)
            inliers=0
            if len(geo_source):
                geo_positions=geo_source@current[:3,:3].T+current[:3,3]
                r=geo_target-geo_positions
                precision,sigma=self.geometry_precision(current,local_structure,stage)
                square=torch.einsum('ni,nij,nj->n',r,precision,r).clamp_min(0)
                weights=(1-square/self.tukey_thresholds[stage]**2).clamp_min(0).square()
                inliers=int((weights.detach()>0).sum())
                information=precision*weights[:,None,None]/sigma.square()
                j=jacobian(geo_positions,current[:3,3])
                h=h+torch.einsum('nia,nij,njb->ab',j,information,j)
                b=b+torch.einsum('nia,nij,nj->a',j,information,r)
            damping=self.damping*h.diag().mean().detach().clamp_min(1.)
            increment=torch.linalg.solve(h+damping*torch.eye(6,device=h.device,dtype=h.dtype),b)
            if not torch.isfinite(increment).all():raise FloatingPointError('nonfinite shared-volume pose update')
            current=update_pose(current,increment)
            diagnostics.append(dict(outside_inliers=inliers,increment_norm=float(increment.detach().norm())))
        return current.to(dtype),diagnostics

    def forward(self,points,observed,initial,source,target,structure,visible_ids,truth=None):
        pose=initial
        losses=[]
        stages=[]
        for stage,spacing in enumerate(self.search_spacings):
            mean,covariance,appearance,offsets=self.correlate_shared(observed,points,pose,source,target,structure,visible_ids,stage)
            coverage=None
            if truth is not None:
                delta=points@truth[:3,:3].T+truth[:3,3]-(points@pose[:3,:3].T+pose[:3,3])
                inside=(delta.detach().abs()<=2*spacing).all(-1)
                coverage=float(inside.float().mean().detach())
                labels=(-((delta.detach()[:,None]-offsets)**2).sum(-1)/(2*(spacing*.5)**2)).softmax(-1)
                losses.append(-(labels[inside]*appearance[inside].log_softmax(-1)).sum(-1).mean() if inside.any() else appearance.sum()*0)
            pose,iterations=self.solve_shared(points,pose,mean,covariance,source,target,structure,visible_ids,stage)
            strength,sigma=self.geometry_scales(stage)
            stages.append(dict(spacing=spacing,threshold=self.tukey_thresholds[stage],structure_strength=float(strength.detach()),geometry_sigma_m=float(sigma.detach()),supervision_fraction=coverage,iterations=iterations))
        return pose,losses,dict(visible_fused=int((visible_ids>=0).sum()),geometry_outside=len(source)-len(torch.unique(visible_ids[visible_ids>=0])),stages=stages)

    def observation_batch(self,sams,rgbs,pixels,resized_hw):
        lengths=[len(value) for value in pixels]
        padded=torch.nn.utils.rnn.pad_sequence(pixels,batch_first=True)
        sample=padded[:,:,None]
        rgb=torch.stack(rgbs)
        scale=padded.new_tensor([[hw[1]/rgb.shape[-1],hw[0]/rgb.shape[-2]] for hw in resized_hw])
        sam_grid=sample*scale[:,None,None]*2/1024-1
        detail=self.detail(rgb)
        rgb_grid=(sample/4+.5)*padded.new_tensor([2/detail.shape[-1],2/detail.shape[-2]])-1
        semantic=F.grid_sample(torch.stack(sams),sam_grid,align_corners=False)[:,:,:,0].transpose(1,2)
        image=F.grid_sample(detail,rgb_grid,align_corners=False)[:,:,:,0].transpose(1,2)
        descriptors=F.normalize(self.descriptor(torch.cat((semantic,image),-1)),dim=-1,eps=1e-8)
        return [value[:length] for value,length in zip(descriptors,lengths)]

    def forward_batch(self,points,observations,initials,sources,targets,structures,visible_ids,truths):
        poses=list(initials)
        losses=[[] for _ in points]
        stages=[[] for _ in points]
        lengths=[len(value) for value in points]
        for stage,spacing in enumerate(self.search_spacings):
            positions=[point@pose[:3,:3].T+pose[:3,3] for point,pose in zip(points,poses)]
            logits,offsets=self.appearance_logits(torch.cat(observations),torch.cat(positions),spacing)
            appearance=logits.split(lengths)
            updated=[]
            for index,(point,observed,pose,source,target,structure,visible,truth) in enumerate(zip(points,observations,poses,sources,targets,structures,visible_ids,truths)):
                mean,covariance,local_logits,_=self.correlate_shared(observed,point,pose,source,target,structure,visible,stage,appearance[index])
                delta=point@truth[:3,:3].T+truth[:3,3]-positions[index]
                inside=(delta.detach().abs()<=2*spacing).all(-1)
                labels=(-((delta.detach()[:,None]-offsets)**2).sum(-1)/(2*(spacing*.5)**2)).softmax(-1)
                losses[index].append(-(labels[inside]*local_logits[inside].log_softmax(-1)).sum(-1).mean() if inside.any() else local_logits.sum()*0)
                refined,iterations=self.solve_shared(point,pose,mean,covariance,source,target,structure,visible,stage)
                stages[index].append(dict(supervision_fraction=float(inside.float().mean().detach()),iterations=iterations))
                updated.append(refined)
            poses=updated
        return poses,[sum(value)/len(value) for value in losses],stages


def registration_loss_covolume(model,points,all_points,observed,truth,initial,source,target,structure,visible_ids):
    pose,correlation,diagnostics=model(points,observed,initial,source,target,structure,visible_ids,truth)
    distance=(all_points@pose[:3,:3].T+pose[:3,3]-(all_points@truth[:3,:3].T+truth[:3,3])).norm(dim=-1)
    huber=F.huber_loss(distance,torch.zeros_like(distance),delta=.2)
    ce=sum(correlation)/len(correlation)
    return ce+huber,dict(correlation=float(ce.detach()),pose=float(huber.detach()),supervision_fractions=[row['supervision_fraction'] for row in diagnostics['stages']],**diagnostics)


def registration_loss_batch(model,points,all_points,observed,truths,initials,sources,targets,structures,visible_ids):
    poses,correlation,stages=model.forward_batch(points,observed,initials,sources,targets,structures,visible_ids,truths)
    losses=[]
    pose_losses=[]
    for raw,pose,truth,ce in zip(all_points,poses,truths,correlation):
        distance=(raw@pose[:3,:3].T+pose[:3,3]-(raw@truth[:3,:3].T+truth[:3,3])).norm(dim=-1)
        huber=F.huber_loss(distance,torch.zeros_like(distance),delta=.2)
        losses.append(ce+huber)
        pose_losses.append(huber)
    coverage=[sum(row[stage]['supervision_fraction'] for row in stages)/len(stages) for stage in range(len(model.search_spacings))]
    return torch.stack(losses).mean(),dict(correlation=float(torch.stack(correlation).mean().detach()),pose=float(torch.stack(pose_losses).mean().detach()),supervision_fractions=coverage,batch=len(points))
