import math

import torch
from torch.nn import functional as F

from .geometry import bearing_linearization,lidar_normal,normal_system,predictive_score,retract
from .model import RayBundleCorrespondence


class RepairedRayBundle(RayBundleCorrespondence):
    def lidar_system(self,frame,pose):
        anchored_target=frame['source'] @ frame['T0'][:3,:3].T+frame['T0'][:3,3]
        return lidar_normal(pose,frame['source'],anchored_target,frame['weights'],
                            float(frame['independent_sigma']),frame['shared_covariance'])

    @torch.no_grad()
    def retrieve(self,frame,ray_map,pose=None,deltas=None,covariances=None):
        pose=frame['T0'] if pose is None else pose.detach()
        n=len(frame['descriptors'])
        if deltas is None:
            h,g=self.lidar_system(frame,pose)
            deltas=-torch.linalg.solve(h,g).expand(n,6)
            covariances=torch.linalg.inv(h).expand(n,6,6)
        query=self.encode(frame['descriptors'])
        reference=self.encode(ray_map['descriptors'])
        covariance=torch.linalg.inv(ray_map['precision'])
        local=torch.ones(len(reference),dtype=torch.bool,device=pose.device)
        if 'reference_centers' in ray_map:
            local=(ray_map['reference_centers']-frame['T0'][:3,3]).norm(dim=-1)<150.+ray_map['reference_radii']
        best=pose.new_full((n,self.max_candidates),-torch.inf)
        identities=torch.full(best.shape,-1,dtype=torch.long,device=pose.device)
        for start in range(0,len(reference),256):
            stop=min(start+256,len(reference))
            if not bool(local[start:stop].any()):
                continue
            xyz=ray_map['mean'][None,start:stop].expand(n,-1,-1)
            origins=frame['origins'][:,None].expand_as(xyz)
            directions=frame['directions'][:,None].expand_as(xyz)
            residual,j,d,visible=bearing_linearization(pose,xyz,origins,directions)
            residual=residual+(j @ deltas[:,None,:,None]).squeeze(-1)
            s=d @ covariance[None,start:stop] @ d.transpose(-1,-2)
            s=s+j @ covariances[:,None] @ j.transpose(-1,-2)
            s=s+self.angular_sigma**2*torch.eye(2,dtype=s.dtype,device=s.device)
            determinant=(s[...,0,0]*s[...,1,1]-s[...,0,1]*s[...,1,0]).clamp_min(1e-30)
            x,y=residual.unbind(-1)
            mahal=(s[...,1,1]*x*x-(s[...,0,1]+s[...,1,0])*x*y+s[...,0,0]*y*y)/determinant
            scores=12*(query @ reference[start:stop].T)-.5*(mahal+determinant.log()+2*math.log(2*math.pi))
            scores=scores.masked_fill(~visible | ~local[None,start:stop],-torch.inf)
            ids=torch.arange(start,stop,device=pose.device).expand(n,-1)
            combined=torch.cat((best,scores),1)
            combined_ids=torch.cat((identities,ids),1)
            best,order=combined.topk(self.max_candidates,dim=1)
            identities=combined_ids.gather(1,order)
        return identities.masked_fill(~torch.isfinite(best),-1)

    def soft_assignment(self,scores,indices,cameras):
        probabilities=F.softmax(scores,-1)
        alpha=probabilities[:,:-1]
        rows=[]
        for camera in torch.unique(cameras):
            mask=cameras==camera
            ids=indices[mask]
            values=alpha[mask]
            count=int(indices.max())+1
            totals=values.new_zeros(max(count,1)).index_add(0,ids.clamp_min(0).reshape(-1),values.reshape(-1))
            scaled=values/totals[ids.clamp_min(0)].clamp_min(1.)
            rows.append((mask,torch.cat((scaled,1.-scaled.sum(-1,keepdim=True)),-1)))
        result=torch.zeros_like(probabilities)
        for mask,value in rows:
            result[mask]=value
        return result

    def state_for(self,ray_map,candidates,previous=None):
        active=torch.unique(candidates[candidates>=0])
        lookup=candidates.new_full((len(ray_map['mean']),),-1)
        lookup[active]=torch.arange(len(active),device=candidates.device)
        indices=torch.where(candidates>=0,lookup[candidates.clamp_min(0)],-1)
        mean=ray_map['mean'][active]
        points=mean.clone()
        alpha=mean.new_zeros(candidates.shape)
        if previous is not None:
            old_candidates,old_active,old_points,old_alpha=previous
            old_lookup=lookup.new_full(lookup.shape,-1)
            old_lookup[old_active]=torch.arange(len(old_active),device=lookup.device)
            retained=old_lookup[active]>=0
            points[retained]=old_points[old_lookup[active[retained]]]
            same=(candidates[:,:,None]==old_candidates[:,None,:]) & (candidates[:,:,None]>=0)
            alpha=(same*old_alpha[:,None,:]).sum(-1)
        return active,indices,mean,ray_map['precision'][active],points,alpha

    def cavities(self,frame,pose,points,mean,precision,indices,alpha):
        origins=frame['origins'][:,None].expand(*indices.shape,3)
        directions=frame['directions'][:,None].expand(*indices.shape,3)
        residual,j,d,visible=bearing_linearization(pose,points[indices.clamp_min(0)],origins,directions)
        h,g=self.lidar_system(frame,pose)
        groups=frame.get('group_ids',torch.arange(len(indices),device=indices.device))
        systems=[]
        for group in torch.unique(groups):
            members=groups==group
            system=normal_system(h,g,points,mean,precision,indices,residual,j,d,alpha*(~members)[:,None],self.angular_sigma**2)
            systems.append((members,system))
        return residual,j,d,visible,h,g,systems

    def forward(self,frame,ray_map,candidates=None,hard=None):
        dummy=torch.full((len(frame['descriptors']),self.max_candidates),-1,dtype=torch.long,device=frame['T0'].device)
        self.validate(frame,ray_map,dummy)
        hard=not self.training if hard is None else bool(hard)
        candidates=self.retrieve(frame,ray_map)
        active,indices,mean,precision,points,alpha=self.state_for(ray_map,candidates)
        pose=frame['T0'].clone()
        cameras=frame.get('camera_ids',torch.zeros(len(candidates),device=candidates.device,dtype=torch.long))
        initial_scores=None
        proposals=[]
        for step in range(self.rounds+int(hard)):
            if len(active)==0:
                probabilities=pose.new_zeros(len(candidates),self.max_candidates+1)
                probabilities[:,-1]=1
                return dict(pose=frame['T0'],probabilities=probabilities,candidates=candidates,
                    selected=dummy[:,0],active_ids=active,points=points,point_covariance=precision,
                    scores=probabilities,initial_scores=probabilities,proposal_history=proposals)
            linear=self.cavities(frame,pose,points,mean,precision,indices,alpha)
            if step:
                deltas=pose.new_zeros(len(indices),6)
                covariance=pose.new_zeros(len(indices),6,6)
                for members,system in linear[-1]:
                    deltas[members]=system[0].detach()
                    covariance[members]=system[2].detach()
                refreshed=self.retrieve(frame,ray_map,pose,deltas,covariance)
                previous=(candidates,active,points,alpha)
                candidates=refreshed
                active,indices,mean,precision,points,alpha=self.state_for(ray_map,candidates,previous)
                if len(active)==0:
                    return self._null_result(frame,ray_map,candidates,proposals)
                linear=self.cavities(frame,pose,points,mean,precision,indices,alpha)
            proposals.append(candidates.detach().clone())
            residual,j,d,visible,h,g,systems=linear
            prior=self.prior(frame['descriptors'],ray_map['descriptors'][active],indices)
            scores=torch.empty_like(prior)
            for members,system in systems:
                value=predictive_score(system,indices[members],residual[members],j[members],d[members],self.angular_sigma**2)
                scores[members,:-1]=prior[members,:-1]+value
            scores[:,-1]=prior[:,-1]-math.log(4*math.pi)
            scores=torch.cat((scores[:,:-1].masked_fill((indices<0)|~visible,-torch.inf),scores[:,-1:]),-1)
            if initial_scores is None:
                initial_scores=scores
            probabilities=self._hard_assignment(scores,indices,cameras) if hard and step==self.rounds else self.soft_assignment(scores,indices,cameras)
            alpha=probabilities[:,:-1]
            system=normal_system(h,g,points,mean,precision,indices,residual,j,d,alpha,self.angular_sigma**2)
            delta,delta_points=system[:2]
            scale=torch.minimum(delta.new_tensor(1.),torch.minimum(1./delta[:3].norm().clamp_min(1e-8),.15/delta[3:].norm().clamp_min(1e-8)))
            pose=retract(pose,scale*delta)
            points=points+scale*delta_points
        chosen=probabilities.argmax(-1)
        selected=torch.where(chosen<indices.shape[1],candidates.gather(1,chosen.clamp_max(indices.shape[1]-1)[:,None]).squeeze(1),-1)
        if hard and not bool((selected>=0).any()):
            pose=frame['T0'].clone()
        point_covariance=system[3]+system[4].transpose(-1,-2) @ system[2] @ system[4]
        return dict(pose=pose,probabilities=probabilities,candidates=candidates,active_ids=active,
            points=points,point_covariance=point_covariance,scores=scores,initial_scores=initial_scores,
            selected=selected,proposal_history=proposals)

    def _null_result(self,frame,ray_map,candidates,proposals):
        probabilities=frame['T0'].new_zeros(len(candidates),self.max_candidates+1)
        probabilities[:,-1]=1
        active=candidates.new_empty(0)
        return dict(pose=frame['T0'],probabilities=probabilities,candidates=candidates,selected=candidates[:,0],
            active_ids=active,points=ray_map['mean'][active],point_covariance=ray_map['precision'][active],
            scores=probabilities,initial_scores=probabilities,proposal_history=proposals)


def repaired_loss(model,frame,ray_map,truth,identities=None):
    result=model(frame,ray_map,hard=False)
    loss=(result['pose'][:3,3]-truth[:3,3]).square().sum()+25*(result['pose'][:3,:3]-truth[:3,:3]).square().sum()
    if identities is not None and bool((identities>=0).any()):
        mask=identities>=0
        positives=identities[mask]
        negative=result['candidates'][mask]
        ids=torch.cat((positives[:,None],negative),1)
        query=model.encode(frame['descriptors'][mask])
        reference=model.encode(ray_map['descriptors'][ids.clamp_min(0)])
        logits=12*(query[:,None]*reference).sum(-1)
        invalid=(negative<0)|(negative==positives[:,None])
        logits=torch.cat((logits[:,:1],logits[:,1:].masked_fill(invalid,-torch.inf)),-1)
        loss=loss+.1*F.cross_entropy(logits,torch.zeros(len(query),dtype=torch.long,device=query.device))
    loss=loss+sum(p.sum()*0 for p in model.parameters())
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('nonfinite repaired training loss')
    return loss
