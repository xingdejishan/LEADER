import torch

from models.far_covolume_standard import CoVolumeStandardFAR
from models.far_covolume import jacobian,tukey


class SigmaFAR(CoVolumeStandardFAR):
    def precision(self,covariance):
        cov=covariance.double()+torch.eye(3,device=covariance.device,dtype=torch.float64)*1e-8
        return torch.linalg.inv(cov)

    def warm_start(self,initial,source,target):
        if torch.is_tensor(target):
            return super().warm_start(initial,source,target)
        return super().warm_start(initial,source,target['mean'])

    def correlate_shared(self,observation,points,pose,source,target,structure,visible_ids,stage,appearance=None):
        if torch.is_tensor(target):
            return super().correlate_shared(observation,points,pose,source,target,structure,visible_ids,stage,appearance)
        positions=points@pose[:3,:3].T+pose[:3,3]
        if appearance is None:
            appearance,offsets=self.appearance_logits(observation,positions,self.search_spacings[stage])
        else:
            offsets=self.template.to(positions)*self.search_spacings[stage]
        geometry=torch.zeros_like(appearance)
        valid=visible_ids>=0
        if valid.any():
            ids=visible_ids[valid]
            assert torch.allclose(points[valid],source[ids],atol=1e-6,rtol=0)
            residual=target['prior'][ids]-positions[valid]
            difference=(offsets[None]-residual[:,None]).double()
            precision=self.precision(target['covariance'][ids])
            square=torch.einsum('nki,nij,nkj->nk',difference,precision,difference).clamp_min(0)
            _,cap,_=self.volume_scales(stage)
            threshold=self.tukey_thresholds[stage]
            logits=-cap*6/threshold**2*tukey(square,threshold)
            geometry=geometry.index_copy(0,valid.nonzero().flatten(),logits.to(appearance))
        probability=(appearance+geometry).softmax(-1)
        mean=probability@offsets
        difference=offsets[None]-mean[:,None]
        covariance=torch.einsum('nk,nki,nkj->nij',probability,difference,difference)
        sigma=.005+.095*self.raw_sigma.sigmoid()
        covariance=covariance+sigma.square()*torch.eye(3,device=pose.device,dtype=pose.dtype)
        return mean,covariance,appearance,offsets

    def solve_shared(self,points,pose,mean,covariance,source,target,structure,visible_ids,stage):
        if torch.is_tensor(target):
            return super().solve_shared(points,pose,mean,covariance,source,target,structure,visible_ids,stage)
        current=pose.double()
        positions=points.double()@current[:3,:3].T+current[:3,3]
        residual=mean.double()
        precision=self.precision(covariance)
        distance=torch.einsum('ni,nij,nj->n',residual,precision,residual).clamp_min(1e-20).sqrt()
        information=precision*(2.5/distance).clamp(max=1)[:,None,None]
        j=jacobian(positions,current[:3,3])
        h=torch.einsum('nia,nij,njb->ab',j,information,j)
        b=torch.einsum('nia,nij,nj->a',j,information,residual)
        outside=torch.ones(len(source),device=pose.device,dtype=torch.bool)
        outside[visible_ids[visible_ids>=0]]=False
        support=positions
        inliers=0
        if outside.any():
            gp=source[outside].double()@current[:3,:3].T+current[:3,3]
            gr=target['mean'][outside].double()-gp
            precision=self.precision(target['posterior_covariance'][outside])
            square=torch.einsum('ni,nij,nj->n',gr,precision,gr).clamp_min(0)
            weights=(1-square/self.tukey_thresholds[stage]**2).clamp_min(0).square()
            inliers=int((weights.detach()>0).sum())
            sigma=self.geometry_scales(stage)[1]
            information=precision*weights[:,None,None]/sigma.square()
            j=jacobian(gp,current[:3,3])
            h=h+torch.einsum('nia,nij,njb->ab',j,information,j)
            b=b+torch.einsum('nia,nij,nj->a',j,information,gr)
            support=torch.cat((positions,gp))
        updated,info=self.bounded_update(current,h,b,support,2*self.search_spacings[stage])
        return updated.to(pose),dict(info,outside_inliers=inliers)
