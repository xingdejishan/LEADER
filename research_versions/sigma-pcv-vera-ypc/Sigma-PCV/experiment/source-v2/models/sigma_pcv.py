import math
import types

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def field_features(field, points, spacing=None):
    scaled=points[None]/field.spacings[:,None,None]
    base=scaled.floor().long()
    fraction=scaled-base.to(points.dtype)
    corner=base[:,:,None]+field.corners
    index=(corner[...,0]^(corner[...,1]*2654435761)^(corner[...,2]*805459861))&field.mask
    axis=torch.where(field.corners[None,None].bool(),fraction[:,:,None],1-fraction[:,:,None])
    weight=axis[...,0]*axis[...,1]*axis[...,2]
    levels=torch.arange(len(field.spacings),device=points.device)[:,None,None]
    features=(field.tables[levels,index]*weight[...,None]).sum(2)
    if spacing is not None:
        features=features*(field.spacings[:,None]>=spacing.detach()[None,:,0]).to(features)[...,None]
    features=features.permute(1,0,2).reshape(len(points),-1)
    return F.normalize(field.mlp(features),dim=-1,eps=1e-8)


def fast_field(self,points):
    return field_features(self,points)


def enable(far):
    far.field.forward=types.MethodType(fast_field,far.field)


def decode(decoder,features):
    hidden=decoder.proj_in(features)
    for block in decoder.blocks:
        value=block['mlp'](hidden)
        value=value.view(*value.shape[:-1],decoder.head_channels,decoder.head_num).max(-1)[0]
        hidden=decoder.relu(block['norm'](hidden+value))
    return decoder.pred_out(hidden),hidden


class SigmaPCV(nn.Module):
    def __init__(self,sigma_min=.05,sigma_max=2.,initial_sigma=.4,point_chunk=8):
        super().__init__()
        self.branch=nn.Linear(512,6)
        nn.init.zeros_(self.branch.weight)
        nn.init.zeros_(self.branch.bias)
        with torch.no_grad():
            self.branch.bias[[0,2,5]]=math.log(math.expm1(initial_sigma))
        self.raw_temperature=nn.Parameter(torch.tensor(math.log((.1-.03)/(10.-.1))))
        self.sigma_min=sigma_min
        self.sigma_max=sigma_max
        self.point_chunk=point_chunk
        self.register_buffer('grid',torch.cartesian_prod(torch.arange(-2,3),torch.arange(-2,3),torch.arange(-2,3)).float())

    def cholesky(self,hidden):
        raw=self.branch(hidden)
        diagonal=F.softplus(raw[:,[0,2,5]]).clamp(self.sigma_min,self.sigma_max)
        off=self.sigma_max*torch.tanh(raw[:,[1,3,4]]/self.sigma_max)
        zero=torch.zeros_like(diagonal[:,0])
        return torch.stack((diagonal[:,0],zero,zero,off[:,0],diagonal[:,1],zero,off[:,1],off[:,2],diagonal[:,2]),-1).reshape(-1,3,3)

    def appearance(self,field,mu,cholesky,descriptors,neighbours,valid):
        delta=torch.einsum('nij,kj->nki',cholesky,self.grid)
        neighbour_mu=mu[neighbours]
        neighbour_d=descriptors[neighbours]
        spacing=cholesky.norm(dim=1).max(-1)[0]
        logits=[]
        tau=.03+9.97*self.raw_temperature.sigmoid()
        for start in range(0,len(cholesky),self.point_chunk):
            stop=min(start+self.point_chunk,len(cholesky))
            position=neighbour_mu[start:stop,:,None]+delta[start:stop,None]
            shape=position.shape
            flat=position.reshape(-1,3)
            if torch.is_grad_enabled() and not flat.requires_grad:
                flat.requires_grad_(True)
            steps=spacing[start:stop,None,None,None].expand(shape[0],shape[1],shape[2],1).reshape(-1,1)
            def query(p,s):
                return field_features(field,p,s)
            features=checkpoint(query,flat,steps) if torch.is_grad_enabled() else query(flat,steps)
            features=features.reshape(shape[0],shape[1],shape[2],32)
            cosine=(features*neighbour_d[start:stop,:,None]).sum(-1)
            mask=valid[start:stop].to(cosine)
            logits.append((cosine*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]/tau)
        return torch.cat(logits),delta

    def posterior(self,field,mu,cholesky,descriptors,neighbours,valid):
        supported=valid.any(-1)
        covariance=cholesky@cholesky.transpose(-1,-2)
        means=mu
        covariances=covariance
        if supported.any():
            ids=supported.nonzero().flatten()
            logits,delta=self.appearance(field,mu,cholesky[ids],descriptors,neighbours[ids],valid[ids])
            probability=(logits-.5*self.grid.square().sum(-1)).softmax(-1)
            shift=(probability[:,:,None]*delta).sum(1)
            difference=delta-shift[:,None]
            posterior=torch.einsum('nk,nki,nkj->nij',probability,difference,difference)+covariance[ids]/12
            means=mu.index_copy(0,ids,mu[ids]+shift)
            covariances=covariance.index_copy(0,ids,posterior)
        return means,covariances,supported

    def ce(self,field,mu,cholesky,descriptors,neighbours,valid,truth):
        with torch.no_grad():
            target=torch.linalg.solve_triangular(cholesky.double(),(truth-mu).double()[...,None],upper=False).squeeze(-1).float()
            inside=valid.any(-1)&(target.abs()<=2).all(-1)
            ids=inside.nonzero().flatten()
        if not len(ids):
            return descriptors.sum()*0+self.raw_temperature*0,0.
        logits,_=self.appearance(field,mu.detach(),cholesky[ids].detach(),descriptors,neighbours[ids],valid[ids])
        axis=(1-(target[ids,None]-self.grid).abs()).clamp_min(0)
        labels=axis[:,:,0]*axis[:,:,1]*axis[:,:,2]
        labels=labels/labels.sum(-1,keepdim=True).clamp_min(1e-8)
        logp=(logits-.5*self.grid.square().sum(-1)).log_softmax(-1)
        return -(labels*logp).sum(-1).mean(),float(inside.float().mean())


def prior_loss(mu,cholesky,reliability,truth,sigma_min):
    error=torch.linalg.solve_triangular(cholesky.double(),(truth-mu).double()[...,None],upper=False).squeeze(-1)
    distance=error.square().sum(-1).clamp_min(1e-20).sqrt()
    huber=torch.where(distance<1,.5*distance.square(),distance-.5)
    value=huber+cholesky.diagonal(dim1=-2,dim2=-1).double().log().sum(-1)-3*math.log(sigma_min)
    factor=math.log(10.)/math.pi
    low=(reliability.clamp(-10*math.pi,10*math.pi).atan()*factor).exp()
    high=(reliability.atan()*factor).exp()
    weights=torch.maximum(low,high)/torch.minimum(low,high).sum().clamp_min(1e-6)
    return (weights*value).sum().float()


def posterior_nll(mean,covariance,truth,supported):
    if not supported.any():
        return mean.sum()*0+covariance.sum()*0
    cov=covariance[supported].double()+torch.eye(3,device=mean.device,dtype=torch.float64)*1e-8
    error=(truth[supported]-mean[supported]).double()
    solution=torch.linalg.solve(cov,error[...,None]).squeeze(-1)
    factor=torch.linalg.cholesky(cov)
    logdet=2*factor.diagonal(dim1=-2,dim2=-1).log().sum(-1)
    assert torch.isfinite(logdet).all()
    return (.5*(error*solution).sum(-1)+.5*logdet+1.5*math.log(2*math.pi)).mean().float()
