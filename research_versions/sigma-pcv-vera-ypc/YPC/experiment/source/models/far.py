import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def skew(v):
    x,y,z=v.unbind(-1)
    o=torch.zeros_like(x)
    return torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(v.shape[:-1]+(3,3))


def so3_exp(v):
    s=(v*v).sum(-1)
    t=s.clamp_min(1e-20).sqrt()
    a=torch.where(s<1e-6,1-s/6+s*s/120,t.sin()/t)
    b=torch.where(s<1e-6,.5-s/24+s*s/720,(1-t.cos())/s.clamp_min(1e-20))
    k=skew(v)
    return torch.eye(3,device=v.device,dtype=v.dtype)+a[...,None,None]*k+b[...,None,None]*(k@k)


class AppearanceField(nn.Module):
    def __init__(self,levels=12,table_bits=17,channels=2):
        super().__init__()
        self.tables=nn.Parameter(torch.empty(levels,2**table_bits,channels))
        nn.init.uniform_(self.tables,-1e-4,1e-4)
        self.register_buffer('spacings',torch.logspace(math.log10(25.6),math.log10(.1),levels))
        self.register_buffer('corners',torch.tensor(list(itertools.product((0,1),repeat=3)),dtype=torch.long))
        self.mlp=nn.Sequential(nn.Linear(levels*channels,64),nn.ReLU(),nn.Linear(64,32))
        self.mask=2**table_bits-1

    def forward(self,points):
        outputs=[]
        for level,spacing in enumerate(self.spacings):
            scaled=points/spacing
            base=scaled.floor().long()
            fraction=scaled-base.to(points.dtype)
            corner=base[:,None]+self.corners
            index=(corner[...,0] ^ (corner[...,1]*2654435761) ^ (corner[...,2]*805459861)) & self.mask
            axis_weight=torch.where(self.corners[None].bool(),fraction[:,None],1-fraction[:,None])
            weight=axis_weight[...,0]*axis_weight[...,1]*axis_weight[...,2]
            outputs.append((self.tables[level][index]*weight[...,None]).sum(1))
        return F.normalize(self.mlp(torch.cat(outputs,-1)),dim=-1,eps=1e-8)


class ForwardAppearanceRegistration(nn.Module):
    def __init__(self,levels=12,table_bits=17,point_chunk=64):
        super().__init__()
        self.detail=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.GroupNorm(4,16),nn.ReLU(),
            nn.Conv2d(16,32,3,2,1),nn.GroupNorm(4,32),nn.ReLU(),nn.Conv2d(32,32,3,1,1),nn.ReLU())
        self.descriptor=nn.Sequential(nn.Linear(288,64),nn.ReLU(),nn.Linear(64,32))
        self.field=AppearanceField(levels,table_bits)
        self.raw_temperature=nn.Parameter(torch.tensor(math.log((2.-.03)/(10.-2.))))
        self.raw_sigma=nn.Parameter(torch.tensor(math.log((.02-.005)/(.1-.02))))
        self.register_buffer('template',torch.tensor(list(itertools.product(range(-2,3),repeat=3)),dtype=torch.float32))
        self.register_buffer('prior_variance',torch.tensor([math.radians(1.2)**2]*3+[.15**2]*3))
        self.point_chunk=point_chunk

    def observation(self,sam,rgb,pixels,resized_hw):
        sample=pixels[None,:,None]
        original_hw=rgb.shape[-2:]
        scale=pixels.new_tensor([resized_hw[1]/original_hw[1],resized_hw[0]/original_hw[0]])
        sam_grid=sample*scale*2/1024-1
        detail=self.detail(rgb[None])
        rgb_grid=(sample/4+.5)*pixels.new_tensor([2/detail.shape[-1],2/detail.shape[-2]])-1
        s=F.grid_sample(sam[None],sam_grid,align_corners=False)[0,:,:,0].T
        d=F.grid_sample(detail,rgb_grid,align_corners=False)[0,:,:,0].T
        return F.normalize(self.descriptor(torch.cat((s,d),-1)),dim=-1,eps=1e-8)

    def correlate(self,observation,positions,spacing):
        offsets=self.template.to(positions)*spacing
        temperature=.03+9.97*self.raw_temperature.sigmoid()
        sigma=.005+.095*self.raw_sigma.sigmoid()
        probabilities=[]
        for start in range(0,len(positions),self.point_chunk):
            query=positions[start:start+self.point_chunk,None]+offsets
            flat=query.reshape(-1,3)
            if self.training and torch.is_grad_enabled():
                # Recompute field activations in backward to bound the 125-way search memory.
                features=checkpoint(self.field,flat.requires_grad_(True))
            else:
                features=self.field(flat)
            features=features.reshape(len(query),len(offsets),32)
            logits=(features*observation[start:start+len(query),None]).sum(-1)/temperature
            probabilities.append(logits.softmax(-1))
        p=torch.cat(probabilities)
        mean=p@offsets
        difference=offsets[None]-mean[:,None]
        covariance=torch.einsum('nk,nki,nkj->nij',p,difference,difference)
        covariance=covariance+sigma.square()*torch.eye(3,device=positions.device,dtype=positions.dtype)
        return mean,covariance,p,offsets

    def solve(self,positions,pivot,mean,covariance):
        dtype=positions.dtype
        j=torch.cat((-skew((positions-pivot).double()),torch.eye(3,device=positions.device,dtype=torch.float64).expand(len(positions),3,3)),-1)
        precision=torch.linalg.inv(covariance.double())
        prior=torch.diag(1/self.prior_variance.double())
        value=positions.new_zeros(6,dtype=torch.float64)
        for iteration in range(3):
            residual=mean.double()-torch.einsum('nik,k->ni',j,value)
            distance=torch.einsum('ni,nij,nj->n',residual,precision,residual).clamp_min(1e-20).sqrt()
            weight=(2.5/distance).clamp(max=1.) if iteration else torch.ones_like(distance)
            information=precision*weight[:,None,None]
            h=torch.einsum('nia,nij,njb->ab',j,information,j)+prior
            b=torch.einsum('nia,nij,nj->a',j,information,mean.double())
            value=torch.linalg.solve(h,b)
        return value.to(dtype)

    def forward(self,points,observation,initial,truth=None):
        pose=initial
        losses=[]
        stages=[]
        for spacing in (.4,.1):
            positions=points@pose[:3,:3].T+pose[:3,3]
            pivot=pose[:3,3]
            mean,covariance,p,offsets=self.correlate(observation,positions,spacing)
            if truth is not None:
                target=points@truth[:3,:3].T+truth[:3,3]
                delta=target-positions
                inside=(delta.detach().abs()<=2*spacing).all(-1)
                if inside.any():
                    distance=((delta.detach()[:,None]-offsets)**2).sum(-1)
                    labels=(-distance/(2*(spacing*.5)**2)).softmax(-1)
                    losses.append(-(labels[inside]*p[inside].clamp_min(1e-12).log()).sum(-1).mean())
                else:
                    losses.append(p.sum()*0)
            increment=self.solve(positions,pivot,mean,covariance)
            rotation=so3_exp(increment[:3])
            top=torch.cat((rotation@pose[:3,:3],(pivot+increment[3:])[:,None]),-1)
            pose=torch.cat((top,pose.new_tensor([[0.,0.,0.,1.]])),0)
            stages.append(dict(increment=increment,probabilities=p))
        return pose,losses,stages


def perturb_pose(truth,generator):
    tail=2.5 if torch.rand((),generator=generator).item()<.15 else 1.
    omega=torch.randn(3,generator=generator)*math.radians(1.2)*tail
    shift=torch.randn(3,generator=generator)*.15*tail
    rotation=so3_exp(omega.to(truth))
    pose=truth.clone()
    pose[:3,:3]=rotation@truth[:3,:3]
    pose[:3,3]=truth[:3,3]+shift.to(truth)
    return pose


def registration_loss(model,points,all_points,observation,truth,generator):
    initial=perturb_pose(truth,generator)
    pose,correlation,_=model(points,observation,initial,truth)
    predicted=all_points@pose[:3,:3].T+pose[:3,3]
    expected=all_points@truth[:3,:3].T+truth[:3,3]
    distance=(predicted-expected).norm(dim=-1)
    huber=F.huber_loss(distance,torch.zeros_like(distance),delta=.2)
    loss=sum(correlation)/2+huber
    return loss,dict(correlation=float((sum(correlation)/2).detach()),pose=float(huber.detach()))
