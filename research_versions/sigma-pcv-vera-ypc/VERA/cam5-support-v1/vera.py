import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree
import cv2
from torchvision.models import resnet18

from common import PLAN, STORE

def gamma(x):
    frequency=2.**torch.arange(8,device=x.device,dtype=x.dtype)
    angle=x[...,None]*frequency*math.pi
    return torch.cat((angle.sin(),angle.cos()),dim=-1).flatten(-2)

def geometry(points,raw,cameras):
    original=points.detach().cpu().double().numpy()
    raw_tree=cKDTree(raw)
    support_distance,support_index=raw_tree.query(original,workers=1)
    p=raw[support_index].astype(np.float64)
    radius=np.linalg.norm(p[:,:2],axis=1).clip(1e-8)
    radial=np.column_stack((p[:,0]/radius,p[:,1]/radius,np.zeros(len(p))))
    up=np.tile([0.,0.,1.],(len(p),1));tangent=np.cross(up,radial)
    frame=np.stack((radial,tangent,up),axis=-1)
    tree=cKDTree(raw)
    distance,index=tree.query(p,k=PLAN['geometry_neighbors_max'],distance_upper_bound=PLAN['geometry_radius_m'],workers=1)
    good=np.isfinite(distance);safe=np.minimum(index,len(raw)-1)
    values=raw[safe];n=good.sum(1).clip(1)
    mean=(values*good[...,None]).sum(1)/n[:,None]
    centered=(values-mean[:,None])*good[...,None]
    covariance=np.einsum('nki,nkj->nij',centered,centered)/n[:,None,None]
    local=np.einsum('nji,njk,nkl->nil',frame,covariance,frame)
    eigen,axes=np.linalg.eigh(local)
    axes=axes[:,:,::-1];eigen=eigen[:,::-1].clip(0)
    for j in range(3):
        sign=np.sign(axes[np.arange(len(p)),np.abs(axes[:,:,j]).argmax(1),j]);axes[:,:,j]*=np.where(sign==0,1,sign)[:,None]
    basis=np.einsum('nij,njk->nik',frame,axes[:,:,:2])*np.sqrt(eigen[:,:2])[:,None]
    basis[good.sum(1)<3]=0
    eta=np.stack(np.meshgrid(np.linspace(-math.sqrt(2),math.sqrt(2),7),np.linspace(-math.sqrt(2),math.sqrt(2),7),indexing='ij'),axis=-1).reshape(49,2)
    candidates=p[:,None]+np.einsum('nij,mj->nmi',basis,eta)
    scores=[];pixels=[];valids=[]
    for c in cameras:
        t=np.array(c['T_CB']);k=np.array(c['K'])
        def project(x):
            xyz=x@t[:3,:3].T+t[:3,3];uv=xyz@k.T
            uv=uv[...,:2]/xyz[...,2,None].clip(1e-8)
            ok=(xyz[...,2]>1e-6)&(uv[...,0]>=0)&(uv[...,0]<=807)&(uv[...,1]>=0)&(uv[...,1]<=615)
            return uv,xyz[...,2],ok
        uv,z,valid=project(raw)
        raster=np.full(616*808,np.inf,dtype=np.float32)
        xy=np.rint(uv[valid]).astype(int);np.minimum.at(raster,xy[:,1]*808+xy[:,0],z[valid])
        raster=cv2.erode(raster.reshape(616,808),np.ones((3,3),np.uint8))
        cuv,cz,cvalid=project(candidates)
        xy=np.rint(cuv).astype(int);xy[...,0]=xy[...,0].clip(0,807);xy[...,1]=xy[...,1].clip(0,615)
        depth=raster[xy[...,1],xy[...,0]]
        cvalid &= np.isfinite(depth)&(np.abs(cz-depth)<=PLAN['zbuffer_tolerance_m'])
        cvalid &= cvalid[:,::-1]
        puv,pz,pvalid=project(p)
        score=((puv-k[:2,2])/np.array([k[0,0],k[1,1]]))**2
        score=score.sum(-1);score[~pvalid]=np.inf
        scores.append(score);pixels.append(cuv);valids.append(cvalid)
    selected=np.argmin(np.stack(scores,-1),axis=1)
    mask=np.stack(valids,1)[np.arange(len(p)),selected]
    active=np.isfinite(np.stack(scores,1)[np.arange(len(p)),selected])&(good.sum(1)>=3)&(support_distance<=PLAN['support_max_distance_m'])&(mask.sum(1)>1)
    mask[~active]=False
    chosen=np.stack(pixels,1)[np.arange(len(p)),selected]
    return {'points':original.astype(np.float32),'support_points':p.astype(np.float32),'support_distance':support_distance.astype(np.float32),'basis':basis.astype(np.float32),'frame':frame.astype(np.float32),'eta':eta.astype(np.float32),'pixels':chosen.astype(np.float32),'valid':mask,'camera':selected.astype(np.int64),'active':active}

class VERA(nn.Module):
    def __init__(self):
        super().__init__()
        resnet=resnet18(weights=None)
        resnet.load_state_dict(torch.load(STORE/'resnet18-f37072fd.pth',map_location='cpu'),strict=True)
        self.stem=nn.Sequential(resnet.conv1,resnet.bn1,resnet.relu,resnet.maxpool,resnet.layer1)
        self.project=nn.Conv2d(320,64,1)
        self.query=nn.Linear(512,64,bias=False)
        self.key=nn.Linear(64,64,bias=False)
        self.position=nn.Linear(48,64,bias=False)
        self.visual=nn.Linear(64,512,bias=False)
        self.displacement=nn.Linear(48,512,bias=False)
        nn.init.zeros_(self.query.weight);nn.init.zeros_(self.visual.weight);nn.init.zeros_(self.displacement.weight)

    def maps(self,images,sam):
        stem=self.stem(images)
        high=F.interpolate(sam,scale_factor=4,mode='bilinear',align_corners=False)
        h,w=stem.shape[-2:]
        y,x=torch.meshgrid(torch.arange(h,device=images.device,dtype=images.dtype),torch.arange(w,device=images.device,dtype=images.dtype),indexing='ij')
        x=(x*4+.5)*(1024/808)/1024*2-1
        y=(y*4+.5)*(781/616)/1024*2-1
        grid=torch.stack((x,y),-1)[None].expand(len(images),-1,-1,-1)
        aligned=F.grid_sample(high,grid,mode='bilinear',align_corners=False,padding_mode='border')
        return self.project(torch.cat((aligned,stem),1))

    def forward(self,features,points,geo,images,sam,cameras):
        maps=self.maps(images,sam)
        device=features.device
        basis=torch.as_tensor(geo['basis'],device=device);frame=torch.as_tensor(geo['frame'],device=device)
        eta=torch.as_tensor(geo['eta'],device=device);pixels=torch.as_tensor(geo['pixels'],device=device)
        mask=torch.as_tensor(geo['valid'],device=device);camera=torch.as_tensor(geo['camera'],device=device)
        active=torch.as_tensor(geo['active'],device=device)
        support=torch.as_tensor(geo['support_points'],device=device)
        anchor=torch.zeros_like(points);enew=points.new_zeros((len(points),2));visual=points.new_zeros((len(points),64))
        def sample(image,uv):
            h,w=image.shape[-2:]
            grid=uv.clone();grid[...,0]=(uv[...,0]/4+.5)/w*2-1;grid[...,1]=(uv[...,1]/4+.5)/h*2-1
            return F.grid_sample(image[None],grid[None],align_corners=False,padding_mode='border')[0].permute(1,2,0)
        for ci in range(len(cameras)):
            indices=torch.where(active&(camera==ci))[0]
            for idx in indices.split(PLAN['chunk']):
                if not len(idx):continue
                tokens=sample(maps[ci],pixels[idx])
                offset=torch.einsum('nij,mj->nmi',basis[idx],eta)
                local=torch.einsum('nji,nmj->nmi',frame[idx],offset)
                keys=self.key(tokens)+self.position(gamma(local))
                scores=torch.einsum('nd,nmd->nm',self.query(features[idx]),keys)/8
                attention=scores.masked_fill(~mask[idx],-torch.inf).softmax(-1)
                e=attention@eta;delta=torch.einsum('nij,nj->ni',basis[idx],e)
                a=support[idx]+delta
                t=torch.as_tensor(cameras[ci]['T_CB'],device=device,dtype=features.dtype)
                k=torch.as_tensor(cameras[ci]['K'],device=device,dtype=features.dtype)
                xyz=a@t[:3,:3].T+t[:3,3];projected=xyz@k.T
                uv=projected[:,:2]/xyz[:,2,None].clamp_min(1e-6)
                visual[idx]=sample(maps[ci],uv[:,None])[:,0]
                anchor[idx]=a-points[idx];enew[idx]=e
        delta_local=torch.einsum('nji,nj->ni',frame,anchor)
        fused=features+self.visual(visual)+self.displacement(gamma(delta_local))*active[:,None]
        return fused,points+anchor,enew,active
