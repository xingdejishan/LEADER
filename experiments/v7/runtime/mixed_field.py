import types
import torch
from torch.nn import functional as F
from fused_field import HashEncode

def encoding(points,tables,spacings):
    corners=torch.tensor([[a,b,c] for a in (0,1) for b in (0,1) for c in (0,1)],device=points.device,dtype=torch.long)
    scaled=points[:,None,:]/spacings[None,:,None]
    base=scaled.floor().long()
    fraction=scaled-base.to(points.dtype)
    corner=base[:,:,None,:]+corners[None,None,:,:]
    indices=(corner[...,0]^(corner[...,1]*2654435761)^(corner[...,2]*805459861))&(tables.shape[1]-1)
    axes=torch.where(corners[None,None].bool(),fraction[:,:,None],1-fraction[:,:,None])
    weights=axes[...,0]*axes[...,1]*axes[...,2]
    indices=indices+torch.arange(len(tables),device=points.device)[None,:,None]*tables.shape[1]
    values=tables.reshape(-1,tables.shape[-1]).index_select(0,indices.flatten()).reshape(*indices.shape,tables.shape[-1])
    return (values*weights[:,:,:,None]).sum(2).reshape(len(points),-1)

class MixedHash(torch.autograd.Function):
    @staticmethod
    def forward(ctx,points,tables,spacings):
        ctx.original=(points,tables,spacings)
        return HashEncode.forward(ctx,points,tables,spacings)

    @staticmethod
    def backward(ctx,grad):
        if not torch.is_grad_enabled():return HashEncode.backward(ctx,grad)
        points,tables,spacings=ctx.original
        corners=torch.tensor([[a,b,c] for a in (0,1) for b in (0,1) for c in (0,1)],device=points.device,dtype=torch.long)
        scaled=points[:,None,:]/spacings[None,:,None]
        base=scaled.floor().long();fraction=scaled-base.to(points.dtype)
        corner=base[:,:,None,:]+corners[None,None]
        indices=(corner[...,0]^(corner[...,1]*2654435761)^(corner[...,2]*805459861))&(tables.shape[1]-1)
        indices=indices+torch.arange(len(tables),device=points.device)[None,:,None]*tables.shape[1]
        axes=torch.where(corners[None,None].bool(),fraction[:,:,None],1-fraction[:,:,None])
        weights=axes[...,0]*axes[...,1]*axes[...,2]
        values=tables.reshape(-1,tables.shape[-1]).index_select(0,indices.flatten()).reshape(*indices.shape,tables.shape[-1])
        derivative=torch.stack((axes[...,1]*axes[...,2],axes[...,0]*axes[...,2],axes[...,0]*axes[...,1]),-1)
        derivative=derivative*(2*corners-1)[None,None]/spacings[None,:,None,None]
        g=grad.reshape(len(points),len(tables),tables.shape[-1])
        point_gradient=(g[:,:,None,:,None]*values[:,:,:,:,None]*derivative[:,:,:,None,:]).sum((1,2,3))
        contributions=(g[:,:,None,:]*weights[:,:,:,None]).reshape(-1,tables.shape[-1])
        table_gradient=torch.zeros_like(tables).reshape(-1,tables.shape[-1]).index_add(0,indices.flatten(),contributions).reshape_as(tables)
        return point_gradient if points.requires_grad else None,table_gradient if tables.requires_grad else None,None

def forward(self,points):
    return F.normalize(self.mlp(MixedHash.apply(points,self.tables,self.spacings)),dim=-1,eps=1e-8)

def enable(model):
    model.field.forward=types.MethodType(forward,model.field)
    return model
