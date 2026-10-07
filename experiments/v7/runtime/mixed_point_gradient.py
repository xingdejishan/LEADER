import types
import torch
import lean_mixed_hash as base


class PointGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,points,tables,gradient,spacings):
        ctx.save_for_backward(points,tables,gradient,spacings)
        temporary=types.SimpleNamespace(original=(points,tables,spacings))
        return base.point_backward(temporary,gradient)[0]

    @staticmethod
    def backward(ctx,upstream):
        points,tables,gradient,spacings=ctx.saved_tensors
        corners=torch.tensor([[a,b,c] for a in (0,1) for b in (0,1) for c in (0,1)],device=points.device,dtype=torch.long)
        scaled=points[:,None,:]/spacings[None,:,None]
        integral=scaled.floor().long();fraction=scaled-integral.to(points.dtype)
        corner=integral[:,:,None,:]+corners[None,None]
        indices=(corner[...,0]^(corner[...,1]*2654435761)^(corner[...,2]*805459861))&(tables.shape[1]-1)
        indices=indices+torch.arange(len(tables),device=points.device)[None,:,None]*tables.shape[1]
        axes=torch.where(corners[None,None].bool(),fraction[:,:,None],1-fraction[:,:,None])
        values=tables.reshape(-1,tables.shape[-1]).index_select(0,indices.flatten()).reshape(*indices.shape,tables.shape[-1])
        signs=(2*corners-1).to(points.dtype)
        derivative=torch.stack((axes[...,1]*axes[...,2],axes[...,0]*axes[...,2],axes[...,0]*axes[...,1]),-1)
        derivative=derivative*signs[None,None]/spacings[None,:,None,None]
        directional=(derivative*upstream[:,None,None,:]).sum(-1)
        g=gradient.reshape(len(points),len(tables),tables.shape[-1])
        table_gradient=None;feature_gradient=None;point_gradient=None
        if ctx.needs_input_grad[1]:
            contributions=(g[:,:,None,:]*directional[:,:,:,None]).reshape(-1,tables.shape[-1])
            table_gradient=torch.zeros_like(tables).reshape(-1,tables.shape[-1]).index_add(0,indices.flatten(),contributions).reshape_as(tables)
        if ctx.needs_input_grad[2]:
            feature_gradient=(values*directional[:,:,:,None]).sum(2).reshape_as(gradient)
        if ctx.needs_input_grad[0]:
            hxy=axes[...,2]*(signs[:,0]*signs[:,1])[None,None]/spacings[None,:,None].square()
            hxz=axes[...,1]*(signs[:,0]*signs[:,2])[None,None]/spacings[None,:,None].square()
            hyz=axes[...,0]*(signs[:,1]*signs[:,2])[None,None]/spacings[None,:,None].square()
            ux,uy,uz=upstream.unbind(-1)
            coefficient=(g[:,:,None,:]*values).sum(-1)
            px=(coefficient*(uy[:,None,None]*hxy+uz[:,None,None]*hxz)).sum((1,2))
            py=(coefficient*(ux[:,None,None]*hxy+uz[:,None,None]*hyz)).sum((1,2))
            pz=(coefficient*(ux[:,None,None]*hxz+uy[:,None,None]*hyz)).sum((1,2))
            point_gradient=torch.stack((px,py,pz),-1)
        return point_gradient,table_gradient,feature_gradient,None


class CompactMixedHash(torch.autograd.Function):
    @staticmethod
    def forward(ctx,points,tables,spacings):
        return base.original_class.forward(ctx,points,tables,spacings)

    @staticmethod
    def backward(ctx,gradient):
        if torch.is_grad_enabled() and base.coordinate_scope:
            points,tables,spacings=ctx.original
            return PointGradient.apply(points,tables,gradient,spacings),None,None
        return base.original_class.backward(ctx,gradient)


def enable(model,module):
    base.enable(model,module)
    from torch.nn import functional as F
    def forward(self,points):
        return F.normalize(self.mlp(CompactMixedHash.apply(points,self.tables,self.spacings)),dim=-1,eps=1e-8)
    model.field.forward=types.MethodType(forward,model.field)
    return model
