import torch
from models.far import so3_exp as original,skew


class Rotation(torch.autograd.Function):
    @staticmethod
    def forward(ctx,vector):
        ctx.save_for_backward(vector)
        return original(vector)

    @staticmethod
    def backward(ctx,gradient):
        vector,=ctx.saved_tensors
        if torch.is_grad_enabled():
            value=original(vector)
            return torch.autograd.grad(value,vector,gradient,create_graph=True,retain_graph=True)[0]
        square=(vector*vector).sum(-1)
        norm=square.clamp_min(1e-20).sqrt()
        sine=norm.sin();cosine=norm.cos()
        a=torch.where(square<1e-6,1-square/6+square*square/120,sine/norm)
        b=torch.where(square<1e-6,.5-square/24+square*square/720,(1-cosine)/square.clamp_min(1e-20))
        a_prime=torch.where(square<1e-6,-1/6+square/60,(norm*cosine-sine)/(2*norm**3))
        b_prime=torch.where(square<1e-6,-1/24+square/360,(norm*sine*.5-(1-cosine))/square.clamp_min(1e-20)**2)
        matrix=skew(vector);second=matrix@matrix
        mixed=a[...,None,None]*gradient+b[...,None,None]*(gradient@matrix.transpose(-1,-2)+matrix.transpose(-1,-2)@gradient)
        direct=torch.stack((mixed[...,2,1]-mixed[...,1,2],mixed[...,0,2]-mixed[...,2,0],mixed[...,1,0]-mixed[...,0,1]),-1)
        radial=a_prime*(gradient*matrix).sum((-1,-2))+b_prime*(gradient*second).sum((-1,-2))
        return direct+2*vector*radial[...,None]


def so3_exp(vector):
    return Rotation.apply(vector)
