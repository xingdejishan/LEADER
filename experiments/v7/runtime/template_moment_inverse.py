import torch


def inverse(points,pose,gamma,spacing,template):
    direction=(points@pose[:3,:3].T)/points.norm(dim=-1).clamp_min(1e-12)[:,None]
    outer=direction[:,:,None]*direction[:,None,:]
    radial_square=direction.square().sum(-1)
    coefficient=(1-gamma)/(gamma*(gamma+(1-gamma)*radial_square))
    eye=torch.eye(3,device=points.device,dtype=points.dtype)
    warp_inverse=eye/gamma[:,None,None]-coefficient[:,None,None]*outer
    moment=(template[:,0].square().sum()*spacing**2)
    return warp_inverse.transpose(-1,-2)@warp_inverse/moment
