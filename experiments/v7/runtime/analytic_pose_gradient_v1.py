import torch


def target_gradient(pose,truth,scales):
    relative=pose[:3,:3]@truth[:3,:3].T
    sine=torch.stack((relative[2,1]-relative[1,2],relative[0,2]-relative[2,0],relative[1,0]-relative[0,1]))*.5
    square=sine.square().sum()
    norm=square.clamp_min(1e-20).sqrt()
    cosine=(relative.trace()-1)*.5
    derivative=.5*(relative.trace()*torch.eye(3,device=pose.device,dtype=pose.dtype)-relative)
    norm_gradient=(derivative.T@sine)/norm*(square>=1e-20)
    angle=torch.atan2(norm,cosine)
    rotation=angle*(cosine*norm_gradient+norm*sine)/(cosine.square()+norm.square())
    rotation=torch.where((square<1e-8)&(cosine>0),derivative.T@sine,rotation)/scales[0]
    return torch.cat((rotation,(pose[:3,3]-truth[:3,3])/scales[3]))


VALUE_GRADIENT='''def value_gradient(model,pose,context,scales,training):
    rotated=context['points']@pose[:3,:3].T
    world=(rotated+pose[:3,3]).requires_grad_(True)
    features=query(model,world)
    visual=.5*(features-context['observed']).square().sum(-1).mean()/context['temperature'] if len(features) else pose.sum()*0
    point_gradient=torch.autograd.grad(visual,world,create_graph=training,retain_graph=True)[0] if len(features) else world*0
    visual_gradient=torch.cat((torch.cross(rotated,point_gradient,dim=-1).sum(0),point_gradient.sum(0)))*scales
    rotated_source=context['source']@pose[:3,:3].T
    residual=rotated_source+pose[:3,3]-context['target']
    precision=context['precision']
    square=torch.einsum('ni,nij,nj->n',residual,precision,residual)
    threshold=context['settings']['robust_delta']
    geometry=threshold**2*(torch.sqrt(1+square/threshold**2)-1).mean()
    weights=(1+square/threshold**2).rsqrt()
    geometry_point=.5*torch.einsum('nij,nj->ni',precision+precision.transpose(-1,-2),residual)*weights[:,None]*context['settings']['geometry_weight']/len(residual)
    geometry_gradient=torch.cat((torch.cross(rotated_source,geometry_point,dim=-1).sum(0),geometry_point.sum(0)))*scales
    value=visual+context['settings']['geometry_weight']*geometry
    return value,visual_gradient+geometry_gradient,features,None,pose
'''
