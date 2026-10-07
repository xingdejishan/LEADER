import torch
from torch.nn import functional as F
from models.far import skew,so3_exp

def move(pose,delta):
    top=torch.cat((so3_exp(delta[:3])@pose[:3,:3],(pose[:3,3]+delta[3:])[:,None]),-1)
    return torch.cat((top,pose[3:4]),0)

def positions(points,pose):
    return points@pose[:3,:3].T+pose[:3,3]

def jac(points,pivot):
    return torch.cat((-skew(points-pivot),torch.eye(3,device=points.device,dtype=points.dtype).expand(len(points),3,3)),-1)

def task_error(pose,truth,scales):
    relative=pose[:3,:3]@truth[:3,:3].T
    sine=torch.stack((relative[2,1]-relative[1,2],relative[0,2]-relative[2,0],relative[1,0]-relative[0,1]))*.5
    square=sine.square().sum()
    norm=square.clamp_min(1e-20).sqrt()
    cosine=(relative.trace()-1)*.5
    angle_square=torch.where((square<1e-8)&(cosine>0),square,torch.atan2(norm,cosine).square())
    return .5*((pose[:3,3]-truth[:3,3]).square().sum()/scales[3].square()+angle_square/scales[0].square())

def query(model,points,chunk=16384):
    return torch.cat([model.field(points[i:i+chunk].float()) for i in range(0,len(points),chunk)])

def frame_context(model,frame,observed,warm,settings):
    source=frame['source'].double();structure=frame['structure'].double()
    strength=torch.stack([model.geometry_scales(i)[0] for i in range(2)]).mean().double()
    sigma=torch.stack([model.geometry_scales(i)[1] for i in range(2)]).mean().double()
    rotation=warm[:3,:3].double()
    precision=torch.linalg.inv(torch.eye(3,device=source.device,dtype=source.dtype)+strength*(rotation[None]@structure@rotation.T[None]))/sigma.square()
    return dict(points=frame['points'].double(),source=source,target=frame['target'].double(),observed=observed,precision=precision,temperature=.03+9.97*model.raw_temperature.sigmoid(),settings=settings)

def energy(model,pose,context):
    f=query(model,positions(context['points'],pose))
    visual=.5*(f-context['observed']).square().sum(-1).mean()/context['temperature'] if len(f) else pose.sum()*0
    residual=positions(context['source'],pose)-context['target']
    square=torch.einsum('ni,nij,nj->n',residual,context['precision'],residual)
    threshold=context['settings']['robust_delta']
    geo=threshold**2*(torch.sqrt(1+square/threshold**2)-1).mean()
    return visual+context['settings']['geometry_weight']*geo,f

def value_gradient(model,pose,context,scales,training):
    z=torch.zeros(6,device=pose.device,dtype=pose.dtype,requires_grad=True)
    local=move(pose,z*scales)
    value,features=energy(model,local,context)
    gradient=torch.autograd.grad(value,z,create_graph=training,retain_graph=True)[0]
    return value,gradient,features,z,local

def volume(model,pose,spacing,context,scales,truth,training):
    settings=context['settings'];point=context['points'];world=positions(point,pose)
    offsets=model.template.double()*spacing
    denominator=offsets.T@offsets
    hs=[];ces=[];gradients=[]
    size=settings['point_chunk']
    for start in range(0,len(world),size):
        central=world[start:start+size]
        flat=(central[:,None,:]+offsets[None]).reshape(-1,3).float()
        features=model.field(flat)
        features=features.reshape(len(central),125,32)
        changes=torch.einsum('nkd,ka->nda',features.double(),offsets)@torch.linalg.inv(denominator)
        gradients.append(changes)
        j=jac(central,pose[:3,3])*scales[None,None,:]
        fj=changes@j
        hs.append(torch.einsum('nda,ndb->ab',fj,fj)/(max(1,len(world))*context['temperature']))
        if truth is not None:
            delta=(positions(point[start:start+size],truth)-central).detach()
            inside=(delta.abs()<=2*spacing).all(-1)
            labels=(-((delta[:,None]-offsets)**2).sum(-1)/(2*(spacing*.5)**2)).softmax(-1)
            logits=(features*context['observed'][start:start+size,None]).sum(-1)/context['temperature']
            ces.append((-(labels*logits.log_softmax(-1)).sum(-1)*inside).sum())
    h=sum(hs) if hs else pose.new_zeros(6,6)
    geo=positions(context['source'],pose);j=jac(geo,pose[:3,3])*scales[None,None,:]
    residual=geo-context['target'];precision=context['precision']
    square=torch.einsum('ni,nij,nj->n',residual,precision,residual)
    weights=(1+square/settings['robust_delta']**2).rsqrt()
    h=h+settings['geometry_weight']*torch.einsum('nia,nij,n,njb->ab',j,precision,weights,j)/len(geo)
    floor=h.diag().mean().detach().clamp_min(1e-8)*1e-6
    h=h+model.damping*torch.diag(h.diag().clamp_min(floor))
    if truth is not None:
        delta=(positions(point,truth)-world).detach()
        count=(delta.abs()<=2*spacing).all(-1).sum().clamp_min(1)
        ce=sum(ces)/count if ces else pose.sum()*0
    else:ce=pose.sum()*0
    return h,torch.cat(gradients) if gradients else pose.new_empty(0,32,3),ce

def constrained(h,g,a,bound):
    constraints=torch.cat((a,-a),0);x=g*0;active=[];converged=False
    for _ in range(32):
        gradient=h@x+g
        if active:
            c=constraints[active]
            kkt=torch.cat((torch.cat((h,c.T),1),torch.cat((c,h.new_zeros(len(active),len(active))),1)),0)
            try:solution=torch.linalg.solve(kkt,torch.cat((-gradient,g.new_zeros(len(active)))))
            except RuntimeError as error:
                if 'singular' not in str(error) and 'diagonal element' not in str(error):raise
                direction=-g
                step=(g@g)/(direction@(h@direction)).clamp_min(1e-30)
                candidate=step*direction
                maximum=(a@candidate).abs().max().clamp_min(1e-15)
                candidate=candidate*(bound/maximum).clamp(max=1.)
                return candidate,dict(active_constraints=len(active),QP_converged=False,fallback='singular_active_set_feasible_Cauchy')
            direction=solution[:6];multipliers=solution[6:]
        else:
            direction=torch.linalg.solve(h,-gradient);multipliers=None
        if float(direction.detach().abs().max())<1e-10:
            if not active or float(multipliers.detach().min())>=-1e-9:converged=True;break
            active.pop(int(multipliers.detach().argmin()));continue
        rate=constraints@direction;slack=bound-constraints@x
        ratios=torch.where(rate>1e-12,slack/rate.clamp_min(1e-12),torch.full_like(rate,float('inf')))
        if active:ratios=ratios.index_fill(0,torch.tensor(active,device=g.device),float('inf'))
        index=int(ratios.detach().argmin());alpha=ratios[index].clamp(min=0,max=1)
        x=x+alpha*direction
        if float(alpha.detach())<1-1e-10 and len(active)<6:active.append(index)
        elif float(alpha.detach())>=1-1e-10:
            if not active:converged=True;break
        else:break
    maximum=(a@x).abs().max().clamp_min(1e-15)
    x=x*(bound/maximum).clamp(max=1.)
    return x,dict(active_constraints=len(active),QP_converged=converged)

def refine(model,frame,observed,settings,shape_enabled=True,truth=None):
    training=truth is not None and torch.is_grad_enabled()
    if truth is not None:truth=truth.double()
    with torch.enable_grad():
        warm,_=model.warm_start(frame['initial'],frame['source'],frame['target'])
        pose=warm.double();context=frame_context(model,frame,observed,pose,settings)
        scales=pose.new_tensor([settings['tau_R']]*3+[settings['tau_t']]*3)
        spacing=settings['initial_spacing'];rejects=0;traces=[];ces=[];shapes=[];status='budget'
        gt_energy=energy(model,truth.double(),context)[0] if training and shape_enabled else None
        if gt_energy is not None:
            perturbation=frame['perturbation'].double()*scales
            disturbed=move(truth.double(),perturbation)
            e,g,_,z,local=value_gradient(model,disturbed,context,scales,True)
            target=task_error(local,truth.double(),scales)
            target_g=torch.autograd.grad(target,z,create_graph=True,retain_graph=True)[0]
            shapes.append((e-gt_energy-target).square()+settings['shape_beta']*(g-target_g).square().mean())
        for iteration in range(settings['max_rounds']):
            e,g,f,z,local=value_gradient(model,pose,context,scales,training)
            if gt_energy is not None:
                target=task_error(local,truth.double(),scales)
                target_g=torch.autograd.grad(target,z,create_graph=True,retain_graph=True)[0]
                shapes.append((e-gt_energy-target).square()+settings['shape_beta']*(g-target_g).square().mean())
            h,changes,ce=volume(model,pose,spacing,context,scales,truth,training)
            ces.append(ce)
            support=torch.cat((positions(context['points'],pose),positions(context['source'],pose)))
            a=(jac(support,pose[:3,3])*scales[None,None,:]).reshape(-1,6)
            delta,qp=constrained(h,g,a,2*spacing)
            increment=delta*scales
            trial=move(pose,increment)
            support_local=torch.cat((context['points'],context['source']))
            actual_step=(positions(support_local,trial)-support).abs().max()
            for _ in range(10):
                if float(actual_step.detach())<=2*spacing*(1+1e-8):break
                delta=delta*.5;increment=delta*scales;trial=move(pose,increment)
                actual_step=(positions(support_local,trial)-support).abs().max()
            assert float(actual_step.detach())<=2*spacing*(1+1e-6)
            predicted=-(g@delta+.5*delta@(h@delta))
            trial_e,trial_g,trial_f,_,_=value_gradient(model,trial,context,scales,training)
            actual=e-trial_e
            ratio=float((actual/predicted.clamp_min(1e-15)).detach()) if float(predicted.detach())>settings['predicted_floor'] else None
            accepted=ratio is not None and ratio>settings['accept_ratio'] and float(actual.detach())>0
            linear_step=(positions(context['points'],trial)-positions(context['points'],pose)).float()
            predicted_feature=torch.einsum('nda,na->nd',changes,linear_step.double())
            feature_delta=trial_f.double()-f.double()
            error=float(((feature_delta-predicted_feature).square().sum().sqrt()/(feature_delta.square().sum().sqrt()+1e-8)).detach())
            norm=float((trial_g if accepted else g).detach().abs().max())
            if accepted:pose=trial;rejects=0
            else:rejects+=1
            traces.append(dict(round=iteration,spacing=spacing,E_before=float(e.detach()),E_trial=float(trial_e.detach()),predicted_reduction=float(predicted.detach()),actual_reduction=float(actual.detach()),ratio=ratio,accepted=accepted,model_error=error,gradient_inf=norm,actual_displacement_inf=float(actual_step.detach()),**qp))
            if norm<=settings['gradient_tolerance'] and error<=settings['model_tolerance']:status='converged';break
            shrink=not accepted or ratio<settings['shrink_ratio'] or (norm<=settings['gradient_tolerance'] and error>settings['model_tolerance'])
            if shrink:
                if spacing<=settings['min_spacing']:status='scale_floor_unconfirmed';break
                spacing=max(settings['min_spacing'],spacing*.5)
            elif ratio>settings['expand_ratio'] and float(actual_step.detach())>1.6*spacing:spacing=min(settings['max_spacing'],spacing*2)
            if rejects>=settings['max_rejects']:status='rejections_unconfirmed';break
            if not training:pose=pose.detach();context={n:v.detach() if torch.is_tensor(v) else v for n,v in context.items()}
        correlation=torch.stack(ces).mean() if ces else pose.sum()*0
        shape=torch.stack(shapes).mean() if shapes else pose.sum()*0
        result=pose.to(frame['initial'].dtype)
        return result,correlation,shape,dict(status=status,rounds=len(traces),accepted=sum(t['accepted'] for t in traces),traces=traces)

def loss(model,frame,observed,settings,shape_enabled=True):
    pose,ce,shape,diagnostics=refine(model,frame,observed,settings,shape_enabled,frame['truth'])
    distance=(positions(frame['raw'],pose)-positions(frame['raw'],frame['truth'])).norm(dim=-1)
    huber=F.huber_loss(distance,torch.zeros_like(distance),delta=.2)
    total=ce+huber+settings['shape_weight']*shape
    return total,dict(correlation=float(ce.detach()),pose=float(huber.detach()),shape=float(shape.detach()),**diagnostics)
