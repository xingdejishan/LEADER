import numpy as np
import torch
from certificate_fusion import minimum,direction,GAMMA
from model import AnchoredFusion


def main():
    x=np.zeros(512); x[0]=1
    a=np.zeros((1,512)); a[0,1]=1
    d,w,info=minimum(x,a)
    assert info['status']=='certified' and np.linalg.norm(d)-GAMMA<2e-8
    a[0,0]=-.2
    d,w,info=minimum(x,a)
    assert info['status']=='radius_blocked'
    f=torch.zeros(4,512); f[:,0]=1
    target=torch.zeros_like(f); target[3,1]=.05
    kind=torch.tensor([1,1,1,2])
    assert abs(direction(f,f,target,kind).item()-.5)<1e-6
    kind=torch.tensor([0,0,0,2])
    assert abs(direction(f,f,target,kind).item()-1)<1e-6
    head=AnchoredFusion(); image=torch.randn(4,128); editable=torch.tensor([True,True,False,True])
    assert torch.equal(head(f,image,editable),f)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001)
    first=[]
    for _ in range(2):
        optimizer.zero_grad(); fused=head(f,image,editable)
        loss=direction(fused,f,target,kind); loss.backward()
        first.append(head.net[0].weight.grad.norm().item()); optimizer.step()
    assert first[0]==0 and first[1]>0
    fused=head(f,image,editable)
    assert torch.equal(fused[~editable],f[~editable])
    assert torch.all((fused-f).norm(dim=-1)<=.050001)
    print('Minimum-norm certificates, class balancing, skipped labels, identity, gradient flow and protection checks passed')


if __name__=='__main__':
    main()
