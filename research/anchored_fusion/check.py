import torch
from torch.nn import functional as F
from model import AnchoredFusion, contrastive

torch.manual_seed(2089)
model = AnchoredFusion()
features, image = torch.randn(12,512), torch.randn(12,128)
editable = torch.arange(12)%3==0
assert sum(p.numel() for p in model.parameters())==37408
assert torch.equal(model(features,image,editable),features)
decoder = torch.nn.Linear(512,4).requires_grad_(False)
loss = decoder(model(features,image,editable)).square().mean()
loss.backward()
assert model.net[-1].weight.grad.abs().sum()>0
assert all(p.grad is None for p in decoder.parameters())
with torch.no_grad():
    model.net[-1].weight.normal_()
    model.net[-1].bias.normal_()
fused = model(features,image,editable)
assert torch.equal(fused[~editable],features[~editable])
assert ((fused-features).norm(dim=-1)<=.05*features.norm(dim=-1)+1e-6).all()
assert torch.equal(decoder(fused)[~editable],decoder(features)[~editable])
anchors = F.normalize(torch.randn(2,4,512),dim=-1)
query = torch.randn(2,512,requires_grad=True)
positive = torch.tensor([[True,True,False,False],[True,False,False,False]])
negative = torch.tensor([[False,False,True,False],[False,True,True,False]])
loss = contrastive(query,anchors,positive,negative)
altered = anchors.clone()
altered[:,3] = F.normalize(torch.randn(2,512),dim=-1)
assert torch.equal(loss,contrastive(query,altered,positive,negative))
gradient = torch.autograd.grad(loss.sum(),query)[0]
assert (contrastive(query-.01*gradient,anchors,positive,negative)<loss).all()
print('PASS identity, 37408 parameters, clip bound, protected features/outputs, frozen decoder gradients, multi-positive gradient and ignored gray labels')
