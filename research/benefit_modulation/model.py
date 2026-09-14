import torch
from torch import nn
from torch.nn import functional as F


def reliability_rank(score):
    _,inverse,counts=torch.unique(score.detach(),sorted=True,return_inverse=True,return_counts=True)
    average_rank=counts.cumsum(0)-1-(counts-1)/2
    return average_rank[inverse]/max(len(score)-1,1)


class BenefitModulation(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(641,32),nn.ReLU(),nn.Linear(32,513))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self,lidar,image,rank,valid,warmup=False):
        lidar=lidar.detach()
        image=torch.where(valid[:,None],image.detach(),torch.zeros_like(image))
        inputs=torch.cat([F.layer_norm(lidar,(512,)),F.layer_norm(image,(128,)),rank.detach()[:,None]],-1)
        output=self.head(inputs)
        modulation=output[:,:512].tanh()
        logits=output[:,512]
        gate=torch.full_like(logits,.5) if warmup else logits.sigmoid()
        amplitude=.02+.08*(1-rank.detach()).square()
        relative=amplitude[:,None]*modulation
        attempt=lidar*(1+torch.where(valid[:,None],relative,torch.zeros_like(relative)))
        fused=lidar*(1+torch.where(valid[:,None],relative*gate[:,None],torch.zeros_like(relative)))
        return dict(fused=fused,attempt=attempt,gate=gate,logits=logits,modulation=modulation,amplitude=amplitude)


def auxiliary_losses(prediction,attempt_prediction,baseline_error,target,logits,valid,selected):
    actual_error=(prediction[:,:3]-target).norm(dim=-1)
    attempt_error=(attempt_prediction[:,:3]-target).norm(dim=-1)
    soft_target=((baseline_error-attempt_error)/.01).sigmoid().detach()
    good=valid&selected
    keep=torch.relu(actual_error[good]-baseline_error[good]).mean() if good.any() else prediction.sum()*0
    per_point=F.binary_cross_entropy_with_logits(logits,soft_target,reduction='none')
    parts=[per_point[mask].mean() for mask in [valid&selected,valid&~selected] if mask.any()]
    gate=torch.stack(parts).mean() if parts else logits.sum()*0
    return keep,gate,soft_target
