import torch
from prototype import ProtectedVisualReliability, boundary_ranking_loss


def selection_state(prediction, valid):
    score=prediction[:,3].detach()
    n=len(score)
    k=max(min(50,n),int(.5*n))
    selected=score.topk(k).indices
    core=selected[:max(1,k//2)]
    gap=(score[core[-1]]-score[selected[-1]]).clamp_min(0)
    core_mask=torch.zeros(n,dtype=torch.bool,device=score.device)
    core_mask[core]=True
    original_mask=torch.zeros_like(core_mask)
    original_mask[selected]=True
    rest=torch.argsort(score,descending=True,stable=True)
    order=torch.cat([selected,rest[~original_mask[rest]]])
    rank=torch.empty(n,dtype=torch.long,device=score.device)
    rank[order]=torch.arange(n,device=score.device)
    return dict(k=k,core=core,original=selected,order=order,rank=rank,gap=gap,adjustable=valid&~core_mask)


def select(prediction,state):
    indices=prediction[:,3].topk(state['k']).indices
    indices=indices[torch.argsort(state['rank'][indices])]
    assert torch.isin(state['core'],indices).all()
    return indices

