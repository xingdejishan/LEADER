import argparse
import json
from pathlib import Path
import time

import torch

from .async_runtime import read_base
from .io import load_arrays
from .repaired import RepairedRayBundle,repaired_loss
from .training_runtime import TrainingBudget,atomic_json,move


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    output.mkdir()
    budget=TrainingBudget(output)
    _,prepared,_=read_base(args.base)
    noise=json.loads((Path(args.base)/'noise.json').read_text())
    labels=json.loads((Path(args.base)/'training_labels.json').read_text())
    data,_=load_arrays(prepared['maps']['odd'])
    ray_map=move(data,'cuda')
    torch.manual_seed(20)
    model=RepairedRayBundle().double().cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    losses=[]
    started=time.monotonic()
    for index in (0,2):
        budget.check()
        frame,meta=load_arrays(prepared['frames']['train'][index],online=True)
        frame['independent_sigma']=torch.tensor(noise['independent_sigma'],dtype=torch.float64)
        frame['shared_covariance']=torch.tensor(noise['shared_covariance'],dtype=torch.float64)
        frame=move(frame,'cuda')
        truth=torch.tensor(labels[meta['frame_key']]['pose'],dtype=torch.float64,device='cuda')
        optimizer.zero_grad()
        loss=repaired_loss(model,frame,ray_map,truth)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        if not bool(torch.isfinite(norm)) or float(norm)==0:
            raise ValueError('missing or invalid repaired gradients')
        optimizer.step()
        losses.append(dict(loss=float(loss.detach()),gradient=float(norm)))
    train_seconds=time.monotonic()-started
    model.eval()
    started=time.monotonic()
    with torch.no_grad():
        result=model(frame,ray_map,hard=True)
    record=dict(updates=2,queries=len(frame['descriptors']),map_tracks=len(ray_map['mean']),
        losses=losses,selected=int((result['selected']>=0).sum()),proposal_rounds=len(result['proposal_history']),
        train_seconds=train_seconds,inference_seconds=time.monotonic()-started,
        gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,weights_discarded=True)
    atomic_json(output/'result.json',record)
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
