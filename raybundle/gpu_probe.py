import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from .cli import train_step
from .io import load_arrays,candidate_indices
from .leader import load_leader,encode_frame
from .model import RayBundleCorrespondence
from .smoke import state_sha
from .training_runtime import TrainingBudget,move,atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output)
    budget=TrainingBudget(out)
    from data.leader_nclt_manifest import ManifestLEADERDataset
    data='/home/zhangcy/projects/nclt_official_300_20260926/NCLT'
    model,center=load_leader('/home/zhangcy/projects/nclt-relation-compare-20260927-a/data/assets/official_l0.pt','cuda')
    original=state_sha(model)
    dataset=ManifestLEADERDataset([Path(data)/'2012-01-22/manifest.json'],data,'train',strict_frozen=False)
    noise=dict(independent_sigma=.1,shared_covariance=np.diag([.2**2]*3+[np.deg2rad(1)**2]*3).tolist())
    sample=dataset[6]
    frame=move(encode_frame(sample,dataset.records[6],model,center,noise),'cuda')
    ray_map,meta=load_arrays('/home/zhangcy/projects/leader-ray-bundle-20260928/smoke-v4/map.npz')
    ray_map=move(ray_map,'cuda')
    adapter=RayBundleCorrespondence().double().cuda()
    ids=candidate_indices(adapter,frame,ray_map,dict(frame_key=dataset.records[6].frame_key),meta)
    optimizer=torch.optim.Adam(adapter.parameters(),lr=1e-4)
    truth=torch.tensor(sample['T_WB'],dtype=torch.float64,device='cuda')
    updates=[]
    started=time.monotonic()
    for _ in range(2):
        budget.check()
        loss,gradient=train_step(adapter,optimizer,frame,ray_map,ids,truth)
        updates.append(dict(loss=loss,gradient=gradient))
    torch.cuda.synchronize()
    assert state_sha(model)==original and all(p.grad is None for p in model.parameters())
    assert float(adapter.adapter[0].weight.grad.norm())>0
    adapter.eval()
    with torch.no_grad():
        result=adapter(frame,ray_map,ids,hard=True)
    atomic_json(out/'probe.json',dict(status='passed',updates=updates,queries=len(ids),seconds=time.monotonic()-started,
                memory_allocated_mib=torch.cuda.max_memory_allocated()/2**20,memory_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,leader_unchanged=True,
                selected=int((result['selected']>=0).sum()),device=torch.cuda.get_device_name(),weights_saved=False))


if __name__=='__main__':
    main()
