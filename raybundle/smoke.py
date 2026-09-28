import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from .cli import train_step
from .io import candidate_indices, save_arrays
from .mapping import read_records, build_map, sha256
from .model import RayBundleCorrespondence
from .resources import ResourceBudget


def state_sha(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    budget = ResourceBudget(args.output, seconds=240)
    torch.set_num_threads(1)
    torch.manual_seed(20)
    from data.leader_nclt_manifest import ManifestLEADERDataset
    from .leader import load_leader, encode_frame
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    manifest = Path(args.data_root)/'2012-01-22/manifest.json'
    reference_records = read_records(args.data_root,[manifest],mapping=True)[:12]
    query_index = 6
    query_key = reference_records[query_index]['key']
    reference_records = [r for r in reference_records if r['key'] != query_key]
    data, metadata = build_map(reference_records,max_features=256,max_tracks=512)
    save_arrays(output/'map.npz',data,metadata)
    print(json.dumps(dict(stage='map',tracks=len(data['mean']),references=len(reference_records))),flush=True)
    budget.check()
    dataset = ManifestLEADERDataset([manifest],args.data_root,'train',strict_frozen=False)
    leader, center = load_leader(args.checkpoint)
    original_sha = state_sha(leader)
    noise = dict(independent_sigma=.1,shared_covariance=np.diag([.2**2]*3+[np.deg2rad(1.)**2]*3).tolist(),
                 calibrated=False,purpose='functional smoke only; not a frozen scientific noise model')
    sample = dataset[query_index]
    truth = torch.tensor(sample['T_WB'],dtype=torch.float64)
    frame = encode_frame(sample,dataset.records[query_index],leader,center,noise,max_queries=24)
    frame_meta = dict(frame_key=query_key,checkpoint_sha256=sha256(args.checkpoint),noise=noise)
    save_arrays(output/'query.npz',frame,frame_meta)
    (output/'training_labels.json').write_text(json.dumps({query_key:dict(pose=truth.tolist())}))
    ray_map = {k:torch.as_tensor(v,dtype=torch.long if v.dtype.kind in 'iu' else torch.float64) for k,v in data.items()}
    model = RayBundleCorrespondence().double()
    ids = candidate_indices(model,frame,ray_map,frame_meta,metadata)
    model.eval()
    with torch.no_grad():
        result = model(frame,ray_map,ids,hard=True)
    selected = int((result['selected'] >= 0).sum())
    if selected == 0:
        raise AssertionError('real smoke produced no geometric visual correspondences')
    print(json.dumps(dict(stage='inference',correspondences=selected,candidates=int((ids>=0).sum()))),flush=True)
    initial = state_sha(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    updates=[]
    for _ in range(2):
        budget.check()
        loss, gradient = train_step(model,optimizer,frame,ray_map,ids,truth)
        updates.append(dict(loss=loss,gradient=gradient))
    if state_sha(model) == initial or float(model.adapter[0].weight.grad.norm()) == 0:
        raise AssertionError('visual adapter did not update through geometry')
    if state_sha(leader) != original_sha or any(p.grad is not None for p in leader.parameters()):
        raise AssertionError('frozen LEADER changed')
    save_arrays(output/'correspondences.npz',{k:result[k] for k in ('pose','probabilities','candidates','active_ids','points','point_covariance','selected')},dict(frame_key=query_key))
    report=dict(status='passed',frames=1,reference_frames=len(reference_records),tracks=len(data['mean']),
                selected_correspondences=selected,updates=updates,leader_unchanged=True,query_gt_passed_to_model=False,
                trainable_changed=True,weights_saved=False,accuracy_result=False,noise_model_calibrated=False,
                seconds=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                prefix=str(Path(__import__('sys').prefix)),device='cpu',threads=torch.get_num_threads())
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
