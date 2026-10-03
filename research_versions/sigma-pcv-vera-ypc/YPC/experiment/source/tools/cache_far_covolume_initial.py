import argparse,json,time
from pathlib import Path

import numpy as np
import torch

from tools.far_runtime import verify_sources
from tools.far_covolume_runtime import TrainingB
from tools.far_v3_budget import setup,budget
from tools.far_v3_runtime import sc2_initial
from tools.magic_nclt_runtime import load_pretrained_model,safe_json_dump,sha256_file


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'frames').mkdir()
    setup(output)
    verify_sources(config)
    device=torch.device('cuda:0')
    dataset=TrainingB(config,sam=False,truth=False)
    dataset.verify_all_input_hashes()
    leader,saved=load_pretrained_model(config['base_checkpoint'],False,device)
    leader.eval()
    center=torch.as_tensor(saved['center_t']).detach().cpu().float().reshape(3).tolist()
    del saved
    frames={}
    started=time.monotonic()
    with torch.no_grad():
        for index,record in enumerate(dataset.records):
            budget(output)
            sample=dict(dataset[index])
            sample.pop('T_WB',None)
            initial,source,target,total=sc2_initial(leader,sample,device,config['seed']+index)
            name='frames/%04d.npz'%index
            np.savez_compressed(output/name,initial=initial.cpu().numpy(),source=source.cpu().numpy(),target=target.cpu().numpy())
            frames[record.frame_key]=dict(file=name,sha256=sha256_file(output/name),date=record.date,timestamp_us=record.timestamp_us,
                scan_sha256=record.scan_sha256,image_sha256=record.image_sha256,selected_correspondences=len(source),total_correspondences=total)
            if index==0 or (index+1)%20==0:
                safe_json_dump(output/'progress.json',dict(phase='raw-SC2-train-cache',frames=index+1,total=1200,seconds=time.monotonic()-started))
    assert len(frames)==1200 and sha256_file(config['base_checkpoint'])==config['base_checkpoint_sha256']
    cache=dict(schema='far_covolume_B_raw_sc2_v1',split='B',ground_truth_used_for_prediction=False,backend='raw_SC2-PCR_no_full_pool_refine',
        base_checkpoint_sha256=config['base_checkpoint_sha256'],seed=config['seed'],manifest_sha256=dataset.manifest_hashes,
        producer_source_sha256={name:digest for name,digest in config['source_sha256'].items() if name in config['cache_producer_files']},center_t=center,frames=frames)
    safe_json_dump(output/'initial_cache.json',cache)
    safe_json_dump(output/'completion.json',dict(frames=1200,failures=0,cache_sha256=sha256_file(output/'initial_cache.json'),seconds=time.monotonic()-started))


if __name__=='__main__':main()
