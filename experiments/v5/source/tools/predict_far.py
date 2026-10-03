import argparse
import json
import os
from pathlib import Path
import time

import MinkowskiEngine as ME
import torch

from models.far import ForwardAppearanceRegistration
from models.magic_fusion import polar_voxel_centers
from data.leader_nclt_manifest import collate_leader_frames,DEFAULT_VOXEL_SIZE,DEFAULT_HORIZONTAL_RESOLUTION
from tools.far_runtime import budget,setup,make_dataset,load_frame,observation,verify_sources,report
from tools.magic_nclt_runtime import load_pretrained_model,make_matcher,safe_json_dump,sha256_file,PoseEstimationFailure
from utils.full_pool_robust_v1 import full_pool_refine


def lidar_prediction(model,sample,device,seed):
    batch=collate_leader_frames([sample])
    sparse=ME.SparseTensor(features=batch['feats'].to(device),coordinates=batch['coords'].to(device))
    encoded=model.encoder(sparse)
    points=polar_voxel_centers(encoded.C,torch.tensor(encoded.tensor_stride,device=device,dtype=torch.float32),DEFAULT_VOXEL_SIZE,DEFAULT_HORIZONTAL_RESOLUTION)
    prediction=model.decoder(encoded.F)
    count=max(min(50,len(prediction)),int(.5*len(prediction)))
    if count<3:raise PoseEstimationFailure('insufficient correspondences',len(prediction),count)
    selected=prediction[:,3].topk(count).indices
    pose=backend(points[selected],prediction[selected,:3],seed)
    return pose,points,prediction,selected


def backend(source,target,seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    matcher=make_matcher(source.device)
    initial=matcher.estimator(source[None],target[None])[0]
    refined=full_pool_refine(initial,source,target)
    if not torch.isfinite(refined).all():raise FloatingPointError('nonfinite backend pose')
    return refined


def predict_frame(model,far,dataset,index,center,device,seed,baseline=False):
    sample,points,pixels,rgb=load_frame(dataset,index,device)
    assert 'T_WB' not in sample
    initial,all_points,prediction,selected=lidar_prediction(model,sample,device,seed)
    refined=initial
    if not baseline and len(points)>=6:
        observed=observation(far,sample,pixels,rgb,device)
        correction,_,_=far(points,observed,initial)
        rotation=correction[:3,:3]@initial[:3,:3].T
        translation=correction[:3,3]-rotation@initial[:3,3]
        targets=prediction[:,:3]@rotation.T+translation
        refined=backend(all_points[selected],targets[selected],seed)
    world=refined.clone()
    world[:3,3]+=center
    return sample,world,dict(status='success',valid_corr=len(prediction),selected_corr=len(selected),visible_points=len(points))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--baseline',action='store_true')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    output=Path(args.output)
    output.mkdir(exist_ok=False)
    setup(output)
    verify_sources(config)
    device=torch.device('cuda:0')
    dataset=make_dataset(config,'dev')
    model,checkpoint=load_pretrained_model(config['base_checkpoint'],False,device)
    center=torch.tensor(checkpoint['center_t'],device=device,dtype=torch.float32)
    del checkpoint
    model.eval()
    far=None
    if not args.baseline:
        saved=torch.load(args.checkpoint,map_location='cpu')
        assert saved['config']==config
        far=ForwardAppearanceRegistration().to(device).eval()
        far.load_state_dict(saved['model'],strict=True)
        del saved
    temporary=output/'predictions.tmp'
    started=time.monotonic()
    failures=0
    with temporary.open('x') as stream,torch.no_grad():
        for index in range(1200):
            budget(output)
            try:
                sample,pose,metadata=predict_frame(model,far,dataset,index,center,device,config['seed']+index,args.baseline)
            except PoseEstimationFailure as error:
                sample=dict(frame_key=dataset.records[index].frame_key,date=dataset.records[index].date,timestamp_us=dataset.records[index].timestamp_us)
                pose=torch.eye(4,device=device)
                metadata=dict(status='identity_fallback',failure_reason=str(error),valid_corr=error.valid_corr,selected_corr=error.selected_corr)
                failures+=1
            row=dict(frame_key=sample['frame_key'],date=sample['date'],timestamp_us=int(sample['timestamp_us']),T_WB_pred=pose.cpu().double().tolist(),**metadata)
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            if (index+1)%100==0:
                stream.flush()
                report(output,phase='baseline' if args.baseline else 'validation',frames=index+1,total=1200,failures=failures)
    os.replace(temporary,output/'predictions.jsonl')
    safe_json_dump(output/'prediction_freeze.json',dict(schema='leader_far_v1',frames=1200,ground_truth_read_online=False,
        validation_manifest_sha256=dataset.manifest_hashes,prediction_sha256=sha256_file(output/'predictions.jsonl'),
        checkpoint_sha256=config['base_checkpoint_sha256'] if args.baseline else sha256_file(args.checkpoint),
        settings_sha256=sha256_file(args.config),source_sha256=config['source_sha256'],seconds=time.monotonic()-started))


if __name__=='__main__':main()
