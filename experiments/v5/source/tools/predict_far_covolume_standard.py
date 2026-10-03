import argparse,json,os,time
from pathlib import Path
from unittest.mock import patch

import torch

from models.far_covolume_standard import CoVolumeStandardFAR as CoVolumeFAR
from tools.far_runtime import observation,verify_sources
from tools.far_covolume_standard_runtime import make_dataset,load_frame
from tools.far_covolume_budget import setup,budget
from tools.far_v3_runtime import sc2_initial
from tools.magic_nclt_runtime import load_pretrained_model,safe_json_dump,sha256_file,PoseEstimationFailure


def predict(config,output,checkpoint=None,far=None,probe=False):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    setup(output)
    verify_sources(config)
    device=torch.device('cuda:0')
    dataset=make_dataset(config,'dev')
    leader,saved=load_pretrained_model(config['base_checkpoint'],False,device)
    leader.eval()
    center=torch.as_tensor(saved['center_t'],device=device,dtype=torch.float32).reshape(3)
    del saved
    if far is None:
        saved=torch.load(checkpoint,map_location='cpu')
        assert saved['config']==config
        far=CoVolumeFAR(**config['far_covolume']).to(device)
        far.load_state_dict(saved['model'],strict=True)
        del saved
    far.eval()
    indices=[0,300,600,900] if probe else range(1200)
    failures=0
    started=time.monotonic()
    with torch.no_grad(),(output/'predictions.tmp').open('x') as stream:
        for count,index in enumerate(indices,1):
            budget(output)
            try:
                sample=dataset[index]
                assert 'T_WB' not in sample
                initial,source,target,total=sc2_initial(leader,sample,device,config['seed']+index)
                sample,points,pixels,rgb,visible_ids,structure=load_frame(dataset,index,device,source,config['point_max'])
                observed=observation(far,sample,pixels,rgb,device)
                refined,_,diagnostics=far(points,observed,initial,source,target,structure,visible_ids)
                world=refined.clone()
                world[:3,3]+=center
                metadata=dict(status='success',visible_points=len(points),valid_corr=total,selected_corr=len(source))
            except PoseEstimationFailure as error:
                record=dataset.records[index]
                sample=dict(frame_key=record.frame_key,date=record.date,timestamp_us=record.timestamp_us)
                world=torch.eye(4,device=device)
                metadata=dict(status='identity_fallback',failure_reason=str(error),valid_corr=error.valid_corr,selected_corr=error.selected_corr)
                failures+=1
            row=dict(frame_key=sample['frame_key'],date=sample['date'],timestamp_us=int(sample['timestamp_us']),T_WB_pred=world.cpu().double().tolist(),**metadata)
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            if count%100==0 or probe:
                stream.flush()
                safe_json_dump(output/'progress.json',dict(phase='covolume-standard-A-development',frames=count,total=len(indices),failures=failures))
    os.replace(str(output/'predictions.tmp'),str(output/'predictions.jsonl'))
    return dict(schema='leader_far_covolume_standard_train_A_dev',frames=len(indices),ground_truth_read_online=False,validation_manifest_sha256=dataset.manifest_hashes,
        prediction_sha256=sha256_file(output/'predictions.jsonl'),checkpoint_sha256=sha256_file(checkpoint) if checkpoint else None,
        source_sha256=config['source_sha256'],seconds=time.monotonic()-started,failures=failures)


def audited_probe(config,output,far):
    import tools.far_v3_runtime as runtime
    factory=runtime.make_matcher
    calls=[]
    def tracked_factory(device):
        matcher=factory(device)
        estimator=matcher.estimator
        def tracked_estimator(*args,**kwargs):
            calls.append(1)
            return estimator(*args,**kwargs)
        matcher.estimator=tracked_estimator
        return matcher
    failure=AssertionError('forbidden legacy cascade called by covolume')
    with patch('tools.predict_far.backend',side_effect=failure),patch('utils.full_pool_robust_v1.full_pool_refine',side_effect=failure),patch.object(runtime,'make_matcher',side_effect=tracked_factory):
        freeze=predict(config,output,far=far,probe=True)
    assert freeze['frames']==4 and freeze['failures']==0 and len(calls)==4
    freeze.update(audited_external_SC2_calls=4,legacy_full_pool_and_second_backend_forbidden=True)
    safe_json_dump(Path(output)/'prediction_freeze.json',freeze)
    return freeze


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','output','checkpoint'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    freeze=predict(config,args.output,args.checkpoint)
    freeze['settings_sha256']=sha256_file(args.config)
    safe_json_dump(Path(args.output)/'prediction_freeze.json',freeze)


if __name__=='__main__':main()
