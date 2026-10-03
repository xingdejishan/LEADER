import json
from pathlib import Path
import time

from common import ROOT,STORE,PLAN,atomic,sha,guard,verify_source
import numpy as np

assert __import__('os').environ['CUDA_VISIBLE_DEVICES']=='-1'
verify_source();guard()
inputs=json.loads((STORE/'inputs.json').read_text())['dev']
truth={};truth_sha={}
for date in PLAN['dev_dates']:
    data=Path(PLAN['data_root']);public=json.loads((data/date/'manifest.json').read_text())
    tbc=np.array(public['roi']['T_BC'])
    root=data/'evaluator_only/test_gt'/date
    manifest=root/'manifest.json';digest=sha(manifest);assert (root/'manifest.sha256').read_text().split()[0]==digest
    spec=json.loads(manifest.read_text());assert spec['split']=='test_evaluator_only'
    truth_sha[date]=digest
    rows={x['timestamp_us']:x for x in spec['frames']}
    for f in inputs:
        if f['date']!=date:continue
        gt=rows[f['timestamp']]['ground_truth'];path=(root/gt['path']).resolve()
        assert root.resolve() in path.parents and sha(path)==gt['sha256']
        truth[f['key']]=np.loadtxt(path).reshape(4,4)@np.linalg.inv(tbc)

errors={};results={};predictions={}
for arm in PLAN['arms']:
    folder=STORE/arm;freeze=json.loads((folder/'prediction-freeze.json').read_text());path=folder/'predictions.jsonl'
    assert sha(path)==freeze['prediction_sha256'] and freeze['frames']==len(inputs)
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert [r['key'] for r in rows]==[r['key'] for r in inputs]
    values=[]
    for r in rows:
        p=np.array(r['pose']);t=truth[r['key']]
        assert np.isfinite(p).all() and np.max(np.abs(p[3]-[0,0,0,1]))<1e-6
        assert np.max(np.abs(p[:3,:3].T@p[:3,:3]-np.eye(3)))<1e-2
        values.append([np.linalg.norm(p[:3,3]-t[:3,3]),np.degrees(np.arccos(np.clip((np.trace(p[:3,:3].T@t[:3,:3])-1)/2,-1,1)))])
    e=np.array(values);errors[arm]=e;predictions[arm]=rows
    results[arm]={'frames':len(rows),'failures':sum(r['status']!='success' for r in rows),'mpe_m':float(e[:,0].mean()),'moe_deg':float(e[:,1].mean()),'mpe_p90_m':float(np.percentile(e[:,0],90)),'moe_p90_deg':float(np.percentile(e[:,1],90)),'inference_seconds':freeze['seconds'],'prediction_sha256':sha(path),'freeze_sha256':sha(folder/'prediction-freeze.json')}
    assert sha(path)==freeze['prediction_sha256']
pairs={}
for baseline in ('l0','lidar_finetune'):
    delta=errors['vera']-errors[baseline];rng=np.random.default_rng(20261002)
    means=[]
    for _ in range(2000):means.append(delta[rng.integers(0,len(delta),len(delta))].mean(0))
    pairs[baseline]={'mean_delta':delta.mean(0).tolist(),'descriptive_frame_bootstrap_95':np.percentile(means,[2.5,97.5],axis=0).tolist(),'position_improved':int((delta[:,0]<-1e-6).sum()),'position_worsened':int((delta[:,0]>1e-6).sum()),'rotation_improved':int((delta[:,1]<-1e-6).sum()),'rotation_worsened':int((delta[:,1]>1e-6).sum())}
np.savez_compressed(STORE/'paired-errors.npz',**errors,keys=np.array([f['key'] for f in inputs]),truth=np.stack([truth[f['key']] for f in inputs]))
report={'protocol':'standard 8 dates x 300: 1200 train / 1200 development, fixed final epoch4, two epochs RPGE frozen then two epochs RPGE decoder open','baseline_coordinate_alignment':PLAN['baseline_coordinate_alignment'],'independent_test':False,'pretrained_source_overlap_unresolved':True,'results':results,'paired':pairs,'plan_sha256':sha(STORE/'plan.json'),'input_sha256':sha(STORE/'inputs.json'),'truth_manifest_sha256':truth_sha,'paired_errors_sha256':sha(STORE/'paired-errors.npz'),'limitations':['four previously used development dates, fixed historical frame identities','four epochs only; no evidence of convergence or independent generalization','frame bootstrap does not prove temporal independence','plane support and pixel repeatability assumptions are not guaranteed','difference from lidar fine-tuning includes new visual features and re-anchoring; no isolated appearance causal claim']}
report['limitations'].append('Original main sparse concatenation kept by explicit user instruction; repeated encodings previously showed different aligned outputs. Conditional new-module parity does not establish original-main end-to-end repeatability or isolate visual causality.')
atomic(STORE/'report.json',report)
atomic(STORE/'completion.json',{'exit':0,'report_sha256':sha(STORE/'report.json'),'time':time.time(),'frames':len(inputs)})
print(json.dumps(report),flush=True)
