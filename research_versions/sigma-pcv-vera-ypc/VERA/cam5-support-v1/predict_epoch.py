import json
from pathlib import Path
import resource
import sys
import time

from common import PLAN, ROOT, STORE, atomic, gpu_setup, guard,sha,verify_source,pace
device=gpu_setup()
import numpy as np
import torch
from runtime import Dataset,System,pose_from_correspondences,state_dict_sha256

epoch=int(sys.argv[1]);assert epoch in PLAN['evaluate_epochs'];arm='vera'
torch.manual_seed(PLAN['seed']);np.random.seed(PLAN['seed'])
system=System(arm,device)
checkpoint=STORE/'vera'/('epoch-%02d.pt'%epoch)
complete=json.loads((STORE/'vera'/('epoch-%02d-receipt.json'%epoch)).read_text())
assert sha(checkpoint)==complete['checkpoint_sha256']
value=torch.load(checkpoint,map_location='cpu');assert value['epoch']==epoch and value['plan_sha256']==sha(STORE/'plan.json')
system.load_state_dict(value['model'],strict=True);del value
system.eval();before=state_dict_sha256(system.state_dict());dataset=Dataset('dev')
out=STORE/'eval'/('e%02d'%epoch)/arm;out.mkdir(exist_ok=True,parents=True);path=out/'predictions.jsonl'
marker=out/'prediction-start.json'
binding={'checkpoint_sha256':sha(checkpoint),'plan_sha256':sha(STORE/'plan.json'),'inputs_sha256':sha(STORE/'inputs.json')}
if marker.exists():assert json.loads(marker.read_text())==binding
else:atomic(marker,binding)
prior=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
assert [r['key'] for r in prior]==[r['key'] for r in dataset.inputs[:len(prior)]]
failures=sum(r['status']!='success' for r in prior);begin=time.perf_counter()
with path.open('a') as stream,torch.no_grad():
    for index in range(len(prior),len(dataset)):
        frame_start=time.monotonic()
        guard();start=time.perf_counter();record=dataset.inputs[index]
        row={'key':record['key'],'index':index,'status':'success'}
        try:
            batch,extra,_=dataset.load(index,arm=='vera',device);assert 'T' not in batch
            output,points,eta,active=system(batch,extra,record,device)
            seed=PLAN['seed']+record['dataset_index']
            pose,count=pose_from_correspondences(system,output,points,batch,seed,device)
            row.update({'valid_corr':len(points),'selected_corr':count,'active_anchors':0 if active is None else int(active.sum()),'eta_mean_norm':0 if eta is None else float(eta.norm(dim=-1).mean()),'seed':seed})
        except (RuntimeError,ValueError,AssertionError) as error:
            if 'out of memory' in str(error).lower():raise
            pose=np.eye(4);failures+=1;row.update({'status':'failure','error':str(error),'valid_corr':0,'selected_corr':0})
        row.update({'pose':pose.tolist(),'seconds':time.perf_counter()-start})
        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        if (index+1)%20==0:print(json.dumps({'event':'predict','arm':arm,'frames':index+1,'failures':failures}),flush=True)
        pace(frame_start)
assert state_dict_sha256(system.state_dict())==before;verify_source()
atomic(out/'prediction-freeze.json',{'frames':len(dataset),'failures':failures,'prediction_sha256':sha(path),'checkpoint_sha256':sha(checkpoint),'model_state_sha256':before,'plan_sha256':sha(STORE/'plan.json'),'seconds':time.perf_counter()-begin,'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cuda_reserved_peak_bytes':torch.cuda.max_memory_reserved(),'development_gt_reads':0})
print(json.dumps({'event':'prediction_complete','arm':arm,'frames':len(dataset),'failures':failures}),flush=True)
