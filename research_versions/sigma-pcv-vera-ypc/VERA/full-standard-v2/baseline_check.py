import json
from pathlib import Path

from common import PLAN,STORE,sha,atomic,gpu_setup,guard
device=gpu_setup()
import numpy as np
import torch
from runtime import Dataset,System,pose_from_correspondences

old=Path('/home/zhangcy/projects/leader-mma-only-20260929/run-v1/l0')
freeze=json.loads((old/'prediction_freeze.json').read_text())
assert freeze['checkpoint_sha256']==PLAN['checkpoint_sha256']
assert sha(old/'predictions.jsonl')==freeze['prediction_sha256']=='db15800cc5d600b55f5f39eec425088b2b68a588fa967b8673b99873e9b756bc'
rows={r['frame_key']:r for r in (json.loads(line) for line in (old/'predictions.jsonl').read_text().splitlines())}
assert len(rows)==1200
system=System('l0',device).eval();data=Dataset('dev');checks=[]
with torch.no_grad():
    for i in (0,300,600,900):
        batch,extra,record=data.load(i,False,device)
        output,points,_,_=system(batch,extra,record,device)
        pose,count=pose_from_correspondences(system,output,points,batch,PLAN['seed']+record['dataset_index'],device)
        original=rows[record['key']]
        difference=float(np.max(np.abs(pose-np.array(original['T_WB_pred']))))
        checks.append({'key':record['key'],'pose_max_abs':difference,'valid_corr':len(points),'old_valid_corr':original['valid_corr'],'selected_corr':count,'old_selected_corr':original['selected_corr']})
        guard()
reuse=all(r['pose_max_abs']<1e-5 and r['valid_corr']==r['old_valid_corr'] and r['selected_corr']==r['old_selected_corr'] for r in checks)
atomic(STORE/'baseline-parity.json',{'old':str(old),'old_prediction_sha256':sha(old/'predictions.jsonl'),'old_freeze_sha256':sha(old/'prediction_freeze.json'),'checks':checks,'reuse':reuse,'reason':'same fixed frame identities and seeds; reuse only if main four-frame parity passes'})
if reuse:
    out=STORE/'l0';out.mkdir(exist_ok=False);path=out/'predictions.jsonl'
    subset=[]
    for index,f in enumerate(data.inputs):
        r=rows[f['key']];subset.append({'key':f['key'],'index':index,'pose':r['T_WB_pred'],'status':r['status'],'valid_corr':r['valid_corr'],'selected_corr':r['selected_corr'],'historical_reuse':True,'seed':PLAN['seed']+f['dataset_index']})
    path.write_text(''.join(json.dumps(r)+'\n' for r in subset))
    atomic(out/'prediction-freeze.json',{'frames':len(subset),'failures':sum(r['status']!='success' for r in subset),'prediction_sha256':sha(path),'checkpoint_sha256':PLAN['checkpoint_sha256'],'plan_sha256':sha(STORE/'plan.json'),'seconds':None,'historical_reuse':True,'original_prediction_sha256':sha(old/'predictions.jsonl'),'parity_sha256':sha(STORE/'baseline-parity.json')})
print(json.dumps({'event':'baseline_checked','reuse':reuse,'checks':checks}),flush=True)
