import json
from pathlib import Path
import time
from common import PLAN,STORE,atomic,guard,sha,verify_source

guard();verify_source();b=PLAN['leader_finetune_baseline'];inputs=json.loads((STORE/'inputs.json').read_text())['dev']
for k,h in [('prediction','prediction_sha256'),('freeze_path','freeze_sha256'),('best_checkpoint','best_checkpoint_sha256'),('evaluation_path','evaluation_sha256')]:assert sha(b[k])==b[h]
rows=[json.loads(x) for x in Path(b['prediction']).read_text().splitlines()]
assert [r['frame_key'] for r in rows]==[r['key'] for r in inputs]
out=STORE/'leader_finetune';out.mkdir(exist_ok=True);path=out/'predictions.jsonl'
if not path.exists():
    with path.open('x') as f:
        for r in rows:
            canonical={'key':r['frame_key'],'pose':r['T_WB_pred'],'status':'success' if r.get('status')=='success' else 'failure'}
            f.write(json.dumps(canonical,allow_nan=False)+'\n')
assert len(path.read_text().splitlines())==1200
atomic(out/'prediction-freeze.json',{'frames':1200,'prediction_sha256':sha(path),'checkpoint_sha256':b['best_checkpoint_sha256'],'development_gt_reads':0,'historical_reuse':True,'parent_prediction_sha256':b['prediction_sha256'],'parent_freeze_sha256':b['freeze_sha256'],'failures':sum(r.get('status')!='success' for r in rows),'seconds':0,'time':time.time(),'parent':b})
print(json.dumps({'reused':True,'frames':1200,'checkpoint_epoch':b['best_epoch'],'prediction_sha256':sha(path)}))
