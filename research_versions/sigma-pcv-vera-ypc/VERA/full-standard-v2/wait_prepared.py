import json
import os
from pathlib import Path
import time

from common import PLAN,STORE,atomic,sha,guard,verify_source,prohibit_dev_gt

verify_source();guard();prohibit_dev_gt()
prior=Path(PLAN['prepared_store']);start=time.time()
print(json.dumps({'event':'waiting_for_full_preparation','store':str(prior),'original_main_unchanged':True}),flush=True)
while True:
    guard();assert time.time()-start<12*3600,'Preparation wait exceeded 12 hours'
    end=prior/'pipeline-exit.json'
    if end.exists():
        receipt=json.loads(end.read_text());assert receipt['exit']==0 and receipt['phase']=='prepare-only',receipt
        break
    launch=json.loads((prior/'launch.json').read_text());process=Path('/proc')/str(launch['pid'])
    assert process.exists(),'Preparation supervisor exited without a completion receipt'
    assert process.stat().st_uid==1003 and int((process/'stat').read_text().rsplit(')',1)[1].split()[19])==launch['start_ticks']
    time.sleep(30)
assert sha(prior/'plan.json')==PLAN['prepared_plan_sha256']
ready=json.loads((prior/'assets-complete.json').read_text());assert ready['all_camera_inputs']==12000
assert sha(prior/'inputs.json')==ready['input_sha256']
inputs=json.loads((prior/'inputs.json').read_text())
assert len(inputs['train'])==len(inputs['dev'])==1200
selection=json.loads((prior/'dataset-selection.json').read_text())
for split in ('train','dev'):
    assert [r['key'] for r in inputs[split]]==selection[split]
    assert [r['dataset_index'] for r in inputs[split]]==list(range(1200))
    assert all([c['camera'] for c in r['cameras']]==[1,2,3,4,5] for r in inputs[split])
for name in ('inputs.json','dataset-selection.json','resnet18-f37072fd.pth'):
    source=prior/name;dest=STORE/name
    assert not dest.exists();os.link(source,dest)
assert sha(STORE/'resnet18-f37072fd.pth')==ready['resnet_sha256']
atomic(STORE/'assets-complete.json',dict(ready,reused_preparation=True,parent_plan_sha256=sha(prior/'plan.json'),parent_completion_sha256=sha(prior/'assets-complete.json'),parent_exit_sha256=sha(prior/'pipeline-exit.json')))
atomic(STORE/'historical-baseline-reuse.json',{'reuse':False,'reason':'User explicitly requires unmodified original main; inspected historical FAR/V2/TIC contain coordinate alignment and have different actual forward behavior. Recompute L0 with the same original-main runtime as both training arms.','source_audit_sha256':PLAN['cache_baseline_audit_sha256'],'checkpoint_sha256':PLAN['checkpoint_sha256']})
print(json.dumps({'event':'full_preparation_reused','inputs':2400,'camera_inputs':12000,'new_sam_embeddings_this_stage':0}),flush=True)
