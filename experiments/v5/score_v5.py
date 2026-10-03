import argparse,json,os,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
from v5_paths import BASE,DATA,STORE,SOURCE,RUNTIME,RAW_GT
sys.path[:0]=[str(SOURCE),str(RUNTIME)]
from tools.far_scoring import DEV_DATES,read_predictions,score_rows,describe,sha256_file
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
os.nice(15)
parser=argparse.ArgumentParser()
parser.add_argument('--epoch',type=int,required=True)
parser.add_argument('--smoke',action='store_true')
args=parser.parse_args()
assert args.epoch in range(5,51,5) or (args.smoke and args.epoch==0)
OUTPUT=STORE/'evaluation-smoke' if args.smoke else STORE/'evaluations'/('epoch{:02d}'.format(args.epoch))
expected_frames=8 if args.smoke else 104732
plan=json.loads((ROOT/'v5-training-plan.json').read_text())
for name,digest in plan['sources'].items():assert sha256_file(ROOT/name)==digest
complete=json.loads((OUTPUT/'completed.json').read_text())
assert complete['complete'] and complete['frames']==expected_frames and complete['frozen_before_GT_evaluation']
path=OUTPUT/'V5'
freeze=json.loads((path/'prediction_freeze.json').read_text())
assert freeze['arm']=='V5' and freeze['frames']==expected_frames and not freeze['GT_read_online'] and freeze['forbidden_GT_attempts']==0
assert sha256_file(path/'predictions.jsonl')==freeze['prediction_sha256']
assert freeze['plan_sha256']==sha256_file(ROOT/'v5-training-plan.json') and freeze['epoch']==args.epoch
expected=[]
truths=[]
hashes={}
counts={}
for date in DEV_DATES:
    public=DATA/'NCLT'/date/'manifest.json'
    assert sha256_file(public)==plan['public_manifest_sha256'][date]
    document=json.loads(public.read_text())
    assert document['split']=='test' and all('ground_truth' not in row for row in document['frames'])
    gt_path=DATA/'evaluator-only-truth'/(date+'.json')
    truth=json.loads(gt_path.read_text())
    hashes[date]=sha256_file(gt_path)
    counts[date]=len(document['frames'])
    T_BC=np.asarray(document['roi']['T_BC'],np.float64)
    for row in document['frames']:
        stamp=str(int(row['timestamp_us']))
        T_WB=np.asarray(truth[stamp]['T_WB'],np.float64)
        T_WC=np.asarray(truth[stamp]['T_WC'],np.float64)
        assert T_WB.shape==(4,4) and np.isfinite(T_WB).all()
        np.testing.assert_allclose(T_WB@T_BC,T_WC,atol=1e-7,rtol=1e-9)
        expected.append(date+'/'+stamp)
        truths.append(T_WB)
assert len(expected)==len(set(expected))==104732
if args.smoke:
    expected=expected[:8]
    truths=truths[:8]
    counts={date:sum(key.startswith(date+'/') for key in expected) for date in DEV_DATES}
rows=read_predictions(path/'predictions.jsonl',expected)
errors=score_rows(rows,truths)
statistics=describe(errors,rows)
per_date={}
offset=0
for date,count in counts.items():
    if count:per_date[date]=describe(errors[offset:offset+count],rows[offset:offset+count])
    offset+=count
np.savez_compressed(OUTPUT/'errors.npz',frame_keys=np.asarray(expected),V5=errors)
report=dict(complete=True,system='LEADER+V5',epoch=args.epoch,frames=expected_frames,smoke_only=args.smoke,full_LiDAR_frames=104735,missing_exact_Cam5=3,GT_protocol='official nearest GT full trajectory',statistics=statistics,per_date=per_date,GT_sha256=hashes,freeze=freeze,plan_sha256=sha256_file(ROOT/'v5-training-plan.json'),evaluator_sha256=sha256_file(__file__),independent_test=False,standalone_LEADER_evaluation=False,LEADER_comparison='existing results only; direct gains require compatible frame and metric protocols')
(OUTPUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps(report))
