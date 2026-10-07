import argparse
import json
import sys
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--predictions',type=Path,required=True)
parser.add_argument('--gt',type=Path,required=True)
args=parser.parse_args()
import numpy as np
import portable_runtime as r
sys.path.insert(0,str(r.ROOT/'evaluation'))
from nclt_reference_policy import score_poses

freeze=r.read(args.predictions/'prediction-freeze.json')
assert freeze['complete'] and freeze['frozen_before_scoring'] and freeze['GT_online_attempts']==0
rows=[]
for item in freeze['chunks']:
    path=args.predictions/item['file']
    assert r.sha(path)==item['sha256']
    chunk=r.read(path)
    assert chunk['binding']==freeze['binding'] and chunk['GT_online_attempts']==0
    rows.extend(chunk['rows'])
assert len(rows)==len({row['frame_key'] for row in rows})==freeze['binding']['frames']==104732
with np.load(args.gt,allow_pickle=False) as source:
    keys=source['frame_keys'].astype(str)
    assert len(keys)==len(set(keys))
    index={key:i for i,key in enumerate(keys)}
    poses=np.asarray([source['poses'][index[row['frame_key']]] for row in rows])
dates=[row['frame_key'].split('/')[0] for row in rows]
timestamps=np.asarray([int(row['frame_key'].split('/')[1]) for row in rows],dtype=np.int64)
predictions=np.asarray([row['T_WB_pred'] for row in rows],dtype=np.float64)
policy=r.ROOT/'evaluation/nclt-reference-policy.json'
protocol=r.read(r.ROOT/'protocol.json')
assert r.sha(policy)==protocol['reference_policy_sha256']
assert r.sha(policy.parent/'nclt-2012-02-12-suspect255-odometry.npz')==protocol['replacement_reference_sha256']
report,position,angle,replaced=score_poses(dates,timestamps,predictions,poses,policy)
report.update(complete=True,prediction_freeze_sha256=r.sha(args.predictions/'prediction-freeze.json'),evaluator_gt_sha256=r.sha(args.gt),position_P90_m=float(np.percentile(position,90)),angle_P90_deg=float(np.percentile(angle,90)),joint_1m_2deg=int(((position<1)&(angle<2)).sum()),per_date={})
for date in sorted(set(dates)):
    selected=np.asarray(dates)==date
    report['per_date'][date]=dict(frames=int(selected.sum()),MPE_m=float(position[selected].mean()),MOE_deg=float(angle[selected].mean()))
path=args.predictions/'report.json'
assert not path.exists()
path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(report))
