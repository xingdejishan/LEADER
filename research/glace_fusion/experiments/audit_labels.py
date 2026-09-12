import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

root=Path('/root/rivermind-data/glace_nclt_corrected_20260912')
meta=json.loads((root/'scene/scene_meta.json').read_text())
E=np.asarray(meta['T_BC_camera_to_body'])
rows=[]
for date in meta['splits']['train']['dates']:
    paths=list(Path('/root/rivermind-data/datasets/NCLT').rglob('groundtruth_'+date+'.csv'))
    data=np.loadtxt(paths[0],delimiter=',')[:,:7]
    data=data[np.isfinite(data).all(axis=1)]
    data=data[np.argsort(data[:,0])]
    _,ix=np.unique(data[:,0],return_index=True);data=data[ix]
    ts=data[:,0];pairs=[p for p in meta['splits']['train']['pairs'] if p['sequence']==date]
    query=np.asarray([p['image_timestamp_us'] for p in pairs])
    j=np.clip(np.searchsorted(ts,query,side='right'),1,len(ts)-1)
    alpha=(query-ts[j-1])/(ts[j]-ts[j-1])
    xyz=data[j-1,1:4]*(1-alpha[:,None])+data[j,1:4]*alpha[:,None]
    rotations=Slerp((ts-ts[0])/1e6,Rotation.from_euler('xyz',data[:,4:7]))((query-ts[0])/1e6).as_matrix()
    expected=np.tile(np.eye(4),(len(pairs),1,1));expected[:,:3,:3]=rotations;expected[:,:3,3]=xyz;expected=expected@E
    actual=np.asarray([np.loadtxt(root/'scene/train/poses'/(p['image']+'.txt')) for p in pairs])
    rows.append({'date':date,'n':len(pairs),'max_pose_matrix_difference':float(np.abs(actual-expected).max()),
                 'gt_bracket_gap_ms_p50_p95_max':np.percentile((ts[j]-ts[j-1])/1000,[50,95,100]).tolist(),
                 'exposure_vs_group_ms_p50_p95_max':np.percentile(np.abs(query-np.asarray([p['group_target_timestamp'] for p in pairs]))/1000,[50,95,100]).tolist()})
    print(json.dumps(rows[-1]),flush=True)
(root/'label_alignment_audit.json').write_text(json.dumps(rows,indent=2))
