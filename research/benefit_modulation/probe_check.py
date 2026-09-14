import json
import numpy as np
from probe_data import OUT, run

rows = json.loads((OUT/'manifest.json').read_text())
refs = [r for r in rows if r['split']=='reference']
queries = {r['frame_id']:r for r in rows if r['split']=='query'}
records = json.loads((OUT/'probe/frames.json').read_text())
world, poses, frame_ids = [], [], []
for i,r in enumerate(refs):
    with np.load(OUT/'lidar'/(r['frame_id']+'.npz')) as l, np.load(OUT/'visual_raw'/(r['frame_id']+'.npz')) as v:
        world.append(l['representative_world'][v['valid']])
        poses.append(l['camera_pose'])
        frame_ids.extend([i]*int(v['valid'].sum()))
world,poses,frame_ids = np.concatenate(world),np.stack(poses),np.array(frame_ids)
counts = 0
for r in records:
    name = r['frame_id']
    row = queries[name]
    with np.load(OUT/'probe'/(name+'.npz')) as a, np.load(OUT/'lidar'/(name+'.npz')) as l:
        candidates = a['candidate_global_indices']
        assert (np.diff(np.sort(candidates,axis=1),axis=1)>0).all()
        distance = np.linalg.norm(world[candidates]-l['representative_world'][a['query_voxel'],None],axis=-1)
        assert np.array_equal(distance,a['spatial_distance'])
        frames = frame_ids[candidates]
        qpose = l['camera_pose']
        delta = np.linalg.norm(poses[:,:3,3]-qpose[:3,3],axis=1)
        angle = np.rad2deg(np.arccos(np.clip((np.einsum('nij,ij->n',poses[:,:3,:3],qpose[:3,:3])-1)/2,-1,1)))
        excluded = (delta<5)&(angle<15)
        excluded |= np.array([x['session_id']==row['session_id'] and abs(int(x['frame_id'])-int(name))<10000000 for x in refs])
        assert not excluded[frames].any()
        positives = distance<=.5
        mask = a['ambiguous']
        assert (positives.any(1)[mask]).all() and ((distance>=2).any(1)[mask]).all()
        for key,value in r['hits'].items():
            hits = positives[np.arange(len(positives)),a[key]]
            assert int(hits[mask].sum())==value
        counts += int(mask.sum())
run.save_json(OUT/'probe/check.json',dict(passed=True,frames=len(records),ambiguous=counts,
    checks=['16 distinct fixed candidates','recomputed Cartesian world-distance labels','no excluded near-repeat candidates','independent top1 numerator reconstruction']))
print('PASS',len(records),counts)
