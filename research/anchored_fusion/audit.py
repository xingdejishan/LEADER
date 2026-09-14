import json
import numpy as np
from experiment import OUT, CACHE, HERE, run

protocol = json.loads((OUT/'protocol.json').read_text())
rows = protocol['rows']
assert len({r['frame_id'] for r in rows})==905
fit = [r for r in rows if r['role']=='fit']
assert len(fit)==578
expected = json.loads((HERE.parent/'benefit_modulation/results/crossframe_probe/provenance.json').read_text())
for r in expected['frames']:
    for kind in ['lidar','visual_raw','mapping']:
        assert run.digest(CACHE/kind/(r['frame_id']+'.npz'))==r[kind]
positions, poses, owners = [],[],[]
for index,row in enumerate(fit):
    with np.load(CACHE/'lidar'/(row['frame_id']+'.npz')) as l, np.load(CACHE/'visual_raw'/(row['frame_id']+'.npz')) as v:
        positions.append(l['representative_world'][v['valid']])
        poses.append(l['camera_pose'])
        owners.extend([index]*int(v['valid'].sum()))
positions,poses,owners = np.concatenate(positions),np.stack(poses),np.array(owners)
candidate_hashes = {}
for row in rows:
    file = OUT/'candidates'/(row['frame_id']+'.npz')
    with np.load(file) as a, np.load(CACHE/'lidar'/file.name) as l:
        candidates = a['candidates']
        assert candidates.shape[1]==16 and (np.diff(np.sort(candidates,axis=1),axis=1)>0).all()
        spatial = np.linalg.norm(positions[candidates]-l['representative_world'][a['indices'],None],axis=-1)
        assert np.array_equal(spatial<=.5,a['positive'])
        assert np.array_equal(spatial>=2,a['negative'])
        pose = l['camera_pose']
        delta = np.linalg.norm(poses[:,:3,3]-pose[:3,3],axis=1)
        angle = np.rad2deg(np.arccos(np.clip((np.einsum('nij,ij->n',poses[:,:3,:3],pose[:3,:3])-1)/2,-1,1)))
        excluded = (delta<5)&(angle<15)
        excluded |= np.array([r['session_id']==row['session_id'] and abs(int(r['frame_id'])-int(row['frame_id']))<10000000 for r in fit])
        assert not excluded[owners[candidates]].any()
    candidate_hashes[row['frame_id']] = run.digest(file)
run.save_json(OUT/'audit.json',dict(passed=True,immutable_cache_frames=905,fit_only_reference_frames=578,
    unique_and_geometry_checked_frames=905,candidate_sha256=candidate_hashes,
    checkpoint_sha256=run.digest(run.WORKSPACE/'research/image_gate_checkpoint/model.safetensors'),
    original_cache_provenance_sha256=run.digest(HERE.parent/'benefit_modulation/results/crossframe_probe/provenance.json')))
print('PASS: 905 immutable caches; fit-only reference library; distinct candidates, exclusion and spatial labels independently checked')
