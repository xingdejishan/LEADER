import json
from pathlib import Path
from common import PLAN,STORE,atomic,guard,verify_source
import numpy as np
from runtime import Dataset
from vera import geometry

guard();verify_source();data=Dataset('train');inputs=data.inputs;rows=[]
for start in (0,300,600,900):
    for offset in (0,99,199,299):
        guard();record=inputs[start+offset]
        sample=data.base[record['dataset_index']]
        path=Path(PLAN['parent_root'])/'storage/geometry'/(record['key'].replace('/','_')+'.npz')
        with np.load(path,allow_pickle=False) as saved:p=saved['points'];old_active=saved['active']
        import torch
        g=geometry(torch.from_numpy(p),sample['points'],record['cameras'])
        active=g['active'];assert np.array_equal(g['valid'],g['valid'][:,::-1])
        assert np.isfinite(g['support_points']).all() and np.isfinite(g['basis']).all()
        assert (g['support_distance'][active]<=PLAN['support_max_distance_m']+1e-5).all()
        assert (g['valid'][active].sum(1)>1).all()
        assert np.max(np.abs(g['eta']+g['eta'][::-1]))<1e-6
        shift=g['support_points'][active]-p[active]
        rows.append({'key':record['key'],'voxels':len(p),'old_active':int(old_active.sum()),'adapted_active':int(active.sum()),'fixed_support_shift_mean_m':float(np.linalg.norm(shift,axis=1).mean()) if active.any() else 0.,'visible_candidates_per_active':float(g['valid'][active].sum(1).mean()) if active.any() else 0.})
atomic(STORE/'cpu-checks.json',{'scope':'16 fixed Train support/FOV geometry checks; Dataset constructor reads Train GT but values not used here; no encoder/GPU or Dev GT; activation not accuracy','rows':rows,'old_active':sum(r['old_active'] for r in rows),'adapted_active':sum(r['adapted_active'] for r in rows),'voxels':sum(r['voxels'] for r in rows)})
print(json.dumps({'passed':True,'frames':len(rows),'old_active':sum(r['old_active'] for r in rows),'adapted_active':sum(r['adapted_active'] for r in rows)}))
