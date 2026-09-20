import json
from pathlib import Path

import numpy as np

root=Path('/home/zhang/leader-query-backbone-factorial')
rows=json.loads((root/'manifest.json').read_text())
fit=json.loads((root/'pca_indices.json').read_text())
training={r['frame_id'] for r in rows if r['split']=='train'}
assert {r['frame_id'] for r in fit}==training
fit={r['frame_id']:r['indices'] for r in fit}
samples=0
records=[]
for row in rows:
    stem=row['frame_id']
    with np.load(root/'dedode'/(stem+'.npz')) as a, np.load(root/'dino'/(stem+'.npz')) as b:
        mask=a['mask']
        assert np.array_equal(mask,b['mask'])
        assert np.array_equal(a['direction'],b['direction'])
        assert np.array_equal(mask.any(-1),mask[:,:,12])
        for data in [a,b]:
            values=data['image']
            assert values.shape==(*mask.shape,128)
            assert np.isfinite(values).all()
            assert (values[~mask]==0).all()
        if stem in fit:
            for camera,indices in enumerate(fit[stem]):
                assert len(set(indices))==len(indices)<=64
                assert mask[indices,camera,12].all()
                samples+=len(indices)
        records.append(dict(frame_id=stem,voxels=len(mask),center_visible=int(mask[:,:,12].any(-1).sum()),neighborhood_visible=int(mask.any((1,2)).sum())))
result=dict(frames=len(records),training_frames=len(fit),pca_samples=samples,identical_backbone_masks=True,identical_center_neighborhood_voxel_coverage=True,finite_features=True,records=records)
(root/'feature_audit.json').write_text(json.dumps(result,indent=2))
print({k:v for k,v in result.items() if k!='records'})
