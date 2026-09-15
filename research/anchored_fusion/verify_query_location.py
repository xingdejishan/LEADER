import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

import query_location as q


def main():
    p=json.loads((q.OUT/'protocol.json').read_text()); xyz={}; counts={}
    for r in p['rows']:
        fid=r['frame_id']
        with np.load(q.e.CACHE/'lidar'/(fid+'.npz')) as l,np.load(q.e.CACHE/'visual_raw'/(fid+'.npz')) as v:
            valid=v['valid']; xyz[fid]=l['source'][valid].astype(float)@l['GT'][:3,:3].astype(float).T+l['GT'][:3,3]
            counts[fid]=int(valid.sum())
    fit=np.concatenate([xyz[r['frame_id']] for r in p['rows'] if r['role']=='fit'])
    centers=np.load(q.OUT/'centroids.npz')['centroids']
    again=KMeans(n_clusters=25,n_init=10,max_iter=300,random_state=2089).fit(fit).cluster_centers_
    distances=np.linalg.norm(centers[:,None]-again[None],axis=-1); a,b=linear_sum_assignment(distances)
    error=float(distances[a,b].max()); assert error<1e-8
    dev=np.concatenate([xyz[r['frame_id']] for r in p['rows'] if r['role']=='development'])
    labels=np.linalg.norm(dev[:,None]-centers[None],axis=-1).argmin(1)
    near=cKDTree(fit).query(dev)[0]
    with np.load(q.OUT/'development.npz') as f:
        assert np.array_equal(labels,f['target'])
        assert np.array_equal(near<=2,f['supported'])
        for arm in ['L','V','S','frequency']:
            assert np.isfinite(f[arm]).all() and np.allclose(np.exp(f[arm]).sum(1),1,atol=1e-6)
    records={}
    for arm in ['L','V','S']:
        log=json.loads((q.OUT/arm/'training.json').read_text()); selected=json.loads((q.OUT/arm/'selection.json').read_text())
        assert [r['epoch'] for r in log]==list(range(1,101))
        assert selected['epoch']==max(selected['all_choices'],key=lambda x:x['macro'])['epoch']
        state=torch.load(q.OUT/arm/'best.pt',map_location='cpu'); assert sum(v.numel() for v in state.values())==41451
        records[arm]=dict(epoch=selected['epoch'],parameters=41451,checkpoint_sha256=q.e.run.digest(q.OUT/arm/'best.pt'))
    result=dict(training_only_clustering_reproduced=True,max_center_error_m=error,development_labels_reproduced=True,coverage_reproduced=True,normalized_posteriors=True,fit_points=len(fit),development_points=len(dev),models=records,
        initial_launch='Class-index dtype fixed before first optimizer step; restarted with same fixed seed/configuration; no training or evaluation-driven changes.')
    q.e.run.save_json(q.OUT/'verification.json',result)
    print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    main()
