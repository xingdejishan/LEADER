import json
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch

from data.leader_nclt_manifest import collate_leader_frames,DEFAULT_VOXEL_SIZE,DEFAULT_HORIZONTAL_RESOLUTION,TRAIN_DATES
from models.magic_fusion import polar_voxel_centers
from tools.magic_nclt_runtime import make_matcher,PoseEstimationFailure,sha256_file


def sc2_initial(model,sample,device,seed):
    assert 'T_WB' not in sample
    batch=collate_leader_frames([sample])
    sparse=ME.SparseTensor(features=batch['feats'].to(device),coordinates=batch['coords'].to(device))
    encoded=model.encoder(sparse)
    points=polar_voxel_centers(encoded.C,torch.tensor(encoded.tensor_stride,device=device,dtype=torch.float32),DEFAULT_VOXEL_SIZE,DEFAULT_HORIZONTAL_RESOLUTION)
    prediction=model.decoder(encoded.F)
    count=max(min(50,len(prediction)),int(.5*len(prediction)))
    if count<3:raise PoseEstimationFailure('insufficient correspondences',len(prediction),count)
    selected=prediction[:,3].topk(count).indices
    source=points[selected]
    target=prediction[selected,:3]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    initial=make_matcher(device).estimator(source[None],target[None])[0]
    if not torch.isfinite(initial).all():raise FloatingPointError('nonfinite raw SC2 pose')
    return initial,source,target,len(prediction)


class InitialCache:
    def __init__(self,config,dataset,center):
        manifest=Path(config['train_initial_cache'])
        assert sha256_file(manifest)==config['train_initial_cache_sha256']
        self.root=manifest.parent
        self.cache=json.loads(manifest.read_text())
        cache=self.cache
        assert cache['split']=='train' and cache['ground_truth_used_for_prediction'] is False
        assert cache['backend']=='raw_SC2-PCR_no_full_pool_refine'
        assert cache['base_checkpoint_sha256']==config['base_checkpoint_sha256'] and cache['seed']==config['seed']
        assert cache['manifest_sha256']==dataset.manifest_hashes
        assert set(cache['frames'])==set(record.frame_key for record in dataset.records)
        assert len(cache['frames'])==1200 and set(record.date for record in dataset.records)==set(TRAIN_DATES)
        assert all(config['source_sha256'][name]==digest for name,digest in cache['producer_source_sha256'].items())
        np.testing.assert_array_equal(np.asarray(cache['center_t'],np.float32),center.detach().cpu().numpy())
        for record in dataset.records:
            item=cache['frames'][record.frame_key]
            assert item['scan_sha256']==record.scan_sha256 and item['image_sha256']==record.image_sha256
            assert item['date']==record.date and item['timestamp_us']==record.timestamp_us
            assert sha256_file(self.root/item['file'])==item['sha256']

    def frame(self,key,device):
        item=self.cache['frames'][key]
        path=self.root/item['file']
        assert sha256_file(path)==item['sha256']
        with np.load(path,allow_pickle=False) as archive:
            initial=torch.from_numpy(archive['initial'].copy()).to(device)
            source=torch.from_numpy(archive['source'].copy()).to(device)
            target=torch.from_numpy(archive['target'].copy()).to(device)
        assert initial.shape==(4,4) and source.shape==target.shape and source.ndim==2 and source.shape[1]==3 and len(source)>=3
        assert all(torch.isfinite(value).all() for value in (initial,source,target))
        return initial,source,target
