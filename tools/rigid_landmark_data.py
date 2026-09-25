import json

import numpy as np
import torch

from tools.train_local905 import digest


def frames(feature_cache, pose_cache, subset, device='cuda'):
    feature_manifest = json.loads((feature_cache / 'manifest.json').read_text())
    pose_manifest = json.loads((pose_cache / 'manifest.json').read_text())
    if (pose_manifest['feature_manifest_sha256'] != digest(feature_cache / 'manifest.json') or
            pose_manifest['baseline_poses_sha256'] != digest(pose_cache / 'baseline_poses.json')):
        raise ValueError('Frozen cache changed')
    poses = json.loads((pose_cache / 'baseline_poses.json').read_text())[subset]
    training_index = training_valid = training_frame = None
    if subset == 'train':
        with np.load(feature_cache / 'training_pairs.npz') as archive:
            training_index, training_valid, training_frame = (
                archive['candidate_index'], archive['candidate_valid'], archive['frame'])
    result = []
    allowed = {'source', 'lidar', 'image', 'predicted', 'valid', 'selected',
               'candidate_index', 'candidate_valid'}
    for fid, (record, pose) in enumerate(zip(feature_manifest['frames'][subset], poses)):
        if record['scan'] != pose['scan']:
            raise ValueError('Pose/query identity differs')
        path = feature_cache / record['path']
        if digest(path) != record['sha256']:
            raise ValueError('Query fingerprint changed')
        with np.load(path) as data:
            if set(data.files) - allowed:
                raise ValueError('Unexpected query fields')
            selected = data['selected']
            selected = selected[data['valid'][selected]]
            if subset == 'train':
                mask = training_frame == fid
                index, valid = training_index[mask], training_valid[mask]
                if len(index) != len(selected):
                    raise ValueError('Training point mapping differs')
            else:
                index, valid = data['candidate_index'][selected], data['candidate_valid'][selected]
            take = np.linspace(0, len(selected)-1, min(256, len(selected)), dtype=np.int64)
            chosen = selected[take]
            item = {key: torch.as_tensor(data[key][chosen], device=device)
                    for key in ('source', 'lidar', 'image', 'predicted')}
            item.update(candidate_index=torch.as_tensor(index[take], device=device),
                        candidate_valid=torch.as_tensor(valid[take], device=device),
                        baseline=torch.tensor(pose['baseline'], dtype=torch.float32, device=device),
                        scan=record['scan'])
            result.append(item)
    return result


def collate(items):
    count = max(1, max(len(item['source']) for item in items))
    batch = {}
    for key in ('source', 'lidar', 'image', 'predicted', 'candidate_index', 'candidate_valid'):
        shape = (len(items), count, *items[0][key].shape[1:])
        tensor = items[0][key].new_zeros(shape)
        for i, item in enumerate(items):
            tensor[i, :len(item[key])] = item[key]
        batch[key] = tensor
    batch['baseline'] = torch.stack([item['baseline'] for item in items])
    batch['point_valid'] = batch['candidate_valid'].any(-1)
    return batch


def predict(model, reference, batch):
    batch_size, points = batch['source'].shape[:2]
    index = batch['candidate_index'].reshape(batch_size * points, 32).long()
    inputs = (batch['lidar'].reshape(-1, 512).float(), batch['image'].reshape(-1, 256).float(),
              batch['predicted'][..., :3].reshape(-1, 3), reference['lidar'][index],
              reference['image'][index], reference['world'][index], reference['error'][index],
              reference['confidence'][index].tanh(), batch['candidate_valid'].reshape(-1, 32))
    return model(batch['source'], batch['baseline'], batch['point_valid'], inputs)
