import json
from dataclasses import dataclass, field
from typing import Type

import numpy as np
import torch
from sklearn.cluster import KMeans

from scrstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from scrstudio.data.samplers import BatchRandomSamplerConfig, GlobalFeatSamplerConfig


class TrainingMetadata:
    def __init__(self, poses):
        self.count = len(poses)
        self.metadata = {'cluster_centers': torch.from_numpy(KMeans(n_clusters=50, random_state=0).fit(poses[:, :3, 3].astype(np.float32)).cluster_centers_)}

    def __len__(self):
        return self.count


@dataclass
class GeometryBufferConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: GeometryBuffer)
    graph: str = 'pose_overlap.npz'
    encoding: str = 'pose_n2c.pt'
    batch_size: int = 4096
    exclude_session: str = ''
    geometry_folder: str = 'geometry_training_features'


class GeometryBuffer(DataManager):
    def __init__(self, config, device='cuda', **kwargs):
        super().__init__()
        self.config = config
        self.device = device
        rows = json.loads((config.data / 'manifest.json').read_text())['train']
        poses = np.load(config.data / 'train/poses.npy')
        training = np.array([i for i, row in enumerate(rows) if row['session_id'] != config.exclude_session])
        if len(training) < 50:
            raise ValueError('Not enough training images')
        self.eval_dataset = None
        self.train_dataset = TrainingMetadata(poses[training])
        state = torch.load(config.data / 'train' / config.encoding, weights_only=True)
        global_features = state['model.embedding.weight'].to(device).half()
        if len(global_features) != len(rows):
            raise ValueError('Node2Vec row count does not match manifest')
        self.global_sampler = GlobalFeatSamplerConfig(train_covis_graph=config.graph, neighbor_ratio=.5).setup(
            global_feat=global_features, generator=torch.Generator(device=device).manual_seed(26690), data=config.data)
        chunks = {}
        for i in training:
            name = rows[i]['frame_id'] + '.npz'
            feature = dict(np.load(config.data / 'proc/training_features' / name))
            geometry = dict(np.load(config.data / 'proc' / config.geometry_folder / name))
            if config.exclude_session and any(rows[int(j)]['session_id'] == config.exclude_session for j in geometry.get('support_train_indices', [])):
                geometry['geometry_valid'][:] = False
                geometry['xyz_target_world'][:] = 0
            values = dict(features=feature['features'], target_px=feature['uv'], img_idx=np.full(len(feature['uv']), i, np.int64),
                gt_coords=geometry.pop('xyz_target_world'), **{k: geometry[k] for k in ('geometry_valid', 'geometry_quality', 'sigma_parallel_m', 'sigma_perpendicular_m')})
            for key, value in values.items():
                chunks.setdefault(key, []).append(value)
        self.buffer = {key: torch.from_numpy(np.concatenate(value)).to(device) for key, value in chunks.items()}
        K = np.stack([np.load(config.data / 'proc/training_features' / (row['frame_id'] + '.npz'))['K'] for row in rows])
        self.global_buffer = {key: torch.from_numpy(value.astype(np.float32)).to(device) for key, value in dict(
            gt_poses_inv=np.linalg.inv(poses)[:, :3], intrinsics=K, intrinsics_inv=np.linalg.inv(K)).items()}
        self.sampler = iter(BatchRandomSamplerConfig(batch_size=config.batch_size).setup(dataset_size=len(self.buffer['features']),
            generator=torch.Generator(device=device).manual_seed(10280)))

    def next_train(self, step):
        indices = next(self.sampler)
        frame = self.buffer['img_idx'][indices]
        features = torch.cat([self.global_sampler.sample(frame), self.buffer['features'][indices]], dim=1)
        batch = {key: value[indices] for key, value in self.buffer.items() if key not in ('features', 'img_idx')}
        batch.update({key: value[frame] for key, value in self.global_buffer.items()})
        return {'features': features}, batch

    def get_train_batch_size(self):
        return self.config.batch_size

    def get_param_groups(self):
        return {}
