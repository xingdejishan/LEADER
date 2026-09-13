from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch
import torch.nn.functional as F

from scrstudio.data.datasets.camloc_dataset import CamLocDataset, CamLocDatasetConfig
from scrstudio.encoders.base_encoder import PreprocessConfig


@dataclass
class NCLTDatasetConfig(CamLocDatasetConfig):
    _target: Type = field(default_factory=lambda: NCLTDataset)


class NCLTDataset(CamLocDataset):
    def __init__(self, config, preprocess=None, **kwargs):
        if config.augment and config.augment.aug_rotation != 0:
            raise ValueError('NCLT geometry currently requires aug_rotation=0')
        if config.split == 'train':
            options = {} if preprocess is None else {'preprocess': preprocess}
            super().__init__(config, **options, **kwargs)
        else:
            self.config = config
            preprocess = preprocess or PreprocessConfig(mean=None, std=None, grayscale=False, use_half=False, size_multiple=1)
            self.preprocess = preprocess.setup(augment=None, smaller_size=config.smaller_size)
            root = config.data / config.split
            self.rgb_reader = config.rgb.setup(root=root)
            self.rgb_files = self.rgb_reader.file_list
            self.depth_reader = None
            self.calibration_values = np.load(root / config.calib)
            self.pose_values = np.repeat(np.eye(4)[None], len(self.rgb_files), axis=0)
            self.global_feats = np.zeros((len(self.rgb_files), 1), np.float32)
            self.global_feat_dim = 1
            self.metadata = {}
        if self.rgb_files:
            self.valid_mask = torch.from_numpy(np.load(config.data / config.split / 'valid_mask.npy').astype(np.float32))[None, None]

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        mask = F.interpolate(self.valid_mask, size=data['image'].shape[-2:], mode='nearest')[0] > .5
        data['mask'] &= mask
        return data
