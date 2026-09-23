import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def file_sha256(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load_valid_mask(data_root, expected_sha256, resized_size):
    path = Path(data_root) / 'train_scene' / 'train' / 'valid_mask.npy'
    if file_sha256(path) != expected_sha256:
        raise ValueError(f'Local905 valid mask hash mismatch: {path}')
    source = np.load(path, allow_pickle=False)
    if source.shape != (616, 808) or not np.isfinite(source).all():
        raise ValueError(f'Invalid Local905 valid mask: {path}')
    if np.min(source) < 0 or np.max(source) > 1:
        raise ValueError(f'Local905 valid mask outside [0, 1]: {path}')
    width, height = resized_size
    resized = F.interpolate(torch.from_numpy(source.astype(np.float32))[None, None],
                            size=(height, width), mode='nearest')
    padded = F.pad(resized, (0, 1024 - width, 0, 1024 - height))
    return F.avg_pool2d(padded, 16)[0]
