import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .prepare import save_json


def freeze_inputs(root):
    data = root / 'data'
    manifest = json.loads((data / 'manifest.json').read_text())
    artifacts = [data / 'manifest.json', data / 'scene_meta.json', data / 'proc/pcad3LB_128.pth', data / 'proc/geometry_report.json']
    for graph in ('pose', 'lidar'):
        encoding = data / 'train' / (graph + '_n2c.pt')
        values = torch.load(encoding, weights_only=True)['model.embedding.weight']
        assert values.shape == (907, 256) and torch.isfinite(values).all()
        artifacts.extend([encoding, data / 'train' / (graph + '_overlap.npz')])
    for split, expected in [('train', 907), ('val', 303), ('test', 148)]:
        assert len(manifest[split]) == expected
        retrieval = data / split / 'netvlad_feats.npy'
        features = np.load(retrieval)
        assert len(features) == expected and np.isfinite(features).all()
        artifacts.extend([retrieval, data / split / 'calibration.npy', data / split / 'poses.npy', data / split / 'valid_mask.npy'])
    artifacts.append(data / 'train/netvlad_feats_pq.pkl')
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}
    source = Path(__file__).parent
    code = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.glob('*.py')}
    save_json(root / 'frozen_inputs.json', dict(artifacts=hashes, code_at_freeze=code, source=json.loads((source / 'SOURCE.json').read_text()),
        warning='Hashes preserve provenance; testing references remain separate from inference inputs'))
