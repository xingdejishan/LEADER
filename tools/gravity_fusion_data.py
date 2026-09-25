import json

import numpy as np
import torch

from tools.train_local905 import digest


def frames(cache, subset, device='cuda'):
    manifest = json.loads((cache/'manifest.json').read_text())
    allowed = {'source', 'normal', 'image', 'baseline', 'visual_up', 'uncertainty'}
    result = []
    for record in manifest['frames'][subset]:
        path = cache/record['path']
        if digest(path) != record['sha256']:
            raise ValueError('Query input fingerprint changed')
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != allowed:
                raise ValueError('Unexpected query fields')
            item = {key: torch.as_tensor(archive[key], device=device) for key in archive.files}
        item['scan'] = record['scan']
        result.append(item)
    return result


def collate(items):
    count = max(1, max(len(item['source']) for item in items))
    output = {key: torch.stack([item[key] for item in items]) for key in ('baseline', 'visual_up', 'uncertainty')}
    output['valid'] = torch.zeros(len(items), count, dtype=torch.bool, device=output['baseline'].device)
    for key in ('source', 'normal', 'image'):
        value = items[0][key].new_zeros((len(items), count, *items[0][key].shape[1:]))
        for index, item in enumerate(items):
            value[index, :len(item[key])] = item[key]
            output['valid'][index, :len(item[key])] = True
        output[key] = value
    return output
