import json

import numpy as np
import torch

from tools.train_local905 import digest


def frames(cache, subset, device='cuda'):
    manifest = json.loads((cache/'manifest.json').read_text())
    allowed = {'source', 'source_normal', 'source_image', 'reference', 'reference_normal',
               'reference_image', 'valid', 'baseline'}
    result = []
    for record in manifest['frames'][subset]:
        path = cache/record['path']
        if digest(path) != record['sha256']:
            raise ValueError('Query/map pair fingerprint changed')
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != allowed:
                raise ValueError('Unexpected query fields')
            item = {key: torch.as_tensor(archive[key], device=device) for key in archive.files}
        item['scan'] = record['scan']
        result.append(item)
    return result


def collate(items):
    count = max(1, max(len(item['source']) for item in items))
    output = {'baseline': torch.stack([item['baseline'] for item in items])}
    for key in ('source', 'source_normal', 'source_image', 'reference', 'reference_normal', 'reference_image', 'valid'):
        value = items[0][key].new_zeros((len(items), count, *items[0][key].shape[1:]))
        for index, item in enumerate(items):
            value[index, :len(item[key])] = item[key]
        output[key] = value
    return output
