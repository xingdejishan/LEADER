import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from .mapping import sha256


def save_arrays(path, arrays, metadata):
    path = Path(path)
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **{k:v.detach().cpu().numpy() if torch.is_tensor(v) else v for k,v in arrays.items()})
    record = dict(metadata, sha256=sha256(path))
    path.with_suffix('.json').write_text(json.dumps(record, indent=2), encoding='utf8')


def load_arrays(path, online=False):
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    if sha256(path) != meta['sha256']:
        raise ValueError('input archive SHA mismatch')
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 256 * 1024**2:
            raise ValueError('input exceeds the 256 MiB memory budget')
    with np.load(path, allow_pickle=False) as z:
        if online and any(k in z.files for k in ('T_WB','gt_pose','ground_truth','target_pose')):
            raise ValueError('GT found in online input')
        if sum(z[k].nbytes for k in z.files) > 256 * 1024**2:
            raise ValueError('input exceeds the 256 MiB memory budget')
        arrays = {k:torch.as_tensor(z[k].copy(), dtype=torch.long if z[k].dtype.kind in 'iu' else torch.float64) for k in z.files}
    return arrays, meta


def candidate_indices(model, frame, ray_map, frame_meta, map_meta, count=4, max_distance=150.):
    if frame_meta.get('frame_key') in map_meta.get('reference_frames', []):
        raise ValueError('query image is a reference-map image; self-matching forbidden')
    with torch.no_grad():
        query = model.encode(frame['descriptors'])
        reference = model.encode(ray_map['descriptors'])
        scores = query @ reference.T
        if 'reference_centers' in ray_map:
            local = (ray_map['reference_centers'] - frame['T0'][:3, 3]).norm(dim=-1) < max_distance + ray_map['reference_radii']
        else:
            local = torch.ones(len(reference), dtype=torch.bool, device=reference.device)
        if int(local.sum()) == 0:
            return torch.full((len(query), count), -1, dtype=torch.long)
        scores[:, ~local] = -torch.inf
        values, ids = scores.topk(min(count, len(reference)), dim=-1)
        ids[~torch.isfinite(values)] = -1
        if ids.shape[1] < count:
            ids = torch.cat((ids, ids.new_full((len(query), count-ids.shape[1]), -1)), -1)
    return ids


def load_model(path=None, map_sha256=None):
    from .model import RayBundleCorrespondence
    model = RayBundleCorrespondence().double()
    if path:
        checkpoint = torch.load(str(path), map_location='cpu')
        if map_sha256 is not None and checkpoint.get('map_sha256') != map_sha256:
            raise ValueError('model belongs to a different reference map')
        model.load_state_dict(checkpoint['model'], strict=True)
    return model
