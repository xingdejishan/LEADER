import argparse
import json
import time
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.nn import functional as F

from data.local905_query import Local905Query
from models.magic_fusion import polar_voxel_centers, project_voxel_centers
from models.model_mink import LEADER
from tools.train_local905 import digest


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2) + '\n')


def features(args):
    args.out.mkdir(parents=True, exist_ok=False)
    split_path = args.assets / 'split_masked.json'
    split = json.loads(split_path.read_text())
    train_split = {**split, 'splits': {'val': split['splits']['train']}}
    write(args.out / 'training_inputs.json', train_split)
    checkpoint = torch.load(args.assets / 'official_l0.pt', map_location='cpu')
    model = LEADER(in_channels=3, out_channels=4).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    center = torch.tensor(checkpoint['center_t'], device='cuda')
    metadata = dict(base_checkpoint_sha256=digest(args.assets / 'official_l0.pt'),
                    split_sha256=digest(split_path),
                    sam_manifest_sha256=digest(args.assets / 'sam_cache/manifest_905.json'),
                    center_t=checkpoint['center_t'], frames={}, real_images=True,
                    query_cache_fields=['source', 'lidar', 'image', 'predicted', 'valid', 'selected'])
    for subset in ('train', 'val', 'test'):
        query = Local905Query(args.data_root,
                              args.out / 'training_inputs.json' if subset == 'train' else split_path,
                              args.assets / 'sam_cache/manifest_905.json', max_points=0,
                              subset='val' if subset == 'train' else subset)
        folder = args.out / subset
        folder.mkdir()
        records = []
        with torch.no_grad():
            for i, key in enumerate(query.keys):
                tick = time.perf_counter()
                frame = query.load(key)
                sparse = ME.SparseTensor(torch.as_tensor(frame['feats'], device='cuda'),
                                        ME.utils.batched_coordinates([frame['coords']]).cuda())
                encoded = model.encoder(sparse)
                stride = torch.tensor(encoded.tensor_stride, dtype=torch.float32, device='cuda')
                source = polar_voxel_centers(encoded.C, stride, .2, 1024)
                prediction = model.decoder(encoded.F)
                identity = torch.eye(4, device='cuda')[None]
                pixels, valid = project_voxel_centers(
                    source, encoded.C[:, 0].long(), frame['intrinsics'].cuda(),
                    frame['camera_from_lidar'].cuda(), identity, frame['image_bounds'].cuda())
                grid = ((pixels + .5) * (2 / 1024) - 1).reshape(1, -1, 1, 2)
                mask = frame['image_valid_mask'].cuda()
                support = F.grid_sample(mask, grid, align_corners=False)[0, 0, :, 0]
                valid &= support > .5
                embedding = F.grid_sample(frame['sam_features'].cuda() * mask, grid,
                                           align_corners=False)[0, :, :, 0].T
                embedding = embedding / support[:, None].clamp_min(1e-6)
                embedding[~valid] = 0
                selected = prediction[:, 3].topk(max(min(50, len(prediction)), len(prediction) // 2)).indices
                prediction[:, :3] += center
                filename = folder / f'{i:04d}.npz'
                np.savez(filename, source=source.cpu().numpy(),
                         lidar=encoded.F.cpu().numpy().astype(np.float16),
                         image=embedding.cpu().numpy().astype(np.float16),
                         predicted=prediction.cpu().numpy(), valid=valid.cpu().numpy(),
                         selected=selected.cpu().numpy().astype(np.int32))
                records.append(dict(scan=key, path=str(filename.relative_to(args.out)),
                                    sha256=digest(filename), seconds=time.perf_counter() - tick))
                if (i + 1) % 50 == 0 or i + 1 == len(query.keys):
                    print(subset, i + 1, len(query.keys), flush=True)
        metadata['frames'][subset] = records
        write(args.out / 'manifest.json', metadata)


def candidates(reference, predicted, frame_ids=None, dates=None, stamps=None):
    tree = cKDTree(reference['world'])
    distances, index = tree.query(predicted, k=min(128, len(reference['world'])),
                                 distance_upper_bound=3.0, workers=4)
    output = np.zeros((len(predicted), 32), dtype=np.int64)
    valid = np.zeros_like(output, dtype=bool)
    for row in range(len(predicted)):
        counts, chosen = {}, []
        for distance, candidate in zip(distances[row], index[row]):
            if not np.isfinite(distance):
                continue
            fid = int(reference['frame'][candidate])
            if frame_ids is not None:
                if fid == int(frame_ids[row]):
                    continue
                if (reference['date'][candidate] == dates[row] and
                        abs(reference['stamp'][candidate] - stamps[row]) < 10_000_000):
                    continue
            if counts.get(fid, 0) >= 4:
                continue
            counts[fid] = counts.get(fid, 0) + 1
            chosen.append(candidate)
            if len(chosen) == 32:
                break
        output[row, :len(chosen)] = chosen
        valid[row, :len(chosen)] = True
    return output, valid


def memory(args):
    metadata = json.loads((args.out / 'manifest.json').read_text())
    scene = args.data_root / 'train_scene'
    meta = json.loads((scene / 'scene_meta.json').read_text())
    camera_from_body = np.linalg.inv(np.asarray(meta['T_BC_camera_to_body']))
    reference_parts, training_parts = [], []
    for fid, row in enumerate(metadata['frames']['train']):
        with np.load(args.out / row['path']) as data:
            selected = data['selected']
            index = selected[data['valid'][selected]]
            pose = np.loadtxt(scene / 'train/poses' / (Path(row['scan']).stem + '.txt')) @ camera_from_body
            truth = data['source'][index] @ pose[:3, :3].T + pose[:3, 3]
            error = (truth - data['predicted'][index, :3]).astype(np.float32)
            item = dict(lidar=data['lidar'][index], image=data['image'][index],
                        predicted=data['predicted'][index, :3], error=error, world=truth.astype(np.float32),
                        confidence=data['predicted'][index, 3], frame=np.full(len(index), fid),
                        date=np.full(len(index), int(row['scan'].split('/')[1].replace('-', ''))),
                        stamp=np.full(len(index), int(Path(row['scan']).stem), dtype=np.int64))
            training_parts.append(item)
            reference_index = np.flatnonzero(np.linalg.norm(error, axis=1) <= 3.0)
            reference_parts.append({key: value[reference_index] for key, value in item.items()})
    reference = {key: np.concatenate([x[key] for x in reference_parts]) for key in reference_parts[0]}
    training = {key: np.concatenate([x[key] for x in training_parts]) for key in training_parts[0]}
    candidate_index, candidate_valid = candidates(reference, training['predicted'], training['frame'],
                                                 training['date'], training['stamp'])
    np.savez(args.out / 'reference_map.npz', **reference)
    np.savez(args.out / 'training_pairs.npz', **training, candidate_index=candidate_index,
             candidate_valid=candidate_valid)
    for subset in ('val', 'test'):
        for row in metadata['frames'][subset]:
            path = args.out / row['path']
            with np.load(path) as data:
                fields = {key: data[key] for key in data.files}
            index, valid = candidates(reference, fields['predicted'][:, :3])
            fields.update(candidate_index=index, candidate_valid=valid & fields['valid'][:, None])
            np.savez(path, **fields)
            row['sha256'] = digest(path)
    metadata.update(reference_map_sha256=digest(args.out / 'reference_map.npz'),
                    training_pairs_sha256=digest(args.out / 'training_pairs.npz'),
                    reference_points=len(reference['world']), training_points=len(training['world']),
                    training_points_with_candidates=int(candidate_valid.any(-1).sum()),
                    reference_scope='Only 552 training scans; training pairs exclude same scan and within 10s same date',
                    candidate_policy='Nearest known map coordinates within 3m of LiDAR prediction, <=4 per frame, <=32 total')
    write(args.out / 'manifest.json', metadata)
    print(json.dumps({key: value for key, value in metadata.items() if key != 'frames'}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    features(args)
    memory(args)


if __name__ == '__main__':
    main()
