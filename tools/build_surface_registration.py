import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.nn import functional as F

from data.local905_query import Local905Query
from models.magic_fusion import project_voxel_centers
from tools.train_local905 import digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def surface(frame):
    points = torch.tensor(frame['points'][0], device='cuda')
    points = points[torch.linalg.vector_norm(points, dim=1) < 40]
    pixels, valid = project_voxel_centers(points, torch.zeros(len(points), device='cuda', dtype=torch.long),
                                         frame['intrinsics'].cuda(), frame['camera_from_lidar'].cuda(),
                                         torch.eye(4, device='cuda')[None], frame['image_bounds'].cuda())
    grid = ((pixels + .5) * (2/1024) - 1).reshape(1, -1, 1, 2)
    mask = frame['image_valid_mask'].cuda()
    support = F.grid_sample(mask, grid, align_corners=False)[0, 0, :, 0]
    valid &= support > .5
    points, pixels, support = points[valid], pixels[valid], support[valid]
    camera = points @ frame['camera_from_lidar'][0, :3, :3].cuda().T + frame['camera_from_lidar'][0, :3, 3].cuda()
    cells = (pixels[:, 1].long() // 4) * 256 + pixels[:, 0].long() // 4
    depth = torch.full((256*256,), float('inf'), device='cuda')
    depth.scatter_reduce_(0, cells, camera[:, 2], reduce='amin', include_self=True)
    keep = camera[:, 2] <= depth[cells] + .5
    points, pixels, support = points[keep], pixels[keep], support[keep]
    xyz = points.cpu().numpy()
    _, index = np.unique(np.floor(xyz/.2).astype(np.int32), axis=0, return_index=True)
    index.sort()
    if len(index) > 4096:
        index = index[np.linspace(0, len(index)-1, 4096, dtype=np.int64)]
    xyz = xyz[index]
    if len(xyz) < 16:
        raise ValueError('Too few visible surface points')
    distance, neighbors = cKDTree(xyz).query(xyz, k=16, workers=4)
    local = xyz[neighbors]
    local = local-local.mean(1, keepdims=True)
    covariance = np.einsum('bni,bnj->bij', local, local)/16
    value, vector = np.linalg.eigh(covariance)
    normals = vector[:, :, 0]
    keep = (value[:, 0]/value.sum(-1).clip(1e-8) < .1) & (distance[:, -1] < 1.5)
    index = torch.tensor(index[keep], device='cuda')
    grid = ((pixels[index] + .5)*(2/1024)-1).reshape(1, -1, 1, 2)
    image = F.grid_sample(frame['sam_features'].cuda()*mask, grid, align_corners=False)[0, :, :, 0].T
    image = image / support[index, None].clamp_min(1e-6)
    return dict(source=xyz[keep].astype(np.float32), normal=normals[keep].astype(np.float32),
                image=image.cpu().numpy().astype(np.float16))


def main():
    parser = argparse.ArgumentParser()
    for name in ('data_root', 'assets', 'pose_cache', 'previous', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    split_path = args.assets/'split_masked.json'
    split = json.loads(split_path.read_text())
    training_split = {**split, 'splits': {'val': split['splits']['train']}}
    write(args.out/'training_inputs.json', training_split)
    baselines = json.loads((args.pose_cache/'baseline_poses.json').read_text())
    truth = np.load(args.pose_cache/'training_poses.npy')
    for subset in ('val', 'test'):
        frozen = json.loads((args.previous/f'L0_{subset}/predictions.json').read_text())
        baselines[subset] = [dict(scan=row['scan'], baseline=row['T_world_body']) for row in frozen['predictions']]
    manifest = dict(split_sha256=digest(split_path), sam_manifest_sha256=digest(args.assets/'sam_cache/manifest_905.json'),
                    reference_training_poses_sha256=digest(args.pose_cache/'training_poses.npy'), frames={})
    pca_mean = pca_matrix = None
    samples = []
    for subset in ('train', 'val', 'test'):
        folder = args.out / f'{subset}_clouds'
        folder.mkdir()
        query = Local905Query(args.data_root, args.out/'training_inputs.json' if subset=='train' else split_path,
                              args.assets/'sam_cache/manifest_905.json', max_points=0,
                              subset='val' if subset=='train' else subset)
        records = []
        with torch.no_grad():
            for fid, key in enumerate(query.keys):
                cloud = surface(query.load(key))
                if subset == 'train':
                    take = np.linspace(0, len(cloud['image'])-1, min(128, len(cloud['image'])), dtype=np.int64)
                    samples.append(cloud['image'][take].astype(np.float32))
                else:
                    cloud['image'] = ((cloud['image'].astype(np.float32)-pca_mean) @ pca_matrix).astype(np.float16)
                path = folder / f'{fid:04d}.npz'
                np.savez(path, **cloud)
                records.append(dict(scan=key, cloud=str(path.relative_to(args.out))))
                if (fid+1)%100 == 0 or fid+1 == len(query.keys):
                    print('surface', subset, fid+1, flush=True)
        if subset == 'train':
            sample = np.concatenate(samples)
            pca_mean = sample.mean(0)
            covariance = (sample-pca_mean).T @ (sample-pca_mean)/len(sample)
            _, vectors = np.linalg.eigh(covariance)
            pca_matrix = vectors[:, -32:]
            np.savez(args.out/'visual_pca.npz', mean=pca_mean, matrix=pca_matrix)
            for record in records:
                path = args.out/record['cloud']
                with np.load(path) as archive:
                    fields = {key: archive[key] for key in archive.files}
                fields['image'] = ((fields['image'].astype(np.float32)-pca_mean)@pca_matrix).astype(np.float16)
                np.savez(path, **fields)
        manifest['frames'][subset] = records
    reference = []
    train_records = manifest['frames']['train']
    dates = np.array([record['scan'].split('/')[1] for record in train_records])
    stamps = np.array([int(Path(record['scan']).stem) for record in train_records], dtype=np.int64)
    for fid, record in enumerate(train_records):
        with np.load(args.out/record['cloud']) as cloud:
            reference.append(dict(world=cloud['source']@truth[fid, :3, :3].T+truth[fid, :3, 3],
                                  normal=cloud['normal']@truth[fid, :3, :3].T, image=cloud['image']))
    for subset, records in manifest['frames'].items():
        folder = args.out/subset
        folder.mkdir()
        for fid, record in enumerate(records):
            baseline_record = baselines[subset][fid]
            if baseline_record['scan'] != record['scan']:
                raise ValueError('Baseline identity differs')
            baseline = np.asarray(baseline_record['baseline'], dtype=np.float32)
            distance = np.linalg.norm(truth[:, :3, 3]-baseline[:3, 3], axis=1)
            stamp, date = int(Path(record['scan']).stem), record['scan'].split('/')[1]
            if subset == 'train':
                distance[(dates==date)&(np.abs(stamps-stamp)<10_000_000)] = np.inf
            selected = []
            for refid in distance.argsort():
                if not np.isfinite(distance[refid]):
                    continue
                if any(dates[refid]==dates[j] and abs(stamps[refid]-stamps[j])<2_000_000 for j in selected):
                    continue
                selected.append(int(refid))
                if len(selected)==4:
                    break
            maps = {key: np.concatenate([reference[i][key] for i in selected]) for key in ('world', 'normal', 'image')}
            with np.load(args.out/record['cloud']) as cloud:
                chosen = np.linspace(0, len(cloud['source'])-1, min(512, len(cloud['source'])), dtype=np.int64)
                source, normal, image = cloud['source'][chosen], cloud['normal'][chosen], cloud['image'][chosen]
            world = source @ baseline[:3, :3].T + baseline[:3, 3]
            distances, index = cKDTree(maps['world']).query(world, k=8, distance_upper_bound=1.5, workers=4)
            valid = np.isfinite(distances)
            index[~valid] = 0
            path = folder/f'{fid:04d}.npz'
            np.savez(path, source=source, source_normal=normal, source_image=image,
                     reference=maps['world'][index], reference_normal=maps['normal'][index],
                     reference_image=maps['image'][index], valid=valid, baseline=baseline)
            record.update(path=str(path.relative_to(args.out)), sha256=digest(path), reference_frames=selected)
            if (fid+1)%100==0 or fid+1==len(records):
                print('pairs', subset, fid+1, flush=True)
    manifest.update(visual_pca_sha256=digest(args.out/'visual_pca.npz'),
                    query_GT_in_pair_cache=False, target='MOE improvement; translation fixed to frozen LEADER',
                    reference_exclusion='Training: same scan and same-date under10s excluded; references separated2s',
                    reference_scope='Only552 training frames with known map poses')
    write(args.out/'manifest.json', manifest)


if __name__ == '__main__':
    main()
