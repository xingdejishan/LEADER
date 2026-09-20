import hashlib
import json
import shutil
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.spatial import cKDTree
from tqdm import tqdm

from scrstudio.data.utils.readers import folder2lmdb
from scrstudio.encoders.dedode_encoder import DedodeEncoderConfig
from scrstudio.encoders.base_encoder import ImageAugmentConfig

from .dataset import NCLTDatasetConfig
from .geometry import surface_targets, visible_voxels


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    temporary.replace(path)


def prepare_data(bundle, output):
    output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((output / 'manifest.json').read_text()) if (output / 'manifest.json').exists() else None
    manifest = {}
    for split, relative, expected in [('train', 'train_scene/train', 907), ('val', 'validation_scene/train', 303), ('test', 'test_scene/test', 148)]:
        source = bundle / 'data' / relative
        images = sorted((source / 'rgb').glob('*.jpg'))
        assert len(images) == expected, (split, len(images))
        destination = output / split
        destination.mkdir(exist_ok=True)
        poses = np.stack([np.loadtxt(source / 'poses' / (p.stem + '.txt')) for p in images])
        calibration = np.stack([np.loadtxt(source / 'calibration' / (p.stem + '.txt')) for p in images])
        for name, values in [('poses.npy', poses), ('calibration.npy', calibration)]:
            if (destination / name).exists() and not np.array_equal(np.load(destination / name), values):
                raise ValueError(f'Input changed after preparation: {split}/{name}; use a new run directory')
        if previous is not None:
            current_images = [hashlib.sha256(p.read_bytes()).hexdigest() for p in images]
            if current_images != [row['image_sha256'] for row in previous[split]]:
                raise ValueError(f'Images changed after preparation: {split}; use a new run directory')
        if not (destination / 'rgb_lmdb').exists():
            folder2lmdb(source / 'rgb', destination / 'rgb_lmdb')
        names = (destination / 'rgb_lmdb/file_list.txt').read_text().splitlines()
        assert names == [p.name for p in images]
        np.save(destination / 'calibration.npy', calibration)
        np.save(destination / 'poses.npy', poses)
        shutil.copy2(bundle / 'data/valid_mask.npy', destination / 'valid_mask.npy')
        manifest[split] = [dict(frame_id=p.stem, session_id=datetime.fromtimestamp(int(p.stem) / 1e6, timezone.utc).date().isoformat(),
            image_timestamp=int(p.stem), camera_id=5, image_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            geometry_path=str(source / 'lidar_world' / (p.stem + '.npy')) if split == 'train' else None) for p in images]
    assert not (set(r['frame_id'] for r in manifest['train']) & set(r['frame_id'] for r in manifest['val'] + manifest['test']))
    save_json(output / 'manifest.json', manifest)
    shutil.copy2(bundle / 'data/train_scene/scene_meta.json', output / 'scene_meta.json')
    return manifest


def prepare_features(data, seed=2089):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    proc = data / 'proc'
    proc.mkdir(exist_ok=True)
    encoder = DedodeEncoderConfig(detector='L', descriptor='B', k=5000).setup().cuda().eval()
    manifest = json.loads((data / 'manifest.json').read_text())
    if not (proc / 'pcad3LB_128.pth').exists():
        dataset = NCLTDatasetConfig(data=data, split='train').setup(preprocess=encoder.preprocess)
        total = torch.zeros(256, dtype=torch.float64, device='cuda')
        gram = torch.zeros(256, 256, dtype=torch.float64, device='cuda')
        count = 0
        raw = proc / 'raw'
        raw.mkdir(exist_ok=True)
        for index in tqdm(range(len(dataset)), desc='DeDoDe + training PCA'):
            sample = dataset[index]
            with torch.inference_mode(), torch.autocast('cuda'):
                result = encoder.keypoint_features({k: sample[k][None].cuda() for k in ('image', 'mask')})
            features = result['descriptors'].double()
            total += features.sum(0)
            gram += features.T @ features
            count += len(features)
            np.savez(raw / (manifest['train'][index]['frame_id'] + '.npz'),
                uv=result['keypoints'].float().cpu().numpy(), descriptors=result['descriptors'].half().cpu().numpy(),
                scores=result['keypoint_scores'].float().cpu().numpy(), K=sample['intrinsics'].numpy(), image_size_hw=np.array(sample['image'].shape[-2:]))
        mean = total / count
        covariance = (gram - count * mean[:, None] @ mean[None]) / (count - 1)
        values, vectors = torch.linalg.eigh(covariance)
        weight = vectors[:, -128:].T.flip(0).float()
        bias = -(weight @ mean.float())
        torch.save(dict(weight=weight[:, :, None, None].cpu(), bias=bias.cpu()), proc / 'pcad3LB_128.pth')
        save_json(proc / 'pca.json', dict(samples=count, dimensions=128, explained_variance=float(values[-128:].sum() / values.sum()),
            implementation='Exact covariance eigendecomposition in float64; same centered, unwhitened PCA objective as official cuML', fitting_split='train'))
    state = torch.load(proc / 'pcad3LB_128.pth', weights_only=True)
    weight, bias = state['weight'].reshape(128, 256).cuda(), state['bias'].cuda()
    for split in ('train', 'val', 'test'):
        dataset = NCLTDatasetConfig(data=data, split=split).setup(preprocess=encoder.preprocess)
        destination = proc / ('features_' + split)
        destination.mkdir(exist_ok=True)
        for index in tqdm(range(len(dataset)), desc='Compressed features ' + split):
            frame = manifest[split][index]['frame_id']
            target = destination / (frame + '.npz')
            if target.exists():
                continue
            raw_path = proc / 'raw' / (frame + '.npz')
            if split == 'train' and raw_path.exists():
                cached = dict(np.load(raw_path))
                descriptors = torch.from_numpy(cached.pop('descriptors').astype(np.float32)).cuda()
            else:
                sample = dataset[index]
                with torch.inference_mode(), torch.autocast('cuda'):
                    result = encoder.keypoint_features({k: sample[k][None].cuda() for k in ('image', 'mask')})
                descriptors = result['descriptors'].float()
                cached = dict(uv=result['keypoints'].float().cpu().numpy(), scores=result['keypoint_scores'].float().cpu().numpy(),
                    K=sample['intrinsics'].numpy(), image_size_hw=np.array(sample['image'].shape[-2:]))
            cached['features'] = (descriptors @ weight.T + bias).half().cpu().numpy()
            np.savez(target, **cached)
    augmented = proc / 'training_features'
    augmented.mkdir(exist_ok=True)
    dataset = NCLTDatasetConfig(data=data, split='train', augment=ImageAugmentConfig(aug_rotation=0)).setup(preprocess=encoder.preprocess)
    for index in tqdm(range(len(dataset)), desc='Augmented training buffer'):
        frame = manifest['train'][index]['frame_id']
        target = augmented / (frame + '.npz')
        if target.exists():
            continue
        random.seed(seed + index)
        torch.manual_seed(seed + index)
        sample = dataset[index]
        with torch.inference_mode(), torch.autocast('cuda'):
            result = encoder.keypoint_features({k: sample[k][None].cuda() for k in ('image', 'mask')}, n=1024)
        np.savez(target, features=(result['descriptors'].float() @ weight.T + bias).half().cpu().numpy(),
            uv=result['keypoints'].float().cpu().numpy(), scores=result['keypoint_scores'].float().cpu().numpy(),
            K=sample['intrinsics'].numpy(), T_WC=sample['pose'].numpy(), image_size_hw=np.array(sample['image'].shape[-2:]))


def prepare_geometry(data):
    report_path = data / 'proc/geometry_report.json'
    if report_path.exists() and json.loads(report_path.read_text()).get('supervision_version') == 'train-surfaces-v4':
        if all(len(list((data / 'proc' / name).glob('*.npz'))) == 907 for name in ('geometry_training_features', 'geometry_features_train')):
            return
    manifest = json.loads((data / 'manifest.json').read_text())['train']
    poses = np.load(data / 'train/poses.npy')
    calibration = np.load(data / 'train/calibration.npy')
    image_mask = np.load(data / 'train/valid_mask.npy')
    nearby = cKDTree(poses[:, :3, 3])

    @lru_cache(maxsize=16)
    def cloud(index):
        path = Path(manifest[index]['geometry_path'])
        return np.load(path) if path.exists() else np.empty((0, 3), np.float32)

    def local_map(index):
        distance, indices = nearby.query(poses[index, :3, 3], k=9, distance_upper_bound=5.)
        indices = indices[np.isfinite(distance)]
        clouds = [cloud(int(i)) for i in indices]
        votes = np.concatenate([np.unique(np.floor(points / .5).astype(np.int64), axis=0) for points in clouds])
        voxel_ids, counts = np.unique(votes, axis=0, return_counts=True)
        persistent = set(map(tuple, voxel_ids[counts >= 2].tolist()))
        world = np.concatenate(clouds)
        if len(world):
            _, unique = np.unique(np.floor(world / .1).astype(np.int64), axis=0, return_index=True)
            world = world[unique]
            world = world[np.array([tuple(voxel) in persistent for voxel in np.floor(world / .5).astype(np.int64)], dtype=bool)]
        return world, indices
    statistics = []
    visible = []
    coverage = []
    for kind in ('training_features', 'features_train'):
        destination = data / 'proc' / ('geometry_' + kind)
        destination.mkdir(exist_ok=True)
        for index, row in enumerate(tqdm(manifest, desc='Surface geometry ' + kind)):
            path = destination / (row['frame_id'] + '.npz')
            features = np.load(data / 'proc' / kind / path.name)
            existing_version = ''
            if path.exists():
                with np.load(path) as saved:
                    existing_version = str(saved['supervision_version']) if 'supervision_version' in saved else ''
            if existing_version != 'train-surfaces-v4':
                world, support_indices = cloud(index), np.array([index])
                geometry = surface_targets(world, features['uv'], features['K'], poses[index], features['image_size_hw'])
                geometry['supervision_version'] = np.array('train-surfaces-v4')
                geometry['support_train_indices'] = support_indices
                np.savez(path, **geometry)
            else:
                geometry = dict(np.load(path))
            if kind == 'features_train':
                valid = geometry['geometry_valid']
                world, _ = local_map(index)
                mapping = visible_voxels(world, calibration[index], poses[index], image_mask)
                visible.append(set(mapping))
                coverage.append(mapping)
                statistics.append(dict(frame=row['frame_id'], valid=int(valid.sum()), total=len(valid)))
    counts = np.zeros((len(visible), len(visible)), np.float32)
    for i, first in enumerate(visible):
        for j in range(i):
            intersection = first & visible[j]
            if len(intersection) >= 16:
                first_blocks = set().union(*(coverage[i][voxel] for voxel in intersection))
                second_blocks = set().union(*(coverage[j][voxel] for voxel in intersection))
                if min(len(first_blocks), len(second_blocks)) >= 3:
                    counts[i, j] = counts[j, i] = 2 * len(intersection) / max(1, len(first) + len(visible[j]))
    isolated = np.flatnonzero((counts >= .2).sum(1) == 0)
    counts[isolated, isolated] = 1
    sparse.save_npz(data / 'train/lidar_overlap.npz', sparse.coo_matrix(counts))
    save_json(data / 'proc/geometry_report.json', dict(frames=statistics, isolated_nodes=isolated.tolist(),
        geometry_quality_cap=.25, independent_calibration_verification=False,
        supervision_version='train-surfaces-v4', coordinate_targets='Single training scan, conservative ray/surface support',
        graph_aggregation='Up to 9 training scans within 5m; require each 0.5m voxel in >=2 scans; 0.1m deduplication; query frustum, mask and z-buffer',
        label_status='Weak surface supervision: local planar support and occlusion checks; cross-frame static verification unavailable',
        graph_surface_voxel_m=.5, graph_min_shared_voxels=16, graph_threshold=.2,
        isolated_policy='self-loop only, no fabricated cross-image edge'))
