import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch

from data.local905_query import Local905Query
from models.magic_fusion import polar_voxel_centers
from models.model_mink import LEADER
from models.sc2pcr import Matcher
from utils.full_pool_robust_v1 import full_pool_refine


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def shuffled_feature_keys(keys, manifest, seed, minimum_gap_us=3_000_000):
    groups = {}
    for key in keys:
        extrinsic = np.asarray(manifest['frames'][key]['T_camera_lidar'], dtype=np.float32)
        camera = hashlib.sha256(extrinsic.astype(np.float64).tobytes()).hexdigest()[:16]
        groups.setdefault(camera, []).append(key)
    random = np.random.default_rng(seed)
    result = {}
    for camera, group in groups.items():
        group.sort(key=lambda key: int(Path(key).stem))
        stamps = np.asarray([int(Path(key).stem) for key in group], dtype=np.int64)
        valid_offsets = [offset for offset in range(1, len(group))
                         if np.all(np.abs(stamps - np.roll(stamps, -offset)) >= minimum_gap_us)]
        if not valid_offsets:
            raise ValueError(f'No same-camera non-neighbor shuffle for {camera}')
        offset = int(random.choice(valid_offsets))
        result.update({key: group[(index + offset) % len(group)]
                       for index, key in enumerate(group)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--sam_manifest', type=Path)
    parser.add_argument('--query_split', type=Path)
    parser.add_argument('--query_manifest', type=Path)
    parser.add_argument('--null_template', type=Path)
    parser.add_argument('--shuffle_sam_seed', type=int)
    parser.add_argument('--save_correspondences', action='store_true')
    parser.add_argument('--subset', choices=('val', 'test'), default='test')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.query_manifest and not args.query_split:
        raise ValueError('External SAM manifest requires an external query split')
    if args.query_split and args.subset != 'test':
        raise ValueError('External query split is only for locked test data')
    if args.save_correspondences:
        (args.out / 'correspondences').mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    settings = checkpoint['settings']
    magic = bool(settings['magic'])
    if magic != (args.sam_manifest is not None):
        raise ValueError('Checkpoint modality and SAM manifest do not match')
    if digest(args.split) != settings['split_sha256']:
        raise ValueError('Split differs from training')
    if magic and digest(args.sam_manifest) != settings['sam_manifest_sha256']:
        raise ValueError('SAM manifest differs from training')
    if bool(args.null_template) != bool(settings.get('null_template_sha256')):
        raise ValueError('Null template mode differs from training')
    if args.null_template and digest(args.null_template) != settings['null_template_sha256']:
        raise ValueError('Null template differs from training')
    if args.shuffle_sam_seed is not None and not magic:
        raise ValueError('SAM shuffling requires a multimodal checkpoint')
    query_split = args.query_split or args.split
    query_manifest = args.query_manifest or args.sam_manifest
    query = Local905Query(args.data_root, query_split, query_manifest,
                          max_points=settings['max_points'], subset=args.subset)
    null_features = None
    if args.null_template:
        with np.load(args.null_template, allow_pickle=False) as archive:
            null_features = {name: torch.from_numpy(archive[name].copy())[None]
                             for name in archive.files}
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, magic=magic,
                   fusion_variant=settings.get('fusion_variant', 'box')).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    center = torch.tensor(checkpoint['center_t'], dtype=torch.float32, device='cuda')
    matcher = Matcher(inlier_threshold=2.0, d_thre=2, num_iterations=10, ratio=0.15,
                      nms_radius=0.1, max_points=3000, k1=30)
    keys = query.keys
    shuffled = None
    if args.shuffle_sam_seed is not None:
        shuffled = shuffled_feature_keys(keys, query.manifest, args.shuffle_sam_seed)
    predictions = []
    started = time.perf_counter()
    with torch.no_grad():
        for index, key in enumerate(keys):
            frame_start = time.perf_counter()
            feature_key = shuffled[key] if shuffled else None
            cache = None
            try:
                loading_start = time.perf_counter()
                frame = query.load(key, feature_key=feature_key)
                feature_load_seconds = time.perf_counter() - loading_start
                if null_features is not None:
                    extrinsic = frame['camera_from_lidar'][0].double().numpy()
                    camera = hashlib.sha256(extrinsic.tobytes()).hexdigest()[:16]
                    frame['sam_features'] = null_features[camera]
                coordinates = ME.utils.batched_coordinates([frame['coords']]).cuda()
                sparse = ME.SparseTensor(torch.as_tensor(frame['feats'], device='cuda'), coordinates)
                torch.cuda.synchronize()
                encoder_start = time.perf_counter()
                if magic:
                    encoded, stages = model.encoder(sparse, return_stages=True)
                else:
                    encoded = model.encoder(sparse)
                torch.cuda.synchronize()
                encoder_seconds = time.perf_counter() - encoder_start
                stride = torch.tensor(encoded.tensor_stride, device='cuda', dtype=torch.float32)
                points = polar_voxel_centers(encoded.C, stride, settings['voxel_size'], 1024)
                features = encoded.F
                visual_valid = torch.zeros(len(encoded.F), dtype=torch.bool, device='cuda')
                fusion_seconds = 0.0
                if magic:
                    identity = torch.eye(4, device='cuda')[None]
                    fusion_start = time.perf_counter()
                    surface_args = ({'raw_points': frame['points']}
                                    if getattr(model.magic_fusion, 'requires_raw_points', False) else {})
                    fused = model.magic_fusion(
                        features, points, encoded.C, stride,
                        frame['sam_features'].cuda(), frame['intrinsics'].cuda(),
                        frame['camera_from_lidar'].cuda(), identity,
                        frame['image_bounds'].cuda(), stages=stages,
                        voxel_size=settings['voxel_size'], horizontal=1024,
                        image_valid_mask=frame['image_valid_mask'].cuda()
                        if 'image_valid_mask' in frame else None,
                        return_validity=args.save_correspondences, **surface_args)
                    if args.save_correspondences:
                        features, visual_valid = fused
                    else:
                        features = fused
                    torch.cuda.synchronize()
                    fusion_seconds = time.perf_counter() - fusion_start
                output = model.decoder(features)
                count = max(min(50, len(output)), int(0.5 * len(output)))
                if count < 3:
                    raise RuntimeError('Insufficient predicted correspondences')
                indices = output[:, 3].topk(count).indices
                source = points[indices].float()
                target = output[indices, :3].float()
                solver_start = time.perf_counter()
                initial = matcher.estimator(source[None], target[None])[0]
                initial_world = initial.clone()
                transform = full_pool_refine(initial, source, target)
                torch.cuda.synchronize()
                solver_seconds = time.perf_counter() - solver_start
                initial_world[:3, 3] += center
                transform[:3, 3] += center
                if not torch.isfinite(transform).all():
                    raise FloatingPointError('Nonfinite predicted pose')
                if args.save_correspondences:
                    cache = {
                        'voxel_coordinates': encoded.C.cpu().numpy().astype(np.int32),
                        'input_local_xyz': points.cpu().numpy().astype(np.float32),
                        'predicted_centered_xyz': output[:, :3].cpu().numpy().astype(np.float32),
                        'predicted_world_xyz': (output[:, :3] + center).cpu().numpy().astype(np.float32),
                        'predicted_reliability': output[:, 3].cpu().numpy().astype(np.float32),
                        'selected_indices': indices.cpu().numpy().astype(np.int32),
                        'visual_candidate_valid': visual_valid.cpu().numpy().astype(np.bool_),
                        'T_initial_world_body': initial_world.cpu().numpy().astype(np.float64),
                        'T_refined_world_body': transform.cpu().numpy().astype(np.float64),
                    }
                row = {'scan': key, 'status': 'ok', 'T_world_body': transform.cpu().tolist(),
                       'seconds': time.perf_counter() - frame_start,
                       'sam_cache_load_seconds': feature_load_seconds if magic else None,
                       'encoder_seconds': encoder_seconds,
                       'fusion_seconds': fusion_seconds,
                       'solver_seconds': solver_seconds,
                       'correspondences_file': f'correspondences/{index:04d}.npz'
                       if args.save_correspondences else None}
            except Exception as error:
                row = {'scan': key, 'status': 'failed', 'error': type(error).__name__,
                       'detail': str(error), 'seconds': time.perf_counter() - frame_start}
            if args.save_correspondences:
                if cache is None:
                    cache = {}
                cache['scan'] = np.asarray(key)
                cache['status'] = np.asarray(row['status'])
                cache_path = args.out / 'correspondences' / f'{index:04d}.npz'
                np.savez_compressed(cache_path, **cache)
                row['correspondences_sha256'] = digest(cache_path)
            predictions.append(row)
            print(f'{index + 1}/{len(keys)} {row["status"]} {key}', flush=True)
    payload = {
        'protocol': 'local905_gt_isolated_online_v1',
        'checkpoint_sha256': digest(args.checkpoint),
        'split_sha256': digest(query_split),
        'training_split_sha256': digest(args.split),
        'sam_manifest_sha256': digest(query_manifest) if magic else None,
        'training_sam_manifest_sha256': digest(args.sam_manifest) if magic else None,
        'null_template_sha256': digest(args.null_template) if args.null_template else None,
        'magic': magic, 'subset': args.subset,
        'shuffle_sam_seed': args.shuffle_sam_seed,
        'shuffle_mapping_sha256': hashlib.sha256(json.dumps(shuffled, sort_keys=True).encode()).hexdigest()
        if shuffled else None, 'shuffle_minimum_gap_us': 3_000_000 if shuffled else None,
        'expected_frames': len(keys),
        'correspondence_cache': args.save_correspondences,
        'elapsed_seconds': time.perf_counter() - started,
        'predictions': predictions,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'predictions.json'
    if path.exists():
        raise FileExistsError(path)
    temporary = args.out / 'predictions.tmp'
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)
    (args.out / 'predictions.sha256').write_text(digest(path) + '\n', encoding='ascii')


if __name__ == '__main__':
    main()
