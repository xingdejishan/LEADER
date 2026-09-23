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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--sam_manifest', type=Path)
    parser.add_argument('--shuffle_sam_seed', type=int)
    parser.add_argument('--subset', choices=('val', 'test'), default='test')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    settings = checkpoint['settings']
    magic = bool(settings['magic'])
    if magic != (args.sam_manifest is not None):
        raise ValueError('Checkpoint modality and SAM manifest do not match')
    if digest(args.split) != settings['split_sha256']:
        raise ValueError('Split differs from training')
    if magic and digest(args.sam_manifest) != settings['sam_manifest_sha256']:
        raise ValueError('SAM manifest differs from training')
    if args.shuffle_sam_seed is not None and not magic:
        raise ValueError('SAM shuffling requires a multimodal checkpoint')
    query = Local905Query(args.data_root, args.split, args.sam_manifest,
                          max_points=settings['max_points'], subset=args.subset)
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, magic=magic).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    center = torch.tensor(checkpoint['center_t'], dtype=torch.float32, device='cuda')
    matcher = Matcher(inlier_threshold=2.0, d_thre=2, num_iterations=10, ratio=0.15,
                      nms_radius=0.1, max_points=3000, k1=30)
    keys = query.keys
    shift = None
    if args.shuffle_sam_seed is not None:
        shift = 1 + (args.shuffle_sam_seed % (len(keys) - 1))
    predictions = []
    started = time.perf_counter()
    with torch.no_grad():
        for index, key in enumerate(keys):
            frame_start = time.perf_counter()
            feature_key = keys[(index + shift) % len(keys)] if shift else None
            try:
                frame = query.load(key, feature_key=feature_key)
                coordinates = ME.utils.batched_coordinates([frame['coords']]).cuda()
                sparse = ME.SparseTensor(torch.as_tensor(frame['feats'], device='cuda'), coordinates)
                if magic:
                    encoded, stages = model.encoder(sparse, return_stages=True)
                else:
                    encoded = model.encoder(sparse)
                stride = torch.tensor(encoded.tensor_stride, device='cuda', dtype=torch.float32)
                points = polar_voxel_centers(encoded.C, stride, settings['voxel_size'], 1024)
                features = encoded.F
                if magic:
                    identity = torch.eye(4, device='cuda')[None]
                    features = model.magic_fusion(
                        features, points, encoded.C, stride,
                        frame['sam_features'].cuda(), frame['intrinsics'].cuda(),
                        frame['camera_from_lidar'].cuda(), identity,
                        frame['image_bounds'].cuda(), stages=stages,
                        voxel_size=settings['voxel_size'], horizontal=1024,
                        image_valid_mask=frame['image_valid_mask'].cuda()
                        if 'image_valid_mask' in frame else None)
                output = model.decoder(features)
                count = max(min(50, len(output)), int(0.5 * len(output)))
                if count < 3:
                    raise RuntimeError('Insufficient predicted correspondences')
                indices = output[:, 3].topk(count).indices
                source = points[indices].float()
                target = output[indices, :3].float()
                transform = matcher.estimator(source[None], target[None])[0]
                transform = full_pool_refine(transform, source, target)
                transform[:3, 3] += center
                if not torch.isfinite(transform).all():
                    raise FloatingPointError('Nonfinite predicted pose')
                row = {'scan': key, 'status': 'ok', 'T_world_body': transform.cpu().tolist(),
                       'seconds': time.perf_counter() - frame_start}
            except Exception as error:
                row = {'scan': key, 'status': 'failed', 'error': type(error).__name__,
                       'detail': str(error), 'seconds': time.perf_counter() - frame_start}
            predictions.append(row)
            print(f'{index + 1}/{len(keys)} {row["status"]} {key}', flush=True)
    payload = {
        'protocol': 'local905_gt_isolated_online_v1',
        'checkpoint_sha256': digest(args.checkpoint),
        'split_sha256': digest(args.split),
        'sam_manifest_sha256': digest(args.sam_manifest) if magic else None,
        'magic': magic, 'subset': args.subset,
        'shuffle_sam_seed': args.shuffle_sam_seed,
        'shuffle_offset': shift, 'expected_frames': len(keys),
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
