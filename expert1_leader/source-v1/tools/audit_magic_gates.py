import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import MinkowskiEngine as ME
import numpy as np
import torch

from models.magic_fusion import polar_voxel_centers
from models.model_mink import LEADER
from models.sc2pcr import Matcher
from run_mink import TRR, get_data_loader
from tools.train_local905 import batch_loss, digest
from utils.full_pool_robust_v1 import full_pool_refine


def frozen_bn(module):
    for item in module.modules():
        if isinstance(item, torch.nn.modules.batchnorm._BatchNorm):
            item.eval()


def pose(points, prediction, center):
    count = max(min(50, len(prediction)), int(0.5 * len(prediction)))
    indices = prediction[:, 3].topk(count).indices
    matcher = Matcher(inlier_threshold=2.0, d_thre=2, num_iterations=10,
                      ratio=0.15, nms_radius=0.1, max_points=3000, k1=30)
    torch.manual_seed(17)
    transform = matcher.estimator(points[indices][None].float(),
                                  prediction[indices, :3][None].float())[0]
    transform = full_pool_refine(transform, points[indices].float(),
                                 prediction[indices, :3].float())
    transform[:3, 3] += center
    return transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--sam_manifest', type=Path, required=True)
    parser.add_argument('--base_checkpoint', type=Path, required=True)
    parser.add_argument('--fusion_init', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    base = torch.load(args.base_checkpoint, map_location='cpu')
    init = torch.load(args.fusion_init, map_location='cpu')
    if base['settings']['max_points'] != 0:
        raise ValueError('Gate requires full original point input')
    if base['settings']['split_sha256'] != digest(args.split):
        raise ValueError('Split hash mismatch')
    if init['base_checkpoint_sha256'] != digest(args.base_checkpoint):
        raise ValueError('Fusion initialization is from another base')
    flags = SimpleNamespace(dataset='Local905', dataset_folder=str(args.data_root),
                            local905_split=str(args.split), local905_max_points=0,
                            voxel_size=0.2, horizontal_res=1024, batch_size=1,
                            val_batch_size=1, num_workers=0, mode='train',
                            magic_manifest=str(args.sam_manifest))
    _, val_loader = get_data_loader(flags)
    batch = next(iter(val_loader))
    center = torch.tensor(base['center_t'], dtype=torch.float32, device='cuda')
    original = LEADER(in_channels=3, out_channels=4, feat_channels=512, magic=False).cuda().eval()
    original.load_state_dict(base['model'])
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, magic=True).cuda().eval()
    model.load_state_dict(base['model'], strict=False)
    model.magic_fusion.load_state_dict(init['fusion'])
    with torch.no_grad():
        sparse_original = ME.SparseTensor(batch['feats'].cuda(), batch['coords'].cuda())
        sparse_fused = ME.SparseTensor(batch['feats'].cuda(), batch['coords'].cuda())
        encoded_original = original.encoder(sparse_original)
        original_coordinates = encoded_original.C.clone()
        original_features = encoded_original.F.clone()
        encoded_same_sparse = original.encoder(sparse_original)
        sparse_repeat = ME.SparseTensor(batch['feats'].cuda(), batch['coords'].cuda())
        encoded_repeat = original.encoder(sparse_repeat)
        encoded_fused, stages = model.encoder(sparse_fused, return_stages=True)
        stride = torch.tensor(encoded_original.tensor_stride, dtype=torch.float32, device='cuda')
        points = polar_voxel_centers(original_coordinates, stride, 0.2, 1024)
        fused_points = polar_voxel_centers(encoded_fused.C, stride, 0.2, 1024)
        original_keys = [tuple(row) for row in original_coordinates.cpu().tolist()]
        fused_lookup = {tuple(row): index for index, row in enumerate(encoded_fused.C.cpu().tolist())}
        voxel_equal = len(original_keys) == len(fused_lookup) and set(original_keys) == set(fused_lookup)
        if not voxel_equal:
            raise ValueError('Baseline and fusion models produced different voxel identities')
        reorder = torch.tensor([fused_lookup[key] for key in original_keys], device='cuda')
        repeat_lookup = {tuple(row): index for index, row in enumerate(encoded_repeat.C.cpu().tolist())}
        repeat_reorder = torch.tensor([repeat_lookup[key] for key in original_keys], device='cuda')
        same_sparse_lookup = {tuple(row): index for index, row in enumerate(encoded_same_sparse.C.cpu().tolist())}
        same_sparse_reorder = torch.tensor([same_sparse_lookup[key] for key in original_keys], device='cuda')
        original_output = original.decoder(original_features)
        fused_features = model.magic_fusion(
            encoded_fused.F, fused_points, encoded_fused.C, stride,
            batch['sam_features'].cuda(), batch['intrinsics'].cuda(),
            batch['camera_from_lidar'].cuda(), torch.eye(4, device='cuda')[None],
            batch['image_bounds'].cuda(), stages=stages, voxel_size=0.2,
            horizontal=1024, image_valid_mask=batch['image_valid_mask'].cuda())
        fused_output = model.decoder(fused_features)[reorder]
        original_pose = pose(points, original_output, center)
        fused_pose = pose(points, fused_output, center)
    identity = {
        'voxel_coordinates_equal': voxel_equal,
        'feature_max_abs_difference': float((original_features - encoded_fused.F[reorder]).abs().max()),
        'same_model_repeat_feature_max_abs_difference': float((original_features - encoded_repeat.F[repeat_reorder]).abs().max()),
        'same_sparse_repeat_feature_max_abs_difference': float((original_features - encoded_same_sparse.F[same_sparse_reorder]).abs().max()),
        'original_feature_mutation_after_repeat': float((original_features - encoded_original.F).abs().max()),
        'world_coordinates_max_abs_difference': float((original_output[:, :3] - fused_output[:, :3]).abs().max()),
        'reliability_max_abs_difference': float((original_output[:, 3] - fused_output[:, 3]).abs().max()),
        'final_pose_max_abs_difference': float((original_pose - fused_pose).abs().max()),
        'finite': bool(torch.isfinite(fused_output).all() and torch.isfinite(fused_pose).all()),
    }
    del original, encoded_original, encoded_fused, stages, fused_features
    model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(False)
    frozen_reference = {name: value.detach().cpu().clone() for name, value in
                        model.state_dict().items() if not name.startswith('magic_fusion.')}
    optimizer = torch.optim.Adam(model.magic_fusion.parameters(), lr=1e-3)
    gradient_norms = {}
    for step in range(3):
        model.train()
        model.encoder.eval()
        model.decoder.eval()
        frozen_bn(model)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = batch_loss(model, batch, center, TRR(scale=10), True, 0.2, 1024)
        loss.backward()
        if step >= 1:
            for name in ('query', 'key', 'value'):
                gradient_norms[name] = float(sum(
                    getattr(attention, name).weight.grad.norm().item()
                    for attention in model.magic_fusion.attention
                    if getattr(attention, name).weight.grad is not None))
        optimizer.step()
    current = model.state_dict()
    frozen_equal = all(torch.equal(current[name].cpu(), value)
                       for name, value in frozen_reference.items())
    fusion_changed = any(not torch.equal(current['magic_fusion.' + name].cpu(), value)
                         for name, value in init['fusion'].items())
    gates = {
        'protocol': 'official_full_points_magic_pretraining_gates_v1',
        'base_checkpoint_sha256': digest(args.base_checkpoint),
        'fusion_init_sha256': digest(args.fusion_init),
        'sam_manifest_sha256': digest(args.sam_manifest),
        'split_sha256': digest(args.split),
        'sample_raw_points': int(len(val_loader.dataset.lidar_dataset[0][2])),
        'max_points': 0,
        'identity': identity,
        'frozen_leader_parameters_and_buffers_unchanged_after_three_steps': frozen_equal,
        'fusion_parameters_changed_after_three_steps': fusion_changed,
        'attention_gradient_norms_after_second_step': gradient_norms,
        'peak_allocated_mb': float(torch.cuda.max_memory_allocated() / 1024 ** 2),
    }
    gates['passed'] = (identity['voxel_coordinates_equal'] and identity['finite']
                       and identity['feature_max_abs_difference'] <= 1e-5
                       and identity['world_coordinates_max_abs_difference'] <= 1e-5
                       and identity['reliability_max_abs_difference'] <= 1e-5
                       and identity['final_pose_max_abs_difference'] <= 1e-4
                       and frozen_equal and fusion_changed
                       and all(gradient_norms.get(name, 0) > 0
                               for name in ('query', 'key', 'value')))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gates, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(gates), flush=True)
    if not gates['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
