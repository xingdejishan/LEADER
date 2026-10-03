import argparse
import json
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch

from data.local905_mink import Local905_mink
from data.magic_data import CalibratedImageDataset
from models.magic_fusion import polar_voxel_centers
from models.model_mink import LEADER


def _load_model(weights_path, magic, device):
    checkpoint = torch.load(weights_path, map_location='cpu')
    state = checkpoint.get('model', checkpoint)
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512,
                   width=1024, magic=magic)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith('magic_fusion.') for key in missing):
        raise ValueError(f'Incompatible L0 checkpoint: missing={missing}, unexpected={unexpected}')
    return model.to(device).eval()


def _frame(dataset, index, device):
    item = dataset[index]
    coordinates, features, _, _, _ = item[:5]
    sam, intrinsics, camera_from_lidar, image_bounds = item[5:9]
    image_valid_mask = item[9] if len(item) > 9 else None
    coords = ME.utils.batched_coordinates([coordinates]).to(device)
    feats = torch.from_numpy(np.asarray(features)).float().to(device)
    sparse = ME.SparseTensor(feats, coordinates=coords)
    batch_size = 1
    identity = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    return {
        'sparse': sparse,
        'sam': sam.unsqueeze(0).to(device),
        'intrinsics': intrinsics.unsqueeze(0).to(device),
        'camera_from_lidar': camera_from_lidar.unsqueeze(0).to(device),
        'image_bounds': image_bounds.unsqueeze(0).to(device),
        'image_valid_mask': image_valid_mask.unsqueeze(0).to(device) if image_valid_mask is not None else None,
        'recovery': identity,
        'batch_size': batch_size,
    }


def _predict(model, frame):
    encoded, stages = model.encoder(frame['sparse'], return_stages=True)
    stride = torch.as_tensor(encoded.tensor_stride, device=encoded.F.device, dtype=encoded.F.dtype)
    points = polar_voxel_centers(encoded.C, stride, 0.2, 1024)
    fused = model.magic_fusion(
        encoded.F, points, encoded.C, stride, frame['sam'], frame['intrinsics'],
        frame['camera_from_lidar'], frame['recovery'], frame['image_bounds'],
        stages=stages, voxel_size=0.2, horizontal=1024,
        image_valid_mask=frame['image_valid_mask'])
    output = model.decoder(fused)
    return encoded, stages, fused, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--sam-manifest', type=Path, required=True)
    parser.add_argument('--l0-checkpoint', type=Path, required=True)
    parser.add_argument('--frame-index', type=int, default=0)
    parser.add_argument('--max-points', type=int, default=1024)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    lidar = Local905_mink(str(args.data_root), str(args.split), 'train',
                          voxel_size=0.2, horizontal_res=1024, max_points=args.max_points)
    dataset = CalibratedImageDataset(lidar, str(args.data_root), str(args.sam_manifest),
                                    lidar.valid_mask_sha256)
    if args.frame_index < 0 or args.frame_index >= len(dataset):
        raise ValueError('Requested frame index is outside the training split')
    frame = _frame(dataset, args.frame_index, device)
    baseline = _load_model(args.l0_checkpoint, False, device)
    fused_model = _load_model(args.l0_checkpoint, True, device)
    with torch.no_grad():
        baseline_encoded = baseline.encoder(frame['sparse'])
        baseline_output = baseline.decoder(baseline_encoded.F)
        encoded, stages, fused, fused_output = _predict(fused_model, frame)
    if not torch.equal(encoded.C, baseline_encoded.C):
        raise AssertionError('Step-zero sparse coordinate keys differ from corrected L0')
    torch.testing.assert_close(encoded.F, baseline_encoded.F, rtol=0, atol=0)
    torch.testing.assert_close(fused, baseline_encoded.F, rtol=0, atol=0)
    torch.testing.assert_close(fused_output, baseline_output, rtol=0, atol=0)
    stride_set = [list(stage.tensor_stride) for stage in stages]
    if stride_set != [[2, 2, 2], [8, 8, 8], [16, 16, 16]]:
        raise AssertionError(f'Unexpected RPGE tap strides: {stride_set}')
    optimizer = torch.optim.SGD(fused_model.parameters(), lr=1e-6)
    optimizer.zero_grad(set_to_none=True)
    _, _, _, first_prediction = _predict(fused_model, frame)
    first_prediction.square().mean().backward()
    projection_grad = float(fused_model.magic_fusion.mma.output_projection.weight.grad.norm())
    if projection_grad <= 0:
        raise AssertionError('Zero-initialized residual projection did not receive a first-step gradient')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _, _, _, second_prediction = _predict(fused_model, frame)
    second_prediction.square().mean().backward()
    gradient_norms = {
        'vra_query': float(fused_model.magic_fusion.attention[0].query.weight.grad.norm()),
        'mma_gate': float(fused_model.magic_fusion.mma.eq4_f1[0].conv.kernel.grad.norm()),
        'rpge': float(next(fused_model.encoder.parameters()).grad.norm()),
        'mmregressor': float(next(fused_model.decoder.parameters()).grad.norm()),
    }
    if any(value <= 0 for value in gradient_norms.values()):
        raise AssertionError(f'Gradient did not reach every intended module: {gradient_norms}')
    print(json.dumps({
        'frame_key': dataset.keys[args.frame_index],
        'split': 'train_only',
        'input_voxels': int(frame['sparse'].F.shape[0]),
        'stage_voxels': [int(stage.F.shape[0]) for stage in stages],
        'stage_strides': stride_set,
        'output_voxels': int(encoded.F.shape[0]),
        'step0_exact_l0_parity': True,
        'first_step_residual_projection_grad': projection_grad,
        'second_step_gradient_norms': gradient_norms,
        'device': str(device),
    }, sort_keys=True))


if __name__ == '__main__':
    main()
