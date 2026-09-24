import argparse
import hashlib
import json
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import torch
from safetensors.torch import load_file

from data.local905_mink import Local905_mink, RAW_DTYPE
from data.local905_query import load_sparse_scan
from data.robotcar_sdk.python.velodyne import get_velo
from models.model_mink import LEADER
from utils.pose_util import cartesian_to_polar_expansion


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def check_frame(path, key, voxel_size):
    scan, label = get_velo(str(path))
    raw = np.fromfile(path, dtype=RAW_DTYPE)
    local_scan = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(np.float32) * 0.005 - 100
    local_label = raw['intensity'].astype(np.float32)
    raw_difference = float(np.max(np.abs(scan - local_scan)))
    label_difference = float(np.max(np.abs(label.astype(np.float32) - local_label)))
    ranges = np.linalg.norm(scan, axis=1)
    keep = (ranges > 1.0) & (ranges < 100.0)
    scan, label = scan[keep], label[keep]
    polar = cartesian_to_polar_expansion(scan, voxel_size * 1024)
    features = np.column_stack((polar[:, 2], polar[:, 1], label)).astype(np.float32)
    original_coords, original_features = ME.utils.sparse_quantize(
        coordinates=polar, features=features, quantization_size=voxel_size)
    local_coords, local_features = load_sparse_scan(path, key, 0, voxel_size)
    return {
        'key': key,
        'raw_points': int(len(raw)),
        'filtered_points': int(len(scan)),
        'voxels': int(len(original_coords)),
        'raw_xyz_max_abs_difference': raw_difference,
        'intensity_max_abs_difference': label_difference,
        'voxel_coordinates_equal': bool(np.array_equal(original_coords, local_coords)),
        'voxel_features_equal': bool(np.array_equal(original_features, local_features)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--extra', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--all_diagnostic', action='store_true')
    args = parser.parse_args()
    split = json.loads(args.split.read_text(encoding='utf-8'))
    keys = ([split['splits']['train'][0], split['splits']['val'][0]] +
            (split['splits']['test'] if args.all_diagnostic else
             [split['splits']['test'][0], split['splits']['test'][-1]]))
    frames = [check_frame(args.data_root / key, key, 0.2) for key in keys]
    state = load_file(str(args.weights))
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, magic=False)
    model.load_state_dict(state, strict=True)
    extra = json.loads(args.extra.read_text(encoding='utf-8'))
    center = np.asarray(extra['center_t'], dtype=np.float64)
    dataset = Local905_mink(args.data_root, args.split, 'train', voxel_size=0.2, max_points=0)
    expected_center = dataset.get_center_t() + [0, 0, -200]
    reports = {
        'protocol': 'official_leader_local905_input_audit_v1',
        'weights_sha256': digest(args.weights),
        'extra_sha256': digest(args.extra),
        'split_sha256': digest(args.split),
        'original_training_dates_from_loader': ['2012-01-22', '2012-02-02', '2012-02-18', '2012-05-11'],
        'validation_date_in_pretraining': True,
        'diagnostic_313_date_in_pretraining': True,
        'candidate_confirmation_date_in_pretraining': False,
        'official_checkpoint_center_t': center.tolist(),
        'local_train_center_t': expected_center.tolist(),
        'center_difference_m': float(np.linalg.norm(center - expected_center)),
        'official_model_strict_load': True,
        'actual_max_points': 0,
        'voxel_size': 0.2,
        'frames': frames,
        'audited_frame_count': len(frames),
    }
    reports['passed'] = all(
        row['raw_xyz_max_abs_difference'] == 0
        and row['intensity_max_abs_difference'] == 0
        and row['voxel_coordinates_equal']
        and row['voxel_features_equal'] for row in frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': reports['passed'], 'frames': frames,
                      'center_difference_m': reports['center_difference_m']}), flush=True)
    if not reports['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
