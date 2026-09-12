import argparse
import json
import shutil
from pathlib import Path

import numpy as np

try:
    from .nclt_camera import (TRAIN_DATES, TEST_DATES, camera_rows, validate_dates,
                              stored_intrinsics, NCLTTrajectory, trajectory_path)
except ImportError:
    from nclt_camera import (TRAIN_DATES, TEST_DATES, camera_rows, validate_dates,
                             stored_intrinsics, NCLTTrajectory, trajectory_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_folder', required=True,
                        help='LEADER NCLT dataset folder (root containing NCLT/)')
    parser.add_argument('--camera_root', required=True,
                        help='NCLT_camera_v1 root with all_images.csv and calibration/')
    parser.add_argument('--out', required=True)
    parser.add_argument('--camera_number', type=int, default=5)
    parser.add_argument('--body_to_lb3_ssc_deg', default='0.035,0.002,-1.23,-179.93,-0.23,0.50',
                        help='LB3 pose in body coordinates, SSC degrees (NCLT convention)')
    parser.add_argument('--allow_partial', action='store_true')
    parser.add_argument('--train_dates', nargs='*', default=list(TRAIN_DATES))
    parser.add_argument('--test_dates', nargs='*', default=list(TEST_DATES))
    parser.add_argument('--copy_images', action='store_true',
                        help='copy images instead of symlinking')
    return parser.parse_args()


def calibration_chain(calibration_root, camera_number, body_to_lb3_ssc_deg, image_size_hw):
    """NCLT calibration chain: returns (K_scaled, T_BC camera->body)."""
    from scipy.spatial.transform import Rotation

    def ssc_pose(values):
        values = np.asarray(values, dtype=float).reshape(6)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('xyz', values[3:], degrees=True).as_matrix()
        T[:3, 3] = values[:3]
        return T

    root = Path(calibration_root) / 'calibration' / 'cam_params'
    if not root.is_dir():
        root = Path(calibration_root) / 'cam_params'
    metadata = Path(calibration_root) / 'calibration' / 'processing_metadata.json'
    if not root.is_dir() and metadata.is_file():
        original = json.loads(metadata.read_text())['original_calibration_dir']
        root = Path(original) / 'cam_params'
    K = np.loadtxt(root / ('K_cam%d.csv' % camera_number), delimiter=',')
    T_LB3_C = ssc_pose(np.loadtxt(root / ('x_lb3_c%d.csv' % camera_number), delimiter=','))
    T_B_LB3 = ssc_pose(np.asarray(body_to_lb3_ssc_deg.split(','), dtype=float))
    if image_size_hw is not None:
        K[0] *= image_size_hw[1] / 1616
        K[1] *= image_size_hw[0] / 1232
    return K, T_B_LB3 @ T_LB3_C


def main():
    args = parse_args()
    validate_dates(args.train_dates, args.test_dates)
    rows = camera_rows(args.camera_root, args.camera_number)
    available = {r['sequence'] for r in rows}
    missing = (set(args.train_dates) | set(args.test_dates)) - available
    if missing and not args.allow_partial:
        raise SystemExit('Missing camera sequences: ' + ', '.join(sorted(missing))
                         + '; use --allow_partial only for an explicitly partial experiment')
    K_raw, T_BC = calibration_chain(args.camera_root, args.camera_number,
                                   args.body_to_lb3_ssc_deg, None)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Output scene must be empty to prevent stale split/image leakage')
    out.mkdir(parents=True, exist_ok=True)
    meta = {'T_BC_camera_to_body': T_BC.tolist(), 'K_raw': K_raw.tolist(),
            'intrinsics_convention': 'K matches the stored raster; loader alone resizes it',
            'pose_convention': 'T_WC at original_image_timestamp, linear translation + SLERP',
            'camera_number': args.camera_number, 'missing_sequences': sorted(missing), 'splits': {}}
    for split, dates in (('train', args.train_dates), ('test', args.test_dates)):
        split_dir = out / split
        for name in ('rgb', 'poses', 'calibration'):
            (split_dir / name).mkdir(parents=True, exist_ok=True)
        pairs, skipped = [], 0
        for date in dates:
            selected = [r for r in rows if r['sequence'] == date]
            if not selected:
                continue
            trajectory = NCLTTrajectory(trajectory_path(args.dataset_folder, date))
            for row in selected:
                ts = row['timestamp_us']
                if not trajectory.timestamps[0] <= ts <= trajectory.timestamps[-1]:
                    skipped += 1
                    continue
                src = Path(args.camera_root) / row['saved_path']
                K, size = stored_intrinsics(K_raw, row, src)
                stem = str(ts)
                dst = split_dir / 'rgb' / (stem + src.suffix)
                if dst.exists():
                    raise ValueError(f'Duplicate exposure timestamp: {ts}')
                if args.copy_images:
                    shutil.copyfile(src, dst)
                else:
                    dst.symlink_to(src.resolve())
                T_WC = trajectory.at([ts])[0] @ T_BC
                np.savetxt(split_dir / 'poses' / (stem + '.txt'), T_WC, fmt='%.9f')
                np.savetxt(split_dir / 'calibration' / (stem + '.txt'), K, fmt='%.9f')
                pairs.append({'image': stem, 'sequence': date, 'image_timestamp_us': ts,
                              'group_target_timestamp': int(row['group_target_timestamp']),
                              'stored_size_hw': list(size)})
        meta['splits'][split] = {'dates': dates, 'frames': len(pairs),
                                 'skipped_gt_range': skipped, 'pairs': pairs}
        print(f'{split}: {len(pairs)} frames written; {skipped} outside GT range')
    (out / 'scene_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
