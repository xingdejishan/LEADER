"""Build a GLACE scene layout in the LEADER world frame (design point 6).

GLACE training data must live in the SAME world W as LEADER's scene
coordinates. For every synchronized (LiDAR scan, camera image) pair:

    T_WC_GT = T_WB_GT @ T_BC

is written to poses/<ts>.txt (GLACE/DSAC* camera->world convention), the
scaled pinhole K to calibration/<ts>.txt, and the image to rgb/<ts>.jpg.
Output layout matches the vendor CamLocDataset:

    <out>/
        train/ rgb poses calibration [features.npy]
        test/  rgb poses calibration [features.npy]
        scene_meta.json
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_folder', required=True,
                        help='LEADER NCLT dataset folder (root containing NCLT/)')
    parser.add_argument('--camera_root', required=True,
                        help='NCLT_camera_v1 root with all_images.csv and calibration/')
    parser.add_argument('--out', required=True)
    parser.add_argument('--camera_number', type=int, default=5)
    parser.add_argument('--body_to_lb3_ssc_deg', default='0.035,0.002,-1.23,-179.93,-0.23,0.50',
                        help='body->lb3 SSC extrinsic (NCLT calibration chain)')
    parser.add_argument('--image_size', type=int, nargs=2, default=(616, 808),
                        help='(H, W) of the stored images; K is scaled to match')
    parser.add_argument('--max_sync_delta_s', type=float, default=0.05)
    parser.add_argument('--train_dates', nargs='*', default=['2012-01-22', '2012-02-12'])
    parser.add_argument('--test_dates', nargs='*', default=['2012-02-18', '2012-03-31'])
    parser.add_argument('--copy_images', action='store_true',
                        help='copy images instead of symlinking')
    return parser.parse_args()


def calibration_chain(calibration_root, camera_number, body_to_lb3_ssc_deg, image_size_hw):
    """NCLT calibration chain: returns (K_scaled, T_BC camera->body)."""
    import sys
    from scipy.spatial.transform import Rotation

    def ssc_pose(values):
        values = np.asarray(values, dtype=float).reshape(6)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('xyz', values[3:], degrees=True).as_matrix()
        T[:3, 3] = values[:3]
        return T

    root = Path(calibration_root) / 'cam_params'
    K = np.loadtxt(root / ('K_cam%d.csv' % camera_number), delimiter=',')
    T_LB3_C = ssc_pose(np.loadtxt(root / ('x_lb3_c%d.csv' % camera_number), delimiter=','))
    T_B_LB3 = ssc_pose(np.asarray(body_to_lb3_ssc_deg.split(','), dtype=float))
    K[0] *= image_size_hw[1] / 1616  # original width
    K[1] *= image_size_hw[0] / 1232  # original height
    return K, np.linalg.inv(T_B_LB3 @ T_LB3_C)


def leader_gt_poses(dataset_folder, train):
    """Per-scan GT T_WB from the exact LEADER NCLT dataloader (train split only
    selects sequences; scan images are not loaded)."""
    sys_path = None
    try:
        import MinkowskiEngine  # noqa: F401  (datagenerator import side effect)
    except ImportError:
        raise SystemExit('Run inside the LEADER environment (MinkowskiEngine required)')
    from data.NCLTVelodyne_datagenerator_mink import NCLT_mink

    dataset = NCLT_mink(data_path=dataset_folder, train=train)
    poses = dataset.poses
    rots = dataset.rots
    by_ts = {}
    for i, path in enumerate(dataset.pcs):
        ts = int(Path(path).stem)
        T = np.eye(4)
        T[:3, :3] = rots[i]
        T[:3, 3] = poses[i, :3]
        by_ts[ts] = T
    return by_ts


def main():
    args = parse_args()
    image_size_hw = tuple(args.image_size)
    K0, T_BC = calibration_chain(args.camera_root, args.camera_number,
                                 args.body_to_lb3_ssc_deg, image_size_hw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {'T_BC_camera_to_body': T_BC.tolist(), 'K_full_resolution': K0.tolist(),
            'image_size_hw': list(image_size_hw), 'camera_number': args.camera_number,
            'max_sync_delta_s': args.max_sync_delta_s, 'splits': {}}

    import csv
    rows = []
    with open(Path(args.camera_root) / 'all_images.csv') as handle:
        for row in csv.DictReader(handle):
            if row['camera'] != 'Cam%d' % args.camera_number:
                continue
            rows.append(row)
    rows.sort(key=lambda r: int(r['group_target_timestamp']))

    for split, dates in (('train', args.train_dates), ('test', args.test_dates)):
        if not dates:
            continue
        scan_gt = leader_gt_poses(args.dataset_folder, train=(split == 'train'))
        split_dir = out / split
        for name in ('rgb', 'poses', 'calibration'):
            (split_dir / name).mkdir(parents=True, exist_ok=True)
        scan_ts_sorted = np.array(sorted(scan_gt))
        kept, skipped = 0, 0
        deltas = []
        for row in rows:
            if row['sequence'] not in dates:
                continue
            image_ts = int(row['group_target_timestamp'])
            j = int(np.argmin(np.abs(scan_ts_sorted - image_ts)))
            delta_s = (image_ts - int(scan_ts_sorted[j])) / 1e6
            if abs(delta_s) > args.max_sync_delta_s:
                skipped += 1
                continue
            T_WB = scan_gt[int(scan_ts_sorted[j])]
            T_WC = T_WB @ T_BC
            stem = str(image_ts)
            src = Path(args.camera_root) / row['saved_path']
            dst = split_dir / 'rgb' / (stem + '.jpg')
            if not dst.exists():
                if args.copy_images:
                    shutil.copyfile(src, dst)
                else:
                    try:
                        dst.symlink_to(src.resolve())
                    except OSError:
                        shutil.copyfile(src, dst)
            np.savetxt(split_dir / 'poses' / (stem + '.txt'), T_WC, fmt='%.9f')
            np.savetxt(split_dir / 'calibration' / (stem + '.txt'), K0, fmt='%.9f')
            deltas.append({'image': stem, 'scan_ts': int(scan_ts_sorted[j]),
                           'delta_s': delta_s})
            kept += 1
        meta['splits'][split] = {'dates': dates, 'frames': kept, 'skipped_sync': skipped,
                                 'pairs': deltas}
        print(f'{split}: {kept} frames written, {skipped} skipped by sync gate')

    (out / 'scene_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('scene written to', out)


if __name__ == '__main__':
    main()
