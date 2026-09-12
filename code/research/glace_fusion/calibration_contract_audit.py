import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scipy.spatial.transform import Rotation


def ssc_pose(values):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', values[3:], degrees=True).as_matrix()
    T[:3, 3] = values[:3]
    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    meta = json.loads((args.bundle / 'data/train_scene/scene_meta.json').read_text())
    K = np.loadtxt(args.official / 'K_cam5.csv', delimiter=',')
    camera = np.loadtxt(args.official / 'x_lb3_c5.csv', delimiter=',')
    extrinsic = ssc_pose(np.array([.035, .002, -1.23, -179.93, -.23, .50])) @ ssc_pose(camera)
    rows = json.loads((args.bundle / 'data/test_rows.json').read_text())
    records = []
    for row in rows:
        path = args.bundle / 'cache/selected/coordinates' / (row['image'] + '.npz')
        with np.load(path) as cached:
            expected = K.copy()
            expected[:2] *= 480 / 1232
            uv = cached['uv']
            records.append(dict(image=row['image'], intrinsics_max_abs_difference=float(np.max(np.abs(cached['K'] - expected))),
                pixel_grid_center_max_abs_difference=float(np.max(np.abs((uv - 4) / 8 - np.rint((uv - 4) / 8)))),
                exact_scan_exists=(args.bundle / 'data/scans' / row['sequence'] / 'velodyne_sync' / (row['image'] + '.bin')).exists()))
    report = dict(official_source='https://robots.engin.umich.edu/nclt/',
        raw_intrinsics_max_abs_difference=float(np.max(np.abs(np.asarray(meta['K_raw']) - K))),
        camera_to_body_max_abs_difference=float(np.max(np.abs(np.asarray(meta['T_BC_camera_to_body']) - extrinsic))),
        cached_intrinsics_max_abs_difference=max(r['intrinsics_max_abs_difference'] for r in records),
        pixel_centers_max_abs_difference=max(r['pixel_grid_center_max_abs_difference'] for r in records),
        exact_scan_count=sum(r['exact_scan_exists'] for r in records), frames=len(records), records=records,
        official_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.official.iterdir() if p.is_file()},
        limitation='Parameter-chain consistency only. No independently annotated physical features, raw undistortion-map verification, motion/velocity correlation or reference covariance available. Does not prove physical alignment or synchronization.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'official_hashes')}, indent=2))


if __name__ == '__main__':
    main()
