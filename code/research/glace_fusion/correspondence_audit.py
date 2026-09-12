import argparse
import json
from pathlib import Path

import numpy as np

from .lidar_supervision import camera_targets


def metrics(xyz, uv, K, gt, E, scan):
    camera = (xyz - gt[:3, 3]) @ gt[:3, :3]
    projection = camera @ K.T
    with np.errstate(divide='ignore', invalid='ignore'):
        errors = np.linalg.norm(projection[:, :2] / projection[:, 2:] - uv, axis=1)
    errors[(camera[:, 2] <= 0) | ~np.isfinite(errors)] = np.inf
    result = {f'q{threshold}': float(np.mean(errors < threshold)) for threshold in [1, 3, 5, 10, 20]}
    result.update(count=len(xyz), median_reprojection_px=float(np.median(errors)))
    grid = np.minimum((uv / [630, 480] * 4).astype(int), 3)
    cells = grid[:, 1] * 4 + grid[:, 0]
    counts = np.bincount(cells[errors < 10], minlength=16)
    result['inlier_cells_at_least_5'] = int((counts >= 5).sum())
    result['inlier_cells_at_least_20'] = int((counts >= 20).sum())
    result['inliers_10px'] = int((errors < 10).sum())
    if scan.exists():
        dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('intensity', 'u1'), ('ring', 'u1')])
        raw = np.fromfile(scan, dtype=dtype)
        body = np.column_stack([raw[k] for k in ['x', 'y', 'z']]).astype(float) * .005 - 100
        T = gt @ np.linalg.inv(E)
        world = body @ T[:3, :3].T + T[:3, 3]
        target, support = camera_targets(world, uv, K, np.linalg.inv(gt), 480, 630, 3)
        supported = support > 0
        err3d = np.linalg.norm(camera - target, axis=1)
        relative = np.abs(camera[:, 2] - target[:, 2]) / np.maximum(target[:, 2], 1e-8)
        result['depth_support'] = int(supported.sum())
        if supported.any():
            result['median_3d_error_m'] = float(np.median(err3d[supported]))
            result['median_relative_depth_error'] = float(np.median(relative[supported]))
            result['depth_within_25_percent'] = float(np.mean(relative[supported] < .25))
            result['precision_3d_05m'] = float(np.mean(err3d[supported] < .5))
            result['precision_3d_1m'] = float(np.mean(err3d[supported] < 1))
        good = supported & (errors < 10)
        if good.any():
            result['inlier_median_3d_error_m'] = float(np.median(err3d[good]))
            result['inlier_median_relative_depth_error'] = float(np.median(relative[good]))
    return result


def summarize(records):
    groups = dict(all=lambda r: True, supported=lambda r: r['in_orientation_support'],
        outside=lambda r: not r['in_orientation_support'],
        new_supported=lambda r: r['in_orientation_support'] and not r['previous_probe'])
    report = {}
    for name, keep in groups.items():
        rows = [r for r in records if keep(r)]
        report[name] = dict(frames=len(rows), metrics={})
        for key in sorted(set().union(*(r['metrics'].keys() for r in rows))):
            values = np.array([r['metrics'][key] for r in rows if key in r['metrics']])
            report[name]['metrics'][key] = dict(mean=float(values.mean()),
                median=float(np.median(values)), p10=float(np.quantile(values, .1)),
                p90=float(np.quantile(values, .9)))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinates', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    base = Path('/root/rivermind-data')
    rows = json.loads((base / 'glace_stage3_region_full_20260912/rows.json').read_text())
    meta = json.loads((base / 'glace_nclt_rgb_large_20260912/scene/scene_meta.json').read_text())
    E = np.asarray(meta['T_BC_camera_to_body'])
    records = []
    for row in rows:
        data = np.load(args.coordinates / (row['image'] + '.npz'))
        scan = base / 'datasets/NCLT' / row['sequence'] / 'velodyne_sync' / (row['image'] + '.bin')
        record = dict(row)
        record['metrics'] = metrics(data['xyz'], data['uv'], data['K'], data['GT'], E, scan)
        records.append(record)
    args.out.write_text(json.dumps(dict(records=records, summary=summarize(records),
        scope='Same timestamp sparse LiDAR consistency; no dense 3D ground truth; GT used only in audit'), indent=2))


if __name__ == '__main__':
    main()
