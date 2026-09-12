import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    args = parser.parse_args()
    scene = args.run_root / 'scene'
    meta = json.loads((scene / 'scene_meta.json').read_text())
    extrinsic = np.asarray(meta['T_BC_camera_to_body'])
    dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('intensity', 'u1'), ('ring', 'u1')])
    report = dict(scope='Sparse same-timestamp LiDAR depth consistency, not dense 3D ground truth',
        pixel_radius=3, depth_range_m=[2, 80], maximum_neighbor_depth_ratio=1.2, arms={})
    cached = {}
    arms = ['unmasked', 'masked', 'stage1']
    if (args.run_root / 'lidar_auxiliary/eval_test/complete.json').exists():
        arms.append('lidar_auxiliary')
    for arm in arms:
        report['arms'][arm] = {}
        for split in ['train', 'test']:
            folder = args.run_root / arm / ('eval_' + split) if arm != 'stage1' else args.run_root / ('stage1_eval_' + split)
            manifest = json.loads((folder / 'manifest.json').read_text())
            if manifest['inference_contract']['image_resolution'] != 480:
                raise ValueError('This diagnostic expects the 480x630 NCLT protocol')
            rows = {r['image']: r for r in meta['splits'][split]['pairs']}
            records = []
            skipped = []
            for path in sorted((folder / 'coordinates').glob('*.npz')):
                row = rows[path.stem]
                scan = args.dataset_root / row['sequence'] / 'velodyne_sync' / (str(row['image_timestamp_us']) + '.bin')
                if not scan.exists():
                    skipped.append(path.stem)
                    continue
                with np.load(path) as data:
                    xyz, uv, K, gt = [data[key] for key in ['xyz', 'uv', 'K', 'GT']]
                cache_key = split, path.stem
                if cache_key not in cached:
                    raw = np.fromfile(scan, dtype=dtype)
                    body = np.column_stack([raw[key] for key in ['x', 'y', 'z']]).astype(float) * .005 - 100
                    camera = (body - extrinsic[:3, 3]) @ extrinsic[:3, :3]
                    camera = camera[(camera[:, 2] > 2) & (camera[:, 2] < 80)]
                    projection = camera @ K.T
                    pixels = projection[:, :2] / projection[:, 2:]
                    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < 630) & (pixels[:, 1] >= 0) & (pixels[:, 1] < 480)
                    pixels, depths = pixels[inside], camera[inside, 2]
                    if len(pixels) < 3:
                        skipped.append(path.stem)
                        continue
                    raster = np.clip(np.floor(pixels + .5).astype(int), [0, 0], [629, 479])
                    cells = raster[:, 1] * 630 + raster[:, 0]
                    zbuffer = np.full(480 * 630, np.inf)
                    np.minimum.at(zbuffer, cells, depths)
                    visible = depths <= zbuffer[cells] + .1
                    pixels, depths = pixels[visible], depths[visible]
                    distances, indices = cKDTree(pixels).query(uv, k=3, distance_upper_bound=3)
                    covered = np.isfinite(distances).all(1)
                    valid_indices = np.flatnonzero(covered)
                    local_depths = depths[indices[covered]]
                    stable = local_depths.max(1) / local_depths.min(1) <= 1.2
                    valid_indices = valid_indices[stable]
                    target = np.median(local_depths[stable], axis=1)
                    cached[cache_key] = valid_indices, target
                selected, target = cached[cache_key]
                if len(selected) == 0:
                    skipped.append(path.stem)
                    continue
                camera = (xyz[selected] - gt[:3, 3]) @ gt[:3, :3]
                projection = camera @ K.T
                with np.errstate(divide='ignore', invalid='ignore'):
                    error = np.linalg.norm(projection[:, :2] / projection[:, 2:] - uv[selected], axis=1)
                error[(camera[:, 2] <= 0) | ~np.isfinite(error)] = np.inf
                relative = np.abs(camera[:, 2] - target) / target
                good = error < 10
                records.append(dict(image=path.stem, matched_points=len(selected),
                    median_lidar_depth_m=float(np.median(target)),
                    median_predicted_depth_m=float(np.median(camera[:, 2])),
                    median_absolute_relative_depth_error=float(np.median(relative)),
                    fraction_depth_within_25_percent=float(np.mean(relative < .25)),
                    fraction_reprojection_within_10px=float(good.mean()),
                    median_relative_depth_error_among_10px_inliers=float(np.median(relative[good])) if good.any() else None))
            summary = dict(frames=len(records), skipped=skipped, records=records)
            for key in ['median_lidar_depth_m', 'median_predicted_depth_m',
                        'median_absolute_relative_depth_error', 'fraction_depth_within_25_percent',
                        'median_relative_depth_error_among_10px_inliers']:
                values = [r[key] for r in records if r[key] is not None]
                summary[key] = float(np.median(values)) if values else None
            report['arms'][arm][split] = summary
    (args.run_root / 'depth_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({arm: {split: {k: v for k, v in row.items() if k != 'records'}
        for split, row in splits.items()} for arm, splits in report['arms'].items()}, indent=2))


if __name__ == '__main__':
    main()
