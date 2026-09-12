"""Compact LiDAR world-cloud targets for the FULL-scene supervised GLACE run.

Per training image with an exact-timestamp `velodyne_sync` scan, store a
voxel-downsampled point cloud in the NOMINAL camera frame (float16) plus the
nominal camera->world pose (float32 sidecar). At buffer time the patch
reconstructs world coordinates (`world = cam_rel @ R_wc.T + t_wc`) and
projects them through the AUGMENTED camera pose, exactly like the proven
local arm, while keeping ~2GB instead of ~30GB on disk.

float16 precision in the nominal camera frame (2-80 m depth): <=3 cm at 64 m,
<=5 mm at 4 m - far below the Smooth L1 beta=1 m auxiliary scale and the
per-pixel projection error it induces (<0.1 px) is negligible.
"""
import json
from pathlib import Path
from time import time

import numpy as np

SCAN_DTYPE = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'),
                       ('intensity', 'u1'), ('ring', 'u1')])
MIN_DEPTH_M = 2.0
MAX_DEPTH_M = 80.0


def load_scan_world(scan_path, T_WC, T_BC):
    """Exact-timestamp scan -> camera-frame (nominal) + world points.

    T_WC: nominal camera->world GT pose (poses/<stem>.txt). T_BC: camera->body
    extrinsic (scene_meta T_BC_camera_to_body). Body points follow the LEADER
    NCLT convention (uint16 * 0.005 - 100, metres)."""
    data = np.fromfile(scan_path, dtype=SCAN_DTYPE)
    body = np.column_stack([data[k] for k in ('x', 'y', 'z')]).astype(np.float64) * 0.005 - 100.0
    T_BC = np.asarray(T_BC, dtype=np.float64)
    camera = (body - T_BC[:3, 3]) @ T_BC[:3, :3]  # body -> camera (row vectors)
    inside = np.isfinite(camera).all(1) & (camera[:, 2] > MIN_DEPTH_M) & (camera[:, 2] < MAX_DEPTH_M)
    camera = camera[inside]
    T_WC = np.asarray(T_WC, dtype=np.float64)
    world = camera @ T_WC[:3, :3].T + T_WC[:3, 3]
    return camera, world


def voxel_downsample(points, voxel, max_points):
    """Deterministic voxel-grid downsample: first point per occupied voxel in
    scan order; if still above the cap, uniform stride thinning."""
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    picked = np.sort(first)
    if len(picked) > max_points:
        stride = int(np.ceil(len(picked) / max_points))
        picked = picked[::stride]
    return points[picked]


def aggregate_image_scans(image_ts, scan_timestamps, scan_dir, trajectory, T_BC,
                          T_WC_image, window_s):
    """World points from every velodyne_sync scan within +-window_s of the
    image timestamp. Each scan is transformed with the GT trajectory evaluated
    at ITS OWN timestamp (SLERP + linear, identical to the scene labels), so
    the aggregate inherits the LEADER world frame. window_s=0 keeps only the
    exact-timestamp scan (stage-2 behaviour)."""
    image_ts = int(image_ts)
    window_us = int(round(window_s * 1e6))
    lo = np.searchsorted(scan_timestamps, image_ts - window_us, side='left')
    hi = np.searchsorted(scan_timestamps, image_ts + window_us, side='right')
    world_parts = []
    for scan_ts in scan_timestamps[lo:hi]:
        scan = scan_dir / (str(int(scan_ts)) + '.bin')
        if not scan.exists():
            continue
        T_WC_scan = trajectory.at([scan_ts])[0] @ T_BC
        _, world = load_scan_world(scan, T_WC_scan, T_BC)
        if len(world):
            world_parts.append(world)
    if not world_parts:
        return np.zeros((0, 3)), 0
    world = np.concatenate(world_parts)
    return world, len(world_parts)


def build_targets(scene_train, meta, dataset_root, out_dir, *, max_points=8192,
                  voxel=0.1, scan_window_s=0.0, progress_path=None, progress_every=200):
    """Write lidar_world/<stem>.npy (float16 nominal-camera cloud) and
    lidar_world/<stem>_twc.npy (float32 nominal T_WC) for every training image
    with LiDAR support inside the scan window. Returns (written, missing,
    scan_counts) where missing lists unsupported stems and scan_counts maps
    written stems to the number of aggregated scans."""
    scene_train = Path(scene_train)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(dataset_root)
    T_BC = np.asarray(meta['T_BC_camera_to_body'], dtype=np.float64)
    try:
        from .nclt_camera import NCLTTrajectory, trajectory_path
    except ImportError:
        from nclt_camera import NCLTTrajectory, trajectory_path
    rows = {r['image']: r for r in meta['splits']['train']['pairs']}
    stems = sorted(p.stem for p in (scene_train / 'rgb').iterdir())
    trajectories, scan_lists = {}, {}
    for sequence in sorted({rows[s]['sequence'] for s in stems}):
        trajectories[sequence] = NCLTTrajectory(trajectory_path(dataset_root.parent, sequence))
        scan_dir = dataset_root / sequence / 'velodyne_sync'
        scan_lists[sequence] = np.array(sorted(int(p.stem) for p in scan_dir.glob('*.bin')),
                                       dtype=np.int64)
    written, missing, scan_counts = [], [], {}
    started = time()
    for index, stem in enumerate(stems):
        row = rows[stem]
        image_ts = int(row['image_timestamp_us'])
        gt = np.loadtxt(scene_train / 'poses' / (stem + '.txt'))
        world, n_scans = aggregate_image_scans(
            image_ts, scan_lists[row['sequence']],
            dataset_root / row['sequence'] / 'velodyne_sync',
            trajectories[row['sequence']], T_BC, gt, scan_window_s)
        if n_scans == 0:
            missing.append(stem)
        else:
            # world -> nominal (image) camera frame: (P - t) @ R_wc
            world_to_image_cam = (world - gt[:3, 3]) @ gt[:3, :3]
            camera = voxel_downsample(world_to_image_cam, voxel, max_points)
            if len(camera) < 3:
                missing.append(stem)
            else:
                np.save(out_dir / (stem + '.npy'), camera.astype(np.float16))
                np.save(out_dir / (stem + '_twc.npy'), gt.astype(np.float32))
                written.append(stem)
                scan_counts[stem] = n_scans
        if progress_path is not None and (index % progress_every == 0 or index + 1 == len(stems)):
            payload = dict(stage='lidar_targets', completed=index + 1, total=len(stems),
                           written=len(written), missing=len(missing), time=time(), started=started)
            path = Path(str(progress_path) + '.tmp')
            path.write_text(json.dumps(payload, indent=2))
            path.replace(progress_path)
    return written, missing, scan_counts
