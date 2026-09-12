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


def build_targets(scene_train, meta, dataset_root, out_dir, *, max_points=8192,
                  voxel=0.1, progress_path=None, progress_every=200):
    """Write lidar_world/<stem>.npy (float16 nominal-camera cloud) and
    lidar_world/<stem>_twc.npy (float32 nominal T_WC) for every training image
    with an exact-timestamp scan. Returns (written, missing) stem lists."""
    scene_train = Path(scene_train)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(dataset_root)
    T_BC = np.asarray(meta['T_BC_camera_to_body'], dtype=np.float64)
    rows = {r['image']: r for r in meta['splits']['train']['pairs']}
    stems = sorted(p.stem for p in (scene_train / 'rgb').iterdir())
    written, missing = [], []
    started = time()
    for index, stem in enumerate(stems):
        row = rows[stem]
        scan = dataset_root / row['sequence'] / 'velodyne_sync' / (str(row['image_timestamp_us']) + '.bin')
        if not scan.exists():
            missing.append(stem)
        else:
            gt = np.loadtxt(scene_train / 'poses' / (stem + '.txt'))
            camera, _ = load_scan_world(scan, gt, T_BC)
            camera = voxel_downsample(camera, voxel, max_points)
            if len(camera) < 3:
                missing.append(stem)
            else:
                np.save(out_dir / (stem + '.npy'), camera.astype(np.float16))
                np.save(out_dir / (stem + '_twc.npy'), gt.astype(np.float32))
                written.append(stem)
        if progress_path is not None and (index % progress_every == 0 or index + 1 == len(stems)):
            payload = dict(stage='lidar_targets', completed=index + 1, total=len(stems),
                           written=len(written), missing=len(missing), time=time(), started=started)
            path = Path(str(progress_path) + '.tmp')
            path.write_text(json.dumps(payload, indent=2))
            path.replace(progress_path)
    return written, missing
