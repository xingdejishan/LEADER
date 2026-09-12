from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def camera_targets(world, uv, K, T_CW, height, width, radius):
    targets = np.zeros((len(uv), 3), dtype=np.float32)
    valid = np.zeros(len(uv), dtype=np.float32)
    camera = world @ T_CW[:3, :3].T + T_CW[:3, 3]
    camera = camera[np.isfinite(camera).all(1) & (camera[:, 2] > 2) & (camera[:, 2] < 80)]
    projection = camera @ K.T
    pixels = projection[:, :2] / projection[:, 2:]
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    pixels, depths = pixels[inside], camera[inside, 2]
    if len(pixels) < 3:
        return targets, valid
    raster = np.clip(np.floor(pixels + .5).astype(int), [0, 0], [width - 1, height - 1])
    cells = raster[:, 1] * width + raster[:, 0]
    zbuffer = np.full(height * width, np.inf)
    np.minimum.at(zbuffer, cells, depths)
    keep = depths <= zbuffer[cells] + .1
    pixels, depths = pixels[keep], depths[keep]
    if len(pixels) < 3:
        return targets, valid
    distances, indices = cKDTree(pixels).query(uv, k=3, distance_upper_bound=radius)
    selected = np.flatnonzero(np.isfinite(distances).all(1))
    neighbors = depths[indices[selected]]
    stable = neighbors.max(1) / neighbors.min(1) <= 1.2
    selected = selected[stable]
    depth = np.median(neighbors[stable], axis=1)
    rays = np.column_stack([uv[selected], np.ones(len(selected))]) @ np.linalg.inv(K).T
    targets[selected] = rays * depth[:, None]
    valid[selected] = 1
    return targets, valid


def load_camera_targets(folder, image_path, uv, K, T_CW, height, width):
    path = Path(folder) / (Path(image_path).stem + '.npy')
    if not path.exists():
        return np.zeros((len(uv), 3), np.float32), np.zeros(len(uv), np.float32)
    return camera_targets(np.load(path), uv, K, T_CW, height, width, 3 * height / 480)


def load_camera_targets_rel(folder, image_path, uv, K, T_CW, height, width):
    """Full-scene storage layout: <stem>.npy holds the voxel-downsampled cloud
    in the NOMINAL camera frame (float16) and <stem>_twc.npy the nominal
    camera->world pose. Reconstruct world coordinates, then project through
    the (possibly augmented) T_CW exactly like load_camera_targets."""
    folder = Path(folder)
    cloud_path = folder / (Path(image_path).stem + '.npy')
    pose_path = folder / (Path(image_path).stem + '_twc.npy')
    if not cloud_path.exists() or not pose_path.exists():
        return np.zeros((len(uv), 3), np.float32), np.zeros(len(uv), np.float32)
    camera_rel = np.load(cloud_path).astype(np.float64)
    T_WC = np.load(pose_path).astype(np.float64)
    world = camera_rel @ T_WC[:3, :3].T + T_WC[:3, 3]
    return camera_targets(world, uv, K, T_CW, height, width, 3 * height / 480)
