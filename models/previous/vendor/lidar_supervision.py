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
