from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def surface_targets(world, uv, K, T_WC, image_size_hw, radius_px=6.0):
    n = len(uv)
    result = {
        'xyz_target_world': np.zeros((n, 3), np.float32),
        'geometry_valid': np.zeros(n, bool),
        'geometry_quality': np.zeros(n, np.float32),
        'sigma_parallel_m': np.ones(n, np.float32),
        'sigma_perpendicular_m': np.ones(n, np.float32),
        'surface_id': np.zeros((n, 3), np.int64),
    }
    if world is None or len(world) < 12:
        return result
    camera = (world - T_WC[:3, 3]) @ T_WC[:3, :3]
    camera = camera[np.isfinite(camera).all(1) & (camera[:, 2] > 2) & (camera[:, 2] < 80)]
    h, w = image_size_hw
    pixels_h = camera @ K.T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:]
    inside = (pixels >= 0).all(1) & (pixels[:, 0] < w) & (pixels[:, 1] < h)
    camera, pixels = camera[inside], pixels[inside]
    if len(camera) < 12:
        return result
    cells = np.floor(pixels).astype(int)
    cell = cells[:, 1] * w + cells[:, 0]
    zbuffer = np.full(h * w, np.inf)
    np.minimum.at(zbuffer, cell, camera[:, 2])
    visible = camera[:, 2] <= zbuffer[cell] + .1
    camera, pixels = camera[visible], pixels[visible]
    if len(camera) < 12:
        return result
    distance, index = cKDTree(pixels).query(uv, k=12, distance_upper_bound=radius_px)
    rows = np.flatnonzero(np.isfinite(distance).all(1))
    if not len(rows):
        return result
    points = camera[index[rows]]
    centers = points.mean(1)
    centered = points - centers[:, None]
    cov = np.einsum('nki,nkj->nij', centered, centered) / points.shape[1]
    values, vectors = np.linalg.eigh(cov)
    normal = vectors[:, :, 0]
    rays = np.c_[uv[rows], np.ones(len(rows))] @ np.linalg.inv(K).T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    denominator = (normal * rays).sum(1)
    ranges = (normal * centers).sum(1) / np.where(np.abs(denominator) > 1e-8, denominator, 1e-8)
    target = rays * ranges[:, None]
    tangent = np.einsum('nki,nij->nkj', centered, vectors[:, :, 1:])
    intersection = np.einsum('ni,nij->nj', target - centers, vectors[:, :, 1:])
    support = (intersection >= tangent.min(1)).all(1) & (intersection <= tangent.max(1)).all(1)
    rms = np.sqrt(np.maximum(values[:, 0], 0))
    spread = points[:, :, 2].max(1) - points[:, :, 2].min(1)
    valid = support & (rms < .08) & (values[:, 1] > .0004)
    valid &= values[:, 0] < .05 * np.maximum(values[:, 1], 1e-8)
    valid &= (np.abs(denominator) > .2) & (ranges > 2) & (ranges < 100)
    valid &= spread < np.maximum(.5, .05 * centers[:, 2])
    rows, target, ranges, rms, denominator = rows[valid], target[valid], ranges[valid], rms[valid], denominator[valid]
    world_target = target @ T_WC[:3, :3].T + T_WC[:3, 3]
    result['xyz_target_world'][rows] = world_target
    result['geometry_valid'][rows] = True
    result['geometry_quality'][rows] = .25 * np.exp(-rms / .08) * np.abs(denominator)
    result['sigma_parallel_m'][rows] = np.maximum(.3, 3 * rms / np.abs(denominator))
    result['sigma_perpendicular_m'][rows] = np.maximum(.1, ranges * 2 / min(K[0, 0], K[1, 1]))
    result['surface_id'][rows] = np.floor(world_target / .5).astype(np.int64)
    return result


def read_geometry(path):
    with np.load(Path(path), allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    valid = result['geometry_valid'].astype(bool)
    for key in ('xyz_target_world', 'geometry_quality', 'sigma_parallel_m', 'sigma_perpendicular_m'):
        if not np.isfinite(result[key][valid]).all():
            raise ValueError(f'Nonfinite geometry: {key}')
    for key in ('sigma_parallel_m', 'sigma_perpendicular_m'):
        if (result[key][valid] <= 0).any():
            raise ValueError(f'Nonpositive geometry uncertainty: {key}')
    result['geometry_valid'] = valid
    return result


def visible_voxels(world, K, T_WC, mask, voxel_size=.5):
    h, w = mask.shape
    camera = (world - T_WC[:3, 3]) @ T_WC[:3, :3]
    valid = np.isfinite(camera).all(1) & (camera[:, 2] > 2) & (camera[:, 2] < 80)
    camera, world = camera[valid], world[valid]
    projection = camera @ K.T
    uv = projection[:, :2] / projection[:, 2:]
    inside = (uv >= 0).all(1) & (uv[:, 0] < w) & (uv[:, 1] < h)
    camera, world, uv = camera[inside], world[inside], uv[inside]
    cells = np.floor(uv).astype(int)
    visible = mask[cells[:, 1], cells[:, 0]].astype(bool)
    camera, world, uv, cells = camera[visible], world[visible], uv[visible], cells[visible]
    flat = cells[:, 1] * w + cells[:, 0]
    zbuffer = np.full(h*w, np.inf)
    np.minimum.at(zbuffer, flat, camera[:, 2])
    keep = camera[:, 2] < zbuffer[flat] + .1
    ids = np.floor(world[keep] / voxel_size).astype(np.int64)
    blocks = np.clip((uv[keep] / [w, h] * 4).astype(int), 0, 3)
    mapping = {}
    for voxel, block in zip(ids, blocks):
        mapping.setdefault(tuple(voxel.tolist()), set()).add(int(block[0] + 4*block[1]))
    return mapping
