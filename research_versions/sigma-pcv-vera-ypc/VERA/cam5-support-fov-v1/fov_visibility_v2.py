import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch


def load_coverage(path, digest):
    path = Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    value = np.load(path, allow_pickle=False)
    assert value.shape == (616, 808) and value.dtype == np.float32
    assert np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
    return value


def project(points, camera):
    transform = np.asarray(camera['T_CB'], dtype=np.float64)
    intrinsics = np.asarray(camera['K'], dtype=np.float64)
    xyz = np.asarray(points, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]
    uv = (xyz @ intrinsics.T)[..., :2] / xyz[..., 2:3].clip(1e-8)
    inside = np.isfinite(xyz).all(-1) & np.isfinite(uv).all(-1)
    inside &= (xyz[..., 2] > 1e-6) & (uv[..., 0] >= 0) & (uv[..., 0] <= 807)
    inside &= (uv[..., 1] >= 0) & (uv[..., 1] <= 615)
    return uv, xyz[..., 2], inside


def sample_coverage(coverage, uv):
    shape = uv.shape[:-1]
    locations = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    locations = np.nan_to_num(locations, nan=-2., posinf=810., neginf=-2.)
    locations[:, 0] = locations[:, 0].clip(-2, 810)
    locations[:, 1] = locations[:, 1].clip(-2, 618)
    sampled = np.empty(len(locations), dtype=np.float32)
    for start in range(0, len(locations), 30000):
        chunk = locations[start:start + 30000]
        sampled[start:start + len(chunk)] = cv2.remap(coverage, chunk[:, 0:1], chunk[:, 1:2], cv2.INTER_LINEAR,
                                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0).ravel()
    return sampled.reshape(shape)


def prepare(raw, cameras, coverage, threshold=.999):
    result = []
    for camera in cameras:
        uv, depth, valid = project(raw, camera)
        valid &= sample_coverage(coverage, uv) >= threshold
        pixels = np.rint(uv[valid]).astype(np.int64)
        raster = np.full(616 * 808, np.inf, dtype=np.float32)
        np.minimum.at(raster, pixels[:, 1] * 808 + pixels[:, 0], depth[valid])
        raster = cv2.erode(raster.reshape(616, 808), np.ones((3, 3), np.uint8))
        result.append({'camera': camera, 'coverage': coverage, 'threshold': threshold, 'depth': raster})
    return result


def validity(points, context, tolerance=.3):
    uv, depth, valid = project(points, context['camera'])
    full_coverage = sample_coverage(context['coverage'], uv) >= context['threshold']
    safe_uv = np.nan_to_num(uv)
    safe_uv[..., 0] = safe_uv[..., 0].clip(0, 807)
    safe_uv[..., 1] = safe_uv[..., 1].clip(0, 615)
    pixels = np.rint(safe_uv).astype(np.int64)
    observed = context['depth'][pixels[..., 1], pixels[..., 0]]
    valid &= full_coverage & np.isfinite(observed) & (np.abs(depth - observed) <= tolerance)
    return valid


def filter_geometry(geometry, contexts, tolerance=.3):
    result = {name: value.copy() for name, value in geometry.items()}
    for camera_index, context in enumerate(contexts):
        indices = np.flatnonzero(result['camera'] == camera_index)
        if not len(indices):
            continue
        candidates = result['support_points'][indices, None].astype(np.float64)
        candidates = candidates + np.einsum('nij,mj->nmi', result['basis'][indices].astype(np.float64), result['eta'].astype(np.float64))
        mask = result['valid'][indices] & validity(candidates, context, tolerance)
        mask &= mask[:, ::-1]
        center_valid = validity(result['support_points'][indices], context, tolerance)
        active = result['active'][indices] & center_valid & mask[:, 24] & (mask.sum(1) > 1)
        mask[~active] = False
        result['valid'][indices] = mask
        result['active'][indices] = active
    assert np.array_equal(result['valid'], result['valid'][:, ::-1])
    assert result['valid'][result['active'], 24].all()
    assert not result['valid'][~result['active']].any()
    return result


def safe_displacement(support, basis, eta, context, tolerance=.3):
    assert support.ndim == 2 and support.shape[1] == 3
    assert basis.shape == (len(support), 3, 2) and eta.shape == (len(support), 2)
    support_cpu = support.detach().cpu().double().numpy()
    delta = torch.einsum('nij,nj->ni', basis, eta)
    delta_cpu = delta.detach().cpu().double().numpy()
    assert validity(support_cpu, context, tolerance).all(), 'support must already be admissible'
    scale = np.full(len(support), -1., dtype=np.float32)
    for value in (1., .5, .25, .125, .0625, 0.):
        proposed = support_cpu + value * delta_cpu
        passed = validity(proposed, context, tolerance)
        scale[(scale < 0) & passed] = value
    assert (scale >= 0).all()
    multiplier = torch.as_tensor(scale, device=eta.device, dtype=eta.dtype)[:, None]
    safe_eta = eta * multiplier
    anchor = support + torch.einsum('nij,nj->ni', basis, safe_eta)
    final_valid = validity(anchor.detach().cpu().double().numpy(), context, tolerance)
    if not final_valid.all():
        scale[~final_valid] = 0.
        multiplier = torch.as_tensor(scale, device=eta.device, dtype=eta.dtype)[:, None]
        safe_eta = eta * multiplier
        anchor = support + torch.einsum('nij,nj->ni', basis, safe_eta)
    assert validity(anchor.detach().cpu().double().numpy(), context, tolerance).all()
    return anchor, safe_eta, multiplier[:, 0]
