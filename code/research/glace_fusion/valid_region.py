import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def validate_mask(mask):
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2 or not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise ValueError('Expected a finite two-dimensional mask in [0, 1]')
    if not np.any(mask >= 1 - 1e-6):
        raise ValueError('Mask has no fully valid image region')
    return mask


def resize_valid_mask(mask, height, width):
    mask = validate_mask(mask)
    return np.asarray(Image.fromarray(mask).resize((width, height), Image.Resampling.BILINEAR)).copy()


def sample_valid_region(mask, uv):
    mask = validate_mask(mask)
    uv = np.asarray(uv)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
        raise ValueError('Expected finite Nx2 pixel coordinates')
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < mask.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < mask.shape[0])
    valid = np.zeros(len(uv), dtype=bool)
    indices = np.floor(uv[inside]).astype(np.int64)
    valid[inside] = mask[indices[:, 1], indices[:, 0]] >= 1 - 1e-6
    return valid


def grid_valid_region(mask, height, width, stride=8):
    import torch

    ys = (torch.arange(height, device=mask.device) * stride + stride // 2).long()
    xs = (torch.arange(width, device=mask.device) * stride + stride // 2).long()
    inside = (ys[:, None] < mask.shape[-2]) & (xs[None, :] < mask.shape[-1])
    sampled = mask[..., ys.clamp(max=mask.shape[-2] - 1)[:, None], xs.clamp(max=mask.shape[-1] - 1)[None, :]]
    return (sampled >= 1 - 1e-6) & inside


def make_valid_mask(map_u, map_v, source_shape, output_shape):
    if map_u.shape != map_v.shape or not np.isfinite(map_u).all() or not np.isfinite(map_v).all():
        raise ValueError('Invalid undistortion maps')
    valid = cv2.remap(np.ones(source_shape, np.float32), map_u.astype(np.float32),
                      map_v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return validate_mask(cv2.resize(valid, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_AREA))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-u', type=Path, required=True)
    parser.add_argument('--map-v', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-height', type=int, default=1232)
    parser.add_argument('--source-width', type=int, default=1616)
    parser.add_argument('--height', type=int, default=616)
    parser.add_argument('--width', type=int, default=808)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    mask = make_valid_mask(np.load(args.map_u), np.load(args.map_v),
        (args.source_height, args.source_width), (args.height, args.width))
    np.save(args.output, mask)
    report = dict(source_shape=[args.source_height, args.source_width], shape=list(mask.shape),
        valid_fraction=float(np.mean(mask >= 1 - 1e-6)),
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        map_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.map_u, args.map_v]})
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
