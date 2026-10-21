"""Field accuracy versus voxel size, on points drawn from the real scan.

Prints, per voxel size, the cell count, EDT build time and the error of the
trilinearly sampled field against the exact KD-tree distance.

A useful signature to keep in mind: an error that does not move when the voxel
size changes means the lookup is wrong, not the quantisation.
"""
import json
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy import ndimage

sys.path.insert(0, '/home/zhangcy/projects/r5b-lossless-20261019-v1')
sys.path.insert(0, '/home/zhangcy/projects/r5c-distancefield-20261019-v1')
import fast_lib as F
import df_core as DF

H = '/data2/zhangcy/v7-raw-geometry-hard1502-20261013-v1/'
sel = json.load(open(H + 'selection.json'))
refs, qs = sel['references'], sel['queries']
z = np.load(H + 'scored.npz', allow_pickle=True)
keys = [str(k) for k in z['keys']]
v7 = z['refined_V7_poses']
qmap = {str(q['key']): q for q in qs}
rpos = np.asarray([r['world_from_body'] for r in refs])[:, :3, 3]

_DEV = DF.pick_device()


def edt_field(points, voxel, trunc=1.0):
    lo = points.min(0) - (trunc + voxel)
    hi = points.max(0) + (trunc + voxel)
    dims = np.maximum(np.ceil((hi - lo) / voxel).astype(np.int64), 4)
    occ = np.zeros(tuple(dims), bool)
    ijk = np.clip(np.floor((points - lo) / voxel).astype(np.int64), 0, dims - 1)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    t0 = time.perf_counter()
    full = ndimage.distance_transform_edt(~occ).astype(np.float32) * np.float32(voxel)
    build = time.perf_counter() - t0
    return np.minimum(full, np.float32(trunc)), lo, build, dims


def sample_grid(field, lo, voxel, pts, trunc=1.0):
    idx = (pts - lo) / voxel
    dims = np.array(field.shape)
    inside = np.all((idx >= 0) & (idx <= dims - 1), axis=1)
    norm = 2.0 * idx / (dims - 1.0) - 1.0
    t = torch.from_numpy(np.ascontiguousarray(field)).to(_DEV).view(1, 1, *field.shape)
    g = torch.from_numpy(np.ascontiguousarray(norm[:, ::-1])).to(_DEV).float().view(1, 1, 1, -1, 3)
    v = torch.nn.functional.grid_sample(t, g, mode='bilinear', padding_mode='border',
                                        align_corners=True).view(-1)
    v = v.cpu().numpy().astype(np.float64)
    v[~inside] = trunc
    return v


if __name__ == '__main__':
    maps = F.MapSource(refs, rpos)
    frames = [int(x) for x in np.unique(np.linspace(0, 1501, 5, dtype=int))]
    for i in frames:
        c = v7[i][:3, 3]
        M = maps.local(c)
        scan = F.decode(qmap[keys[i]]['scan_path'])
        scan = scan[(np.linalg.norm(scan, axis=1) > 1) & (np.linalg.norm(scan, axis=1) < 60)]
        scan = F.voxel(scan, 0.3)
        P = scan[:3000] @ v7[i][:3, :3].T + c
        exact = np.minimum(cKDTree(M).query(P, k=1, workers=4)[0], 1.0)
        print('\n--- frame %d  map points %d  extent %s'
              % (i, len(M), np.round(M.max(0) - M.min(0), 1)), flush=True)
        for voxel in (0.4, 0.3, 0.2, 0.15):
            field, lo, build, dims = edt_field(M, voxel)
            approx = sample_grid(field, lo, voxel, P)
            err = np.abs(approx - exact)
            print('  voxel %.2f  cells=%6.1fM  build=%6.2fs  MAE=%.4f  p95=%.4f  corr=%.4f'
                  % (voxel, np.prod(dims) / 1e6, build, err.mean(),
                     np.percentile(err, 95), np.corrcoef(approx, exact)[0, 1]), flush=True)
