import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import MinkowskiEngine as ME
from scipy.spatial import cKDTree
from fusion import project, sample_map
from run import REPO, WORKSPACE, read_scan, load_leader, save_json
sys.path.insert(0, str(REPO))
from utils.pose_util import cartesian_to_polar_expansion, polar_expansion_to_cartesian


def representative_points(raw_xyz, input_coordinates, keep_indices, output_coordinates, tensor_stride):
    stride = np.asarray(tensor_stride, dtype=np.int64)
    parents = np.floor_divide(np.asarray(input_coordinates), stride) * stride
    cells, first = np.unique(parents, axis=0, return_index=True)
    lookup = {tuple(cell): int(keep_indices[i]) for cell, i in zip(cells, first)}
    selected = np.array([lookup.get(tuple(cell), -1) for cell in np.asarray(output_coordinates)], dtype=np.int64)
    supported = selected >= 0
    points = np.zeros((len(selected), 3), np.float32)
    points[supported] = raw_xyz[selected[supported]]
    return points, selected, supported


def stages(points, raw, extrinsic, k, mask, supported=None):
    h, w = mask.shape
    uv, depth, inside = project(points, extrinsic, k, (h, w))
    if supported is not None:
        inside &= supported
    black = sample_map(mask[None, None], uv, (h, w))[:, 0] > .999
    su, sd, sv = project(raw, extrinsic, k, (h, w))
    gh, gw = (h + 3) // 4, (w + 3) // 4
    z = torch.full((gh * gw,), float('inf'))
    cells = (su[sv] / 4).long()
    z.scatter_reduce_(0, cells[:, 1] * gw + cells[:, 0], sd[sv], reduce='amin')
    query = (uv / 4).long()
    ref = z[query[:, 1].clamp(0, gh - 1) * gw + query[:, 0].clamp(0, gw - 1)]
    support = torch.isfinite(ref)
    close = (depth - ref).abs() <= .5
    masks = [inside, inside & black, inside & black & support, inside & black & support & close]
    stats = dict(zip(['fov', 'black_mask', 'depth_support', 'depth_consistent'], [int(v.sum()) for v in masks]))
    values = (depth - ref).abs()[masks[2]].numpy()
    stats['supported_depth_error_median'] = float(np.median(values)) if len(values) else None
    return uv, masks[-1], stats


def voxel_zbuffer(points, extrinsic, k, mask):
    h, w = mask.shape
    uv, depth, valid = project(points, extrinsic, k, (h, w))
    valid &= sample_map(mask[None, None], uv, (h, w))[:, 0] > .999
    gh, gw = (h + 7) // 8, (w + 7) // 8
    cells = (uv / 8).long()
    indices = cells[:, 1].clamp(0, gh - 1) * gw + cells[:, 0].clamp(0, gw - 1)
    buffer = torch.full((gh * gw,), float('inf'))
    buffer.scatter_reduce_(0, indices[valid], depth[valid], reduce='amin')
    return valid & (depth <= buffer[indices] + .5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/home/zhang/leader-image-gate'))
    args = parser.parse_args()
    out = args.root / 'projection_audit'
    (out / 'mapping').mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    rows = json.loads((args.root / 'manifest.json').read_text())
    model = load_leader(SimpleNamespace(checkpoint=WORKSPACE / 'research/image_gate_checkpoint'))
    bundle = WORKSPACE / 'glace-local'
    extrinsic = torch.tensor(json.loads((bundle / 'data/validation_scene/scene_meta.json').read_text())['T_BC_camera_to_body'], dtype=torch.float32)
    mask = torch.tensor(np.load(bundle / 'data/valid_mask.npy'), dtype=torch.float32)
    records, roundtrips, offsets = [], [], []
    for i, row in enumerate(rows):
        scan, intensity = read_scan(row['scan'])
        polar = cartesian_to_polar_expansion(scan, .2 * 1024)
        feature = np.column_stack([polar[:, 2], polar[:, 1], intensity]).astype(np.float32)
        coords, features, keep = ME.utils.sparse_quantize(coordinates=polar, features=feature, quantization_size=.2, return_index=True)
        sparse = ME.SparseTensor(torch.tensor(features, device='cuda'), ME.utils.batched_coordinates([coords]).cuda())
        with torch.inference_mode():
            enc = model.encoder(sparse)
        output_coords = enc.C[:, 1:].cpu().numpy()
        stride = enc.tensor_stride
        center = polar_expansion_to_cartesian((enc.C[:, 1:].float() + torch.tensor(stride, device='cuda') / 2) * .2, .2 * 1024).cpu()
        with np.load(args.root / 'lidar' / (row['frame_id'] + '.npz')) as cached:
            distances, order = cKDTree(center.numpy()).query(cached['source'])
            assert distances.max() < 1e-4 and len(np.unique(order)) == len(order), f'Output cells differ: {distances.max()}'
            output_coords = output_coords[order]
            feature_error = np.max(np.abs(enc.F.cpu().numpy()[order] - cached['features']))
            center = torch.tensor(cached['source'])
        xyz, indices, supported = representative_points(scan, coords, keep, output_coords, stride)
        chosen_polar = np.floor(cartesian_to_polar_expansion(xyz[supported], .2 * 1024) / .2).astype(np.int64)
        assert np.array_equal(chosen_polar // stride * stride, output_coords[supported])
        reconstructed = polar_expansion_to_cartesian(torch.tensor(polar), .2 * 1024).numpy()
        roundtrips.append(float(np.max(np.linalg.norm(reconstructed - scan, axis=-1))))
        offsets.extend(np.linalg.norm(center.numpy()[supported] - xyz[supported], axis=-1).tolist())
        im = Image.open(row['image']).convert('RGB')
        w, h = im.size
        nh, nw = int(np.ceil(h * 480 / min(h, w) / 8)) * 8, int(np.ceil(w * 480 / min(h, w) / 8)) * 8
        k = torch.tensor(np.loadtxt(row['calibration']), dtype=torch.float32)
        k[0] *= nw / w
        k[1] *= nh / h
        m = F.interpolate(mask[None, None], size=(nh, nw), mode='nearest')[0, 0]
        raw = torch.tensor(scan)
        uv_c, valid_c, cs = stages(center, raw, extrinsic, k, m)
        uv_r, valid_r, rs = stages(torch.tensor(xyz), raw, extrinsic, k, m, torch.tensor(supported))
        doc_valid = voxel_zbuffer(center, extrinsic, k, m)
        with np.load(args.root / 'visual' / (row['frame_id'] + '.npz')) as cached:
            assert np.array_equal(valid_c.numpy(), cached['valid']), 'Old visibility was not reproduced exactly'
        np.savez(out / 'mapping' / (row['frame_id'] + '.npz'), output_coordinates=output_coords,
                 tensor_stride=np.array(stride), raw_index_filtered_scan=indices,
                 projection_xyz=xyz, projection_supported=supported, localization_xyz=center.numpy(),
                 uv=uv_r.numpy(), valid=valid_r.numpy(), image_hw=np.array([nh, nw]))
        records.append(dict(frame_id=row['frame_id'], split=row['split'], voxels=len(center), input_voxels=len(coords),
                            stride=stride, no_raw_support=int((~supported).sum()), reencoding_feature_max_error=float(feature_error), center=cs, representative=rs,
                            document_voxel_zbuffer=int(doc_valid.sum())))
        if i in [0, 32, 64]:
            rawuv, rawdepth, rawvalid = project(raw, extrinsic, k, (nh, nw))
            rawvalid &= sample_map(m[None, None], rawuv, (nh, nw))[:, 0] > .999
            fig, axes = plt.subplots(1, 3, figsize=(17, 6))
            panels = [(rawuv, rawvalid, 'Raw Cartesian scan (reference)'),
                      (uv_c, valid_c, f'Voxel center: {int(valid_c.sum())}/{len(center)}'),
                      (uv_r, valid_r, f'Raw point per output cell: {int(valid_r.sum())}/{len(center)}')]
            for ax, (uv, valid, title) in zip(axes, panels):
                ax.imshow(im.resize((nw, nh), Image.Resampling.BILINEAR))
                ax.scatter(uv[valid, 0], uv[valid, 1], s=2 if title.startswith('Raw Cartesian') else 8,
                           c=rawdepth[valid] if title.startswith('Raw Cartesian') else 'lime', cmap='turbo', vmin=2, vmax=50)
                ax.set_title(title)
                ax.set_xlim(0, nw)
                ax.set_ylim(nh, 0)
                ax.axis('off')
            fig.tight_layout()
            fig.savefig(out / (row['frame_id'] + '.png'), dpi=140)
            plt.close(fig)
        print(f'audit {i+1}/96 center={cs["depth_consistent"]} raw={rs["depth_consistent"]} doc={int(doc_valid.sum())} N={len(center)} stride={stride}', flush=True)
    summary = {}
    for split in ['train', 'val', 'all']:
        subset = [r for r in records if split == 'all' or r['split'] == split]
        n = sum(r['voxels'] for r in subset)
        summary[split] = dict(frames=len(subset), total_voxels=n, missing_raw=sum(r['no_raw_support'] for r in subset),
                              center={k: sum(r['center'][k] for r in subset) / n for k in ['fov', 'black_mask', 'depth_support', 'depth_consistent']},
                              representative={k: sum(r['representative'][k] for r in subset) / n for k in ['fov', 'black_mask', 'depth_support', 'depth_consistent']},
                              center_frame_mean=float(np.mean([r['center']['depth_consistent'] / r['voxels'] for r in subset])),
                              raw_frame_mean=float(np.mean([r['representative']['depth_consistent'] / r['voxels'] for r in subset])),
                              document_voxel_zbuffer=sum(r['document_voxel_zbuffer'] for r in subset) / n)
    save_json(out / 'report.json', dict(summary=summary, roundtrip_max_m=max(roundtrips),
              center_to_raw_distance_percentiles=np.percentile(offsets, [50, 95, 100]).tolist(), records=records,
              claim='Projection-only audit; raw points are not substituted for LEADER localization coordinates or targets'))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
