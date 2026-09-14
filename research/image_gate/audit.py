import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fusion import project
from run import read_scan


root = Path('/home/zhang/leader-image-gate')
rows = json.loads((root / 'manifest.json').read_text())
bundle = Path(rows[0]['image']).parents[4]
extrinsic = torch.tensor(json.loads((bundle / 'data/validation_scene/scene_meta.json').read_text())['T_BC_camera_to_body'], dtype=torch.float32)
stats = []
for i, row in enumerate(rows):
    image = Image.open(row['image'])
    w, h = image.size
    k = torch.tensor(np.loadtxt(row['calibration']), dtype=torch.float32)
    with np.load(root / 'lidar' / (row['frame_id'] + '.npz')) as data:
        source = torch.tensor(data['source'])
        lidar_seconds = float(data['seconds'])
    with np.load(root / 'visual' / (row['frame_id'] + '.npz')) as data:
        visible = data['valid']
        visual_seconds = float(data['seconds'])
    uv, _, fov = project(source, extrinsic, k, (h, w))
    stats.append(dict(frame_id=row['frame_id'], split=row['split'], voxels=len(source),
                      in_fov=int(fov.sum()), visible=int(visible.sum()),
                      visible_fraction=float(visible.mean()), lidar_seconds=lidar_seconds, visual_seconds=visual_seconds))
    if i == 0:
        raw, _ = read_scan(row['scan'])
        rawuv, depth, rawvalid = project(torch.tensor(raw), extrinsic, k, (h, w))
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax in axes:
            ax.imshow(image)
            ax.axis('off')
        axes[0].scatter(rawuv[rawvalid, 0], rawuv[rawvalid, 1], c=depth[rawvalid], s=1, cmap='turbo', vmin=2, vmax=50)
        axes[0].set_title('Raw scan projection (no GT alignment)')
        axes[1].scatter(uv[fov, 0], uv[fov, 1], c='orange', s=12, label='voxel in FOV')
        axes[1].scatter(uv[visible, 0], uv[visible, 1], c='lime', s=18, label='accepted visible voxel')
        axes[1].set_title('LEADER voxel centers and visibility mask')
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(root / 'projection.png', dpi=130)
summary = {}
for split in ['train', 'val']:
    subset = [x for x in stats if x['split'] == split]
    summary[split] = dict(frames=len(subset), visible_fraction_mean=float(np.mean([x['visible_fraction'] for x in subset])),
                         visible_count_mean=float(np.mean([x['visible'] for x in subset])),
                         fov_fraction=float(sum(x['in_fov'] for x in subset) / sum(x['voxels'] for x in subset)),
                         zero_visible_frames=sum(x['visible'] == 0 for x in subset),
                         lidar_seconds_mean=float(np.mean([x['lidar_seconds'] for x in subset])),
                         visual_seconds_mean=float(np.mean([x['visual_seconds'] for x in subset])))
(root / 'coverage.json').write_text(json.dumps(dict(summary=summary, frames=stats), indent=2))
print(json.dumps(summary, indent=2))
