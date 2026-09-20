import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def diagnose(data):
    rows = json.loads((data / 'manifest.json').read_text())['train']
    poses = np.load(data / 'train/poses.npy')
    report = []
    for index in np.linspace(0, len(rows)-1, 12).astype(int):
        frame = rows[index]['frame_id']
        cloud = Path(rows[index]['geometry_path'])
        if not cloud.exists():
            continue
        world = np.load(cloud)
        feature = dict(np.load(data / 'proc/features_train' / (frame + '.npz')))
        camera = (world - poses[index, :3, 3]) @ poses[index, :3, :3]
        camera = camera[(camera[:, 2] > 2) & (camera[:, 2] < 80)]
        pixel = camera @ feature['K'].T
        uv = pixel[:, :2] / pixel[:, 2:]
        h, w = feature['image_size_hw']
        inside = (uv >= 0).all(1) & (uv[:, 0] < w) & (uv[:, 1] < h)
        camera, uv = camera[inside], uv[inside]
        if len(camera) < 12:
            continue
        distances, neighbors = cKDTree(uv).query(feature['uv'], k=12, distance_upper_bound=6)
        valid = np.isfinite(distances).all(1)
        points = camera[neighbors[valid]]
        center = points.mean(1)
        delta = points - center[:, None]
        cov = np.einsum('nki,nkj->nij', delta, delta) / 12
        values, vectors = np.linalg.eigh(cov)
        planar = (values[:, 0] < .08**2) & (values[:, 1] > .0004) & (values[:, 0] < .05 * values[:, 1])
        report.append(dict(frame=frame, world_points=len(world), visible_points=len(camera),
            supported=int(valid.sum()), planar=int(planar.sum()), median_eigenvalues=np.median(values, axis=0).tolist() if len(values) else []))
    print(json.dumps(report, indent=2))
