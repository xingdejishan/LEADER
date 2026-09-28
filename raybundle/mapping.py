import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


CAM5_K = np.array([[199.716592, 0, 412.930976], [0, 199.716592, 310.584312], [0, 0, 1.]], dtype=np.float64)
TRAIN_DATES = ('2012-01-22', '2012-02-02', '2012-02-18', '2012-05-11')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, date, relative):
    base = Path(root).resolve()
    target = (base / date / relative).resolve()
    if base not in target.parents:
        raise ValueError('dataset path escapes its root')
    return target


def read_records(root, manifests, mapping=False):
    records = []
    for path in manifests:
        payload = json.loads(Path(path).read_text())
        date = payload['date']
        if mapping and (date not in TRAIN_DATES or payload['split'] != 'train'):
            raise ValueError('reference rays require training dates only')
        t_bc = np.asarray(payload['roi']['T_BC'], dtype=np.float64)
        for row in payload['frames']:
            item = dict(key=date + '/' + str(row['timestamp_us']), image=safe_path(root, date, row['image']['path']),
                        scan=safe_path(root, date, row['scan']['path']), image_sha=row['image']['sha256'],
                        scan_sha=row['scan']['sha256'], T_BC=t_bc.copy(), manifest_sha=sha256(path))
            if mapping:
                gt = row['ground_truth']
                gp = safe_path(root, date, gt['path'])
                if gt['pose'] != 'T_WC' or sha256(gp) != gt['sha256']:
                    raise ValueError('invalid reference pose provenance')
                item['T_WC'] = np.loadtxt(gp).reshape(4, 4)
            records.append(item)
    return records


def extract(image_path, max_features=512, expected_sha=None):
    cv2.setNumThreads(1)
    if not 1 <= max_features <= 2048:
        raise ValueError('feature budget must be in [1,2048]')
    if expected_sha and sha256(image_path) != expected_sha:
        raise ValueError('image SHA mismatch')
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (616, 808):
        raise ValueError('expected undistorted 808 by 616 Cam5 image')
    keypoints, desc = cv2.SIFT_create(nfeatures=max_features).detectAndCompute(image, None)
    if desc is None:
        return np.empty((0, 2)), np.empty((0, 128))
    order = sorted(range(len(keypoints)), key=lambda k: (-keypoints[k].response, keypoints[k].pt))[:max_features]
    pixels = np.array([keypoints[k].pt for k in order], dtype=np.float64)
    desc = desc[order].astype(np.float64)
    desc = np.sqrt(desc / np.maximum(desc.sum(1, keepdims=True), 1e-12))
    return pixels, desc


def camera_rays(pixels, intrinsics=CAM5_K):
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    rays = homogeneous @ np.linalg.inv(intrinsics).T
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def fit_bundle(origins, directions, angular_sigma=.005):
    projection = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    h = projection.sum(0)
    eigenvalues = np.linalg.eigvalsh(h)
    if eigenvalues[0] <= 1e-9 or np.max(np.linalg.norm(origins - origins[0], axis=1)) < .25:
        return None
    initial = np.linalg.solve(h, np.einsum('nij,nj->i', projection, origins))
    def residual(point):
        delta = point - origins
        unit = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-8)
        return ((unit - directions) / angular_sigma).reshape(-1)
    fit = least_squares(residual, initial, max_nfev=20)
    delta = fit.x - origins
    depth = (delta * directions).sum(1)
    if not fit.success or depth.min() <= .5 or depth.max() > 2000 or np.max(np.linalg.norm(residual(fit.x).reshape(-1, 3), axis=1)) > 2.5:
        return None
    precision = fit.jac.T @ fit.jac
    if np.linalg.eigvalsh(precision)[0] <= 1e-12:
        return None
    return fit.x, precision


def build_map(records, max_features=512, max_tracks=20000, neighbors=4):
    if not 2 <= len(records) <= 1200 or not 1 <= max_tracks <= 20000:
        raise ValueError('map construction resource budget exceeded')
    features, rays, poses = [], [], []
    for record in records:
        pixels, descriptors = extract(record['image'], max_features, record['image_sha'])
        features.append((pixels, descriptors))
        pose = record['T_WC']
        poses.append(pose)
        rays.append(camera_rays(pixels) @ pose[:3, :3].T)
    centers = np.stack([p[:3, 3] for p in poses])
    parent, members, frame_members = {}, {}, {}
    def find(node):
        if node not in parent:
            parent[node] = node
            members[node] = {node}
            frame_members[node] = {node[0]}
        while parent[node] != node:
            node = parent[node]
        return node
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = set()
    for i in range(len(records)):
        distances = np.linalg.norm(centers - centers[i], axis=1)
        for j in np.argsort(distances):
            if j == i or distances[j] < .25 or distances[j] > 15:
                continue
            pairs.add(tuple(sorted((i, int(j)))))
            if sum(i in pair for pair in pairs) >= neighbors:
                break
    for i, j in sorted(pairs):
        a, b = features[i][1].astype(np.float32), features[j][1].astype(np.float32)
        if min(len(a), len(b)) < 2:
            continue
        forward = matcher.knnMatch(a, b, k=2)
        reverse = {m[0].queryIdx: m[0].trainIdx for m in matcher.knnMatch(b, a, k=2) if len(m) == 2 and m[0].distance < .8 * m[1].distance}
        for pair in forward:
            if len(pair) != 2 or pair[0].distance >= .8 * pair[1].distance:
                continue
            match = pair[0]
            if reverse.get(match.trainIdx) != match.queryIdx:
                continue
            observation = [(i, match.queryIdx), (j, match.trainIdx)]
            if fit_bundle(centers[[i, j]], np.stack([rays[x][y] for x, y in observation])) is None:
                continue
            left, right = [find(x) for x in observation]
            if left == right or frame_members[left].intersection(frame_members[right]):
                continue
            parent[right] = left
            members[left].update(members.pop(right))
            frame_members[left].update(frame_members.pop(right))
    means, precisions, descriptors, observations, origin_list, direction_list, offsets = [], [], [], [], [], [], [0]
    reference_centers, reference_radii = [], []
    for component in members.values():
        if len(component) < 2 or len(means) >= max_tracks:
            continue
        obs = sorted(component)
        origins = np.stack([centers[i] for i, _ in obs])
        directions = np.stack([rays[i][k] for i, k in obs])
        fitted = fit_bundle(origins, directions)
        if fitted is None:
            continue
        means.append(fitted[0])
        precisions.append(fitted[1])
        descriptors.append(np.mean([features[i][1][k] for i, k in obs], axis=0))
        reference_center = origins.mean(0)
        reference_centers.append(reference_center)
        reference_radii.append(np.linalg.norm(origins-reference_center,axis=1).max())
        observations.append([dict(frame=records[i]['key'], pixel=features[i][0][k].tolist()) for i, k in obs])
        origin_list.extend(origins)
        direction_list.extend(directions)
        offsets.append(len(origin_list))
    if not means:
        raise ValueError('no multi-view supported reference tracks')
    data = dict(mean=np.asarray(means), precision=np.asarray(precisions), descriptors=np.asarray(descriptors),
                reference_centers=np.asarray(reference_centers), reference_radii=np.asarray(reference_radii),
                ray_origins=np.asarray(origin_list), ray_directions=np.asarray(direction_list), offsets=np.asarray(offsets))
    metadata = dict(schema='leader-ray-bundle-map-v1', reference_frames=[r['key'] for r in records],
                    image_sha256=[r['image_sha'] for r in records], manifest_sha256=sorted({r['manifest_sha'] for r in records}),
                    observations=observations, extractor='RootSIFT128', angular_sigma=.005,
                    depth_bounds_m=[.5, 2000], weak_depth='finite anisotropic Laplace covariance; no artificial precision floor',
                    reference_poses='training T_WC only; treated as fixed', tracks=len(means))
    return data, metadata
