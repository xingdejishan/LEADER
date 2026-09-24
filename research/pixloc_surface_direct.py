import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree, Delaunay, QhullError


SEED = 2089
SCALES = (1, 4, 16)
OUTPUT_INDICES = (0, 2, 4)
PATCH_AXIS = np.asarray([-4., 0., 4.], dtype=np.float64)
PATCH_PIXELS = np.stack(np.meshgrid(PATCH_AXIS, PATCH_AXIS, indexing="xy"), -1).reshape(-1, 2)
POSE_SCALE = np.asarray([.1, .1, .1, np.deg2rad(1.), np.deg2rad(1.), np.deg2rad(1.)],
                        dtype=np.float64)
POSE_EPS = 1e-4
ROBUST_SCALE = .1
SURFACE_PROTOCOL_VERSION = 3
SURFACE_NEIGHBOR_COUNT = 16
SURFACE_MAX_NEIGHBOR_DISTANCE_M = .75
SURFACE_MAX_PLANE_RMSE_M = .08
SURFACE_MAX_SUPPORT_DISTANCE_M = .25
SURFACE_MAX_PATCH_RADIUS_M = .5
TRAIN_EPOCHS = 20
TRAIN_STEPS_PER_SCALE = 5
INFERENCE_STEPS_PER_SCALE = 15


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_pose(path):
    return np.loadtxt(path, dtype=np.float64).reshape(4, 4)


def load_rgb(path, device):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.
    h, w = tensor.shape[-2:]
    pad_h, pad_w = (-h) % 16, (-w) % 16
    if pad_h or pad_w:
        tensor = F.pad(tensor[None], (0, pad_w, 0, pad_h), mode="replicate")[0]
    return tensor[None]


def load_pixloc(checkpoint_path, device):
    import sys
    import torchvision
    import omegaconf

    official_root = Path("/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/work/pixloc-official")
    sys.path.insert(0, str(official_root))
    original_vgg19 = torchvision.models.vgg19

    def vgg19_from_local_checkpoint(*args, **kwargs):
        kwargs.pop("pretrained", None)
        kwargs["weights"] = None
        return original_vgg19(*args, **kwargs)

    torchvision.models.vgg19 = vgg19_from_local_checkpoint
    from pixloc.pixlib.models.unet import UNet

    conf = omegaconf.OmegaConf.create({
        "name": "unet", "encoder": "vgg19", "decoder": [64, 64, 64, 32],
        "decoder_norm": "nn.BatchNorm2d", "output_scales": [0, 2, 4],
        "output_dim": [32, 128, 128], "compute_uncertainty": True,
        "checkpointed": False, "do_average_pooling": False,
    })
    model = UNet(conf).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {key[len("extractor."):]: value for key, value in checkpoint["model"].items()
             if key.startswith("extractor.")}
    model.load_state_dict(state, strict=True)
    torchvision.models.vgg19 = original_vgg19
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in model.adaptation:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    return model, checkpoint


def pre_features(model, image):
    mean = image.new_tensor(model.mean)[:, None, None]
    std = image.new_tensor(model.std)[:, None, None]
    value = (image - mean) / std
    skip = []
    for block in model.encoder:
        value = block(value)
        skip.append(value)
    if model.conf.decoder:
        pyramid = [skip[-1]]
        for block, skip_value in zip(model.decoder, skip[:-1][::-1]):
            pyramid.append(block(pyramid[-1], skip_value))
        pyramid = pyramid[::-1]
    else:
        pyramid = skip
    return [pyramid[index] for index in OUTPUT_INDICES]


def sample_map(feature_map, pixels, scale, mode="bilinear"):
    if feature_map.ndim == 2:
        feature_map = feature_map[None]
    height, width = feature_map.shape[-2:]
    xy = (pixels / float(scale)).to(dtype=feature_map.dtype)
    grid = torch.stack((2. * xy[:, 0] / max(width - 1, 1) - 1.,
                        2. * xy[:, 1] / max(height - 1, 1) - 1.), -1)
    sampled = F.grid_sample(feature_map[None], grid[None, :, None, :], mode=mode,
                            padding_mode="zeros", align_corners=True)
    return sampled[0, :, :, 0].transpose(0, 1)


def sample_reference_pyramid(model, ref_rows, frame, rows_by_frame, device):
    n, points_per_patch = frame["surface_valid"].shape
    raw_samples = [torch.zeros((n, points_per_patch, channels), dtype=torch.float32,
                               device=device) for channels in (32, 64, 512)]
    ref_conf = [torch.zeros((n, points_per_patch), dtype=torch.float32, device=device)
                for _ in SCALES]
    groups = sorted(set(zip(frame["reference_frames"].tolist(),
                            frame["reference_cameras"].tolist())))
    pixel_tensor = torch.as_tensor(frame["surface_pixels"], dtype=torch.float32, device=device)
    for ref_frame, ref_camera in groups:
        indices_np = np.where((frame["reference_frames"] == ref_frame) &
                              (frame["reference_cameras"] == ref_camera))[0]
        indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)
        ref_view = next(item for item in rows_by_frame[str(ref_frame)]["views"]
                        if int(item["camera"]) == int(ref_camera))
        with torch.no_grad():
            pyramid = pre_features(model, load_rgb(ref_view["image"], device))
            uncertainty = [torch.sigmoid(-layer(value))[0, 0]
                           for layer, value in zip(model.uncertainty, pyramid)]
            for level, (feature, conf, scale) in enumerate(zip(pyramid, uncertainty, SCALES)):
                sample_pixels = pixel_tensor[indices].reshape(-1, 2)
                raw = sample_map(feature[0], sample_pixels, scale)
                raw_samples[level][indices] = raw.reshape(len(indices), points_per_patch, -1)
                ref_conf[level][indices] = sample_map(conf, sample_pixels, scale).reshape(
                    len(indices), points_per_patch)
    return raw_samples, ref_conf


def camera_mask_tensor(view, device):
    mask = np.load(view["mask"], mmap_mode="r")
    return torch.as_tensor(np.asarray(mask, dtype=np.uint8).copy(), device=device)


def project_world_torch(points, y, baseline_pose, camera_to_body, calibration):
    from pixloc.pixlib.geometry.optimization import so3exp_map

    delta = y * torch.as_tensor(POSE_SCALE, dtype=y.dtype, device=y.device)
    rotation = so3exp_map(delta[3:]) @ baseline_pose[:3, :3]
    translation = baseline_pose[:3, 3] + delta[:3]
    camera_rotation = rotation @ camera_to_body[:3, :3]
    camera_translation = translation + rotation @ camera_to_body[:3, 3]
    camera_points = (points - camera_translation) @ camera_rotation
    homogeneous = camera_points @ calibration.T
    depth = camera_points[:, 2]
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3].clamp_min(1e-10)
    return pixels, depth


def project_pose_numpy(y, baseline_pose):
    from scipy.spatial.transform import Rotation

    delta = np.asarray(y, dtype=np.float64) * POSE_SCALE
    result = baseline_pose.copy()
    result[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix() @ baseline_pose[:3, :3]
    result[:3, 3] = baseline_pose[:3, 3] + delta[:3]
    return result


def geometry_jacobian(points, y, baseline, camera_to_body, calibration, scale):
    base_uv, _ = project_world_torch(points, y, baseline, camera_to_body, calibration)
    columns = []
    step = torch.full((6,), POSE_EPS, dtype=y.dtype, device=y.device)
    for axis in range(6):
        perturb = torch.zeros_like(y)
        perturb[axis] = step[axis]
        plus, _ = project_world_torch(points, y + perturb, baseline, camera_to_body, calibration)
        minus, _ = project_world_torch(points, y - perturb, baseline, camera_to_body, calibration)
        columns.append(((plus - minus) / (2. * POSE_EPS * scale)).unsqueeze(-1))
    jacobian = torch.cat(columns, -1)
    return base_uv / float(scale), jacobian


def robust_feature_terms(residual):
    squared = residual.square().sum(-1)
    c2 = ROBUST_SCALE ** 2
    weight = 2. / (2. + squared / c2)
    loss = 2. * c2 * torch.log1p(squared / (2. * c2))
    return loss, weight


def normal_equations(frame, row, bundle, y, level, visual_scale, device,
                     detach_jacobian=False):
    scale = SCALES[level]
    world = torch.as_tensor(frame["surface_points"], dtype=torch.float64, device=device)
    surface_valid = torch.as_tensor(frame["surface_valid"], dtype=torch.bool, device=device)
    camera_ids = torch.as_tensor(frame["cameras"], dtype=torch.long, device=device)
    patch_count, points_per_patch = surface_valid.shape
    flat_world = world.reshape(-1, 3)
    flat_valid = surface_valid.reshape(-1)
    patch_ids = torch.arange(patch_count, device=device).repeat_interleave(points_per_patch)
    cameras_flat = camera_ids.repeat_interleave(points_per_patch)
    ref_feature = bundle["reference_features"][level].reshape(-1,
                      bundle["reference_features"][level].shape[-1])
    ref_conf = bundle["reference_confidence"][level].reshape(-1)
    hessian = torch.zeros((6, 6), dtype=torch.float64, device=device)
    gradient = torch.zeros(6, dtype=torch.float64, device=device)
    objective = torch.zeros((), dtype=torch.float64, device=device)
    valid_counts = torch.zeros(patch_count, dtype=torch.float64, device=device)
    cached = []
    for camera in sorted(bundle["query"]):
        selected = torch.where(cameras_flat == int(camera))[0]
        if not len(selected):
            continue
        item = bundle["query"][int(camera)]
        points = flat_world[selected]
        uv, depth = project_world_torch(points, y, frame["baseline_pose_tensor"],
                                        item["camera_to_body"], item["calibration"])
        uv_feature, j_uv = geometry_jacobian(points, y, frame["baseline_pose_tensor"],
                                             item["camera_to_body"], item["calibration"], scale)
        feature_map = item["maps"][level]
        sample = sample_map(feature_map, uv, scale)
        x_offset = torch.tensor([.5 * scale, 0.], dtype=uv.dtype, device=device)
        y_offset = torch.tensor([0., .5 * scale], dtype=uv.dtype, device=device)
        gx = (sample_map(feature_map, uv + x_offset, scale) -
              sample_map(feature_map, uv - x_offset, scale))
        gy = (sample_map(feature_map, uv + y_offset, scale) -
              sample_map(feature_map, uv - y_offset, scale))
        spatial_grad = torch.stack((gx, gy), -1)
        jacobian = torch.matmul(spatial_grad.to(torch.float64), j_uv)
        if detach_jacobian:
            jacobian = jacobian.detach()
        residual = sample - ref_feature[selected]
        sample_loss, robust_weight = robust_feature_terms(residual)
        qconf = sample_map(item["confidence"][level], uv, scale).squeeze(-1)
        query_mask = sample_map(item["mask"].float(), uv, 1, mode="nearest").squeeze(-1) > .5
        map_h, map_w = feature_map.shape[-2:]
        uv_f = uv / float(scale)
        valid = flat_valid[selected] & (depth > .2) & query_mask
        valid &= (uv[:, 0] >= 1.) & (uv[:, 0] < item["mask"].shape[1] - 2.)
        valid &= (uv[:, 1] >= 1.) & (uv[:, 1] < item["mask"].shape[0] - 2.)
        valid &= (uv_f[:, 0] >= 2.) & (uv_f[:, 0] < map_w - 2.)
        valid &= (uv_f[:, 1] >= 2.) & (uv_f[:, 1] < map_h - 2.)
        valid &= torch.isfinite(residual).all(-1) & torch.isfinite(jacobian).all((-1, -2))
        sample_patch_ids = patch_ids[selected]
        valid_counts.scatter_add_(0, sample_patch_ids, valid.to(torch.float64))
        cached.append((selected, sample_patch_ids, residual, jacobian, sample_loss,
                       robust_weight, ref_conf[selected] * qconf, valid))
    for selected, sample_patch_ids, residual, jacobian, sample_loss, robust_weight, confidence, valid in cached:
        denom = valid_counts[sample_patch_ids].clamp_min(1.)
        weight = robust_weight.to(torch.float64) * confidence.to(torch.float64) / denom
        weight = weight * valid.to(torch.float64)
        j64 = jacobian.to(torch.float64)
        r64 = residual.to(torch.float64)
        hessian += torch.einsum("nci,ncj,n->ij", j64, j64, weight)
        gradient += torch.einsum("nci,nc,n->i", j64, r64, weight)
        objective += torch.sum(sample_loss.to(torch.float64) * confidence.to(torch.float64) *
                               valid.to(torch.float64) / denom)
    return visual_scale * hessian, visual_scale * gradient, objective, valid_counts


def solve_frame(frame, row, bundle, visual_scales, damping, device,
                steps_per_scale=INFERENCE_STEPS_PER_SCALE, initial_y=None,
                training=False):
    lidar = torch.as_tensor(frame["lidar_information"], dtype=torch.float64, device=device)
    pose_scale = torch.as_tensor(POSE_SCALE, dtype=torch.float64, device=device)
    lidar_hessian = pose_scale[:, None] * lidar * pose_scale[None, :]
    y = torch.zeros(6, dtype=torch.float64, device=device) if initial_y is None else initial_y
    base_y = y
    active_samples = []
    total_valid = 0
    for level in (2, 1, 0):
        for _ in range(steps_per_scale):
            visual_hessian, visual_gradient, objective, valid_counts = normal_equations(
                frame, row, bundle, y, level, visual_scales[level], device,
                detach_jacobian=training)
            valid_patches = valid_counts > 0
            total_valid = int(valid_patches.sum().detach().cpu())
            if total_valid < 4:
                break
            prior_gradient = lidar_hessian @ y
            hessian = visual_hessian + lidar_hessian
            gradient = visual_gradient + prior_gradient
            damp = torch.as_tensor(damping[level], dtype=torch.float64, device=device)
            diagonal = hessian.diagonal().clamp_min(1e-9)
            system = hessian + torch.diag(diagonal * damp.clamp(min=1e-5, max=1.))
            step = torch.linalg.solve(system, gradient)
            y = torch.clamp(y - step, -1., 1.)
        active_samples.append({"level_index": level, "valid_patches": total_valid,
                               "feature_objective": float(objective.detach().cpu())})
    if not torch.isfinite(y).all():
        raise RuntimeError("direct alignment produced a non-finite normalized pose update")
    pose = project_pose_numpy(y.detach().cpu().numpy(), frame["baseline_pose"])
    return y, pose, {"success": total_valid >= 4, "valid_patches": total_valid,
                     "scale_steps": active_samples,
                     "normalized_update": y.detach().cpu().numpy().tolist(),
                     "initial_normalized_update": base_y.detach().cpu().numpy().tolist()}


def pose_loss(y, baseline_pose, ground_truth, device):
    scale = torch.as_tensor(POSE_SCALE, dtype=torch.float64, device=device)
    delta = y * scale
    from pixloc.pixlib.geometry.optimization import so3exp_map
    rotation = so3exp_map(delta[3:]) @ torch.as_tensor(baseline_pose[:3, :3],
                                                       dtype=torch.float64, device=device)
    translation = torch.as_tensor(baseline_pose[:3, 3], dtype=torch.float64,
                                  device=device) + delta[:3]
    gt = torch.as_tensor(ground_truth, dtype=torch.float64, device=device)
    relative = rotation @ gt[:3, :3].T
    skew = torch.stack((relative[2, 1] - relative[1, 2],
                        relative[0, 2] - relative[2, 0],
                        relative[1, 0] - relative[0, 1])) * .5
    sine = torch.linalg.vector_norm(skew)
    cosine = ((torch.trace(relative) - 1.) * .5).clamp(-1., 1.)
    angle = torch.atan2(sine, cosine).abs()
    translation_error = torch.linalg.vector_norm(translation - gt[:3, 3]) / .1
    rotation_error = angle / math.radians(1.)
    return translation_error + rotation_error, translation_error.detach(), rotation_error.detach()


def fit_surface(anchor, reference_pixel, reference_pose, ref_row, ref_view,
                map_tree, map_points, ref_mask_cache):
    neighbor_distances, neighbor_indices = map_tree.query(anchor, k=SURFACE_NEIGHBOR_COUNT)
    if float(np.max(neighbor_distances)) > SURFACE_MAX_NEIGHBOR_DISTANCE_M:
        return None
    neighbors = map_points[np.asarray(neighbor_indices, dtype=np.int64)].astype(np.float64)
    center = neighbors.mean(0)
    _, eigenvalues, axes = np.linalg.svd(neighbors - center, full_matrices=False)
    basis = axes[:2].T
    normal = axes[2]
    distance = (neighbors - center) @ normal
    rms = float(np.sqrt(np.mean(distance ** 2)))
    spread = np.std((neighbors - center) @ basis, axis=0)
    if rms > SURFACE_MAX_PLANE_RMSE_M or spread.min() < .03:
        return None
    covariance = np.cov(neighbors.T)
    eig = np.linalg.eigvalsh(covariance)
    if eig[0] / max(eig.sum(), 1e-12) > .2:
        return None
    projected_support = (neighbors - center) @ basis
    try:
        triangulation = Delaunay(projected_support)
    except QhullError:
        return None
    camera_to_body = np.asarray(ref_view["camera_to_body"], dtype=np.float64)
    camera_to_world = reference_pose @ camera_to_body
    calibration = np.loadtxt(ref_view["calibration"], dtype=np.float64)
    rays_camera = (np.linalg.inv(calibration) @
                   np.concatenate((reference_pixel[None] + PATCH_PIXELS,
                                   np.ones((len(PATCH_PIXELS), 1))), axis=1).T).T
    directions = rays_camera @ camera_to_world[:3, :3].T
    denom = directions @ normal
    numerator = np.dot(center - camera_to_world[:3, 3], normal)
    if np.any(np.abs(denom) < 1e-5):
        return None
    distance_along = numerator / denom
    if np.any(distance_along <= .1):
        return None
    samples = camera_to_world[:3, 3] + directions * distance_along[:, None]
    from_hull = triangulation.find_simplex((samples - center) @ basis) >= 0
    near_distance, _ = map_tree.query(samples, k=1)
    local_extent = np.linalg.norm(samples - anchor[None], axis=1)
    valid = (from_hull & (near_distance <= SURFACE_MAX_SUPPORT_DISTANCE_M) &
             (local_extent <= SURFACE_MAX_PATCH_RADIUS_M))
    ref_mask = ref_mask_cache.get((str(ref_row["frame_id"]), int(ref_view["camera"])))
    if ref_mask is None:
        ref_mask = np.asarray(np.load(ref_view["mask"]), dtype=bool)
        ref_mask_cache[(str(ref_row["frame_id"]), int(ref_view["camera"]))] = ref_mask
    height, width = ref_mask.shape
    pix = reference_pixel[None] + PATCH_PIXELS
    in_image = ((pix[:, 0] >= 1) & (pix[:, 0] < width - 2) &
                (pix[:, 1] >= 1) & (pix[:, 1] < height - 2))
    ix = np.clip(np.rint(pix[:, 0]).astype(np.int64), 0, width - 1)
    iy = np.clip(np.rint(pix[:, 1]).astype(np.int64), 0, height - 1)
    valid &= in_image & ref_mask[iy, ix]
    if int(valid.sum()) < 5:
        return None
    return samples.astype(np.float32), valid, rms, float(np.max(local_extent)), int(len(neighbors))


def build_surface_map(observation_path, map_path, lidar_cache, rows_by_frame,
                      output_path):
    observation_path, map_path, output_path = map(Path, (observation_path, map_path, output_path))
    observation_sha, map_sha = sha256_file(observation_path), sha256_file(map_path)
    if output_path.is_file():
        try:
            with np.load(output_path, allow_pickle=False) as existing:
                if (int(existing["surface_protocol_version"].item()) == SURFACE_PROTOCOL_VERSION and
                        str(existing["observation_sha256"].item()) == observation_sha and
                        str(existing["map_sha256"].item()) == map_sha):
                    return {"path": str(output_path), "sha256": sha256_file(output_path),
                            "source_observations": int(existing["source_observation_count"]),
                            "surface_patches": int(len(existing["surface_points"])),
                            "surface_samples": int(existing["surface_valid"].sum()),
                            "reused": True}
        except Exception:
            pass
    with np.load(observation_path, allow_pickle=False) as source:
        if not bool(source["geometry_only"].item()):
            raise RuntimeError("visual map observations do not certify train-map-only geometry")
        map_ids = np.asarray(source["map_ids"], dtype=np.int64)
        anchors = np.asarray(source["world_xyz"], dtype=np.float64)
        ref_pixels = np.asarray(source["ref_uv"], dtype=np.float64)
        ref_frames = np.asarray(source["frame_ids"]).astype(str)
        ref_cameras = np.asarray(source["camera_ids"], dtype=np.int64)
    with np.load(map_path, allow_pickle=False) as source:
        map_points = np.asarray(source["points"], dtype=np.float32)
    tree = cKDTree(map_points)
    pose_cache, mask_cache = {}, {}
    accepted = []
    for index, (map_id, anchor, uv, frame_id, camera) in enumerate(
            zip(map_ids, anchors, ref_pixels, ref_frames, ref_cameras)):
        ref_row = rows_by_frame.get(str(frame_id))
        if ref_row is None or ref_row["split"] != "train":
            raise RuntimeError("surface reference observations must resolve to train frames")
        key = str(frame_id)
        if key not in pose_cache:
            with np.load(Path(lidar_cache) / (key + ".npz"), allow_pickle=False) as pose_data:
                pose_cache[key] = np.asarray(pose_data["GT"], dtype=np.float64)
        ref_view = next(view for view in ref_row["views"] if int(view["camera"]) == int(camera))
        fitted = fit_surface(anchor, uv, pose_cache[key], ref_row, ref_view,
                             tree, map_points, mask_cache)
        if fitted is None:
            continue
        points, valid, plane_rmse, radius, neighbors = fitted
        camera_to_world = pose_cache[key] @ np.asarray(ref_view["camera_to_body"], dtype=np.float64)
        accepted.append((index, int(map_id), anchor.astype(np.float32), points, valid,
                         frame_id, int(camera), uv.astype(np.float32),
                         camera_to_world[:3, 3].astype(np.float32),
                         [plane_rmse, radius, neighbors]))
        if (index + 1) % 5000 == 0:
            print(json.dumps({"stage": "surface_map", "observations_seen": index + 1,
                              "accepted": len(accepted)}), flush=True)
    if not accepted:
        raise RuntimeError("the train-only visual map produced no finite LiDAR surface patches")
    ids = np.asarray([value[0] for value in accepted], dtype=np.int64)
    surface_map_ids = np.asarray([value[1] for value in accepted], dtype=np.int64)
    surface_points = np.stack([value[3] for value in accepted]).astype(np.float32)
    surface_valid = np.stack([value[4] for value in accepted]).astype(bool)
    group_ids = np.unique(surface_map_ids, return_inverse=True)[1].astype(np.int32)
    data = {
        "surface_protocol_version": np.asarray(SURFACE_PROTOCOL_VERSION, dtype=np.int32),
        "observation_sha256": np.asarray(observation_sha),
        "map_sha256": np.asarray(map_sha),
        "source_observation_count": np.asarray(len(map_ids), dtype=np.int32),
        "source_indices": ids,
        "map_ids": surface_map_ids,
        "group_ids": group_ids,
        "anchors": np.stack([value[2] for value in accepted]).astype(np.float32),
        "surface_points": surface_points,
        "surface_valid": surface_valid,
        "reference_frames": np.asarray([value[5] for value in accepted]),
        "reference_cameras": np.asarray([value[6] for value in accepted], dtype=np.int8),
        "reference_pixels": np.stack([value[7] for value in accepted]).astype(np.float32),
        "surface_pixels": (np.stack([value[7] for value in accepted])[:, None, :] +
                            PATCH_PIXELS[None]).astype(np.float32),
        "reference_camera_centers": np.stack([value[8] for value in accepted]).astype(np.float32),
        "surface_quality": np.asarray([value[9] for value in accepted], dtype=np.float32),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)
    result = {"path": str(output_path), "sha256": sha256_file(output_path),
              "source_observations": len(map_ids), "surface_patches": len(accepted),
              "surface_samples": int(surface_valid.sum()), "reused": False}
    print(json.dumps({"stage": "surface_map_complete", **result}), flush=True)
    return result


def load_surface_map(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def project_numpy(points, pose, camera_to_body, calibration):
    camera_pose = pose @ camera_to_body
    camera_points = (points - camera_pose[:3, 3]) @ camera_pose[:3, :3]
    homogeneous = camera_points @ calibration.T
    depth = camera_points[:, 2]
    pixels = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:3], 1e-10)
    return pixels, depth, camera_pose[:3, 3]


def select_frame_surfaces(surface_map, row, baseline_pose, lidar_information,
                          peak_pose=None, input_sha256=""):
    anchors = np.asarray(surface_map["anchors"], dtype=np.float64)
    group_ids = np.asarray(surface_map["group_ids"], dtype=np.int32)
    ref_centers = np.asarray(surface_map["reference_camera_centers"], dtype=np.float64)
    ref_directions = anchors - ref_centers
    ref_directions /= np.maximum(np.linalg.norm(ref_directions, axis=1, keepdims=True), 1e-12)
    selected_patches, selected_cameras = [], []
    camera_stats = []
    views = sorted(row["views"], key=lambda view: int(view["camera"]))
    num_groups = int(group_ids.max()) + 1 if len(group_ids) else 0
    for view in views:
        camera = int(view["camera"])
        camera_to_body = np.asarray(view["camera_to_body"], dtype=np.float64)
        calibration = np.loadtxt(view["calibration"], dtype=np.float64)
        _, _, query_center = project_numpy(anchors[:1], baseline_pose, camera_to_body, calibration)
        query_directions = anchors - query_center
        query_directions /= np.maximum(np.linalg.norm(query_directions, axis=1, keepdims=True), 1e-12)
        similarity = np.sum(ref_directions * query_directions, axis=1)
        order = np.lexsort((np.arange(len(group_ids)), -similarity, group_ids))
        sorted_groups = group_ids[order]
        first = np.r_[True, sorted_groups[1:] != sorted_groups[:-1]]
        candidates = order[first]
        projected, depth, _ = project_numpy(anchors[candidates], baseline_pose,
                                            camera_to_body, calibration)
        mask = np.asarray(np.load(view["mask"]), dtype=bool)
        height, width = mask.shape
        ix = np.floor(projected[:, 0] + .5).astype(np.int64)
        iy = np.floor(projected[:, 1] + .5).astype(np.int64)
        valid = ((depth > .2) & (ix >= 1) & (ix < width - 1) &
                 (iy >= 1) & (iy < height - 1))
        valid_indices = np.where(valid)[0]
        valid[valid_indices] &= mask[iy[valid_indices], ix[valid_indices]]
        candidates, projected, depth, ix, iy = (candidates[valid], projected[valid],
                                                depth[valid], ix[valid], iy[valid])
        visible_count = len(candidates)
        if visible_count:
            flat = iy * width + ix
            zbuffer = np.full(height * width, np.inf, dtype=np.float64)
            np.minimum.at(zbuffer, flat, depth)
            front = depth <= zbuffer[flat] + .4
            candidates, projected, depth, ix, iy = (candidates[front], projected[front],
                                                    depth[front], ix[front], iy[front])
        if len(candidates):
            grid_x = np.clip((projected[:, 0] / width * 18).astype(np.int32), 0, 17)
            grid_y = np.clip((projected[:, 1] / height * 13).astype(np.int32), 0, 12)
            cell = grid_y * 18 + grid_x
            choice_order = np.lexsort((depth, cell))
            cell_ordered = cell[choice_order]
            keep = np.r_[True, cell_ordered[1:] != cell_ordered[:-1]]
            candidates = candidates[choice_order[keep]]
        selected_patches.extend(candidates.tolist())
        selected_cameras.extend([camera] * len(candidates))
        camera_stats.append({"camera": camera, "map_ids_in_frustum": int(visible_count),
                             "selected_cells": int(len(candidates))})
    indices = np.asarray(selected_patches, dtype=np.int64)
    camera_ids = np.asarray(selected_cameras, dtype=np.int64)
    return {
        "frame_id": str(row["frame_id"]),
        "source_indices": indices,
        "surface_points": np.asarray(surface_map["surface_points"])[indices],
        "surface_valid": np.asarray(surface_map["surface_valid"])[indices],
        "anchors": anchors[indices].astype(np.float32),
        "map_ids": np.asarray(surface_map["map_ids"])[indices],
        "reference_frames": np.asarray(surface_map["reference_frames"])[indices],
        "reference_cameras": np.asarray(surface_map["reference_cameras"])[indices],
        "surface_pixels": np.asarray(surface_map["surface_pixels"])[indices],
        "cameras": camera_ids,
        "baseline_pose": np.asarray(baseline_pose, dtype=np.float64),
        "lidar_information": np.asarray(lidar_information, dtype=np.float64),
        "peak_pose": np.asarray(baseline_pose if peak_pose is None else peak_pose, dtype=np.float64),
        "baseline_pose_tensor": torch.as_tensor(baseline_pose, dtype=torch.float64),
        "input_sha256": input_sha256,
        "input_count": int(num_groups),
        "camera_patch_counts": camera_stats,
    }


def load_reference_feature_cache(path, expected_surface_sha, device):
    with np.load(path, allow_pickle=False) as source:
        if str(source["surface_sha256"].item()) != expected_surface_sha:
            raise RuntimeError("PixLoc reference feature cache and surface map differ")
        return {key: torch.as_tensor(source[key], device=device) for key in source.files
                if key.startswith("raw_") or key.startswith("conf_")}


def materialize_global_reference_features(model, surface_map_path, cache_path,
                                          rows_by_frame, device, pixloc_sha):
    surface_map_path, cache_path = Path(surface_map_path), Path(cache_path)
    surface_sha = sha256_file(surface_map_path)
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as existing:
                if (str(existing["surface_sha256"].item()) == surface_sha and
                        str(existing["feature_sha256"].item()) == pixloc_sha):
                    return {"path": str(cache_path), "sha256": sha256_file(cache_path),
                            "reused": True}
        except Exception:
            pass
    frame = load_surface_map(surface_map_path)
    frame["surface_valid"] = np.asarray(frame["surface_valid"], dtype=bool)
    raw, confidence = sample_reference_pyramid(model, None, frame, rows_by_frame, device)
    arrays = {f"raw_{i}": value.detach().cpu().numpy().astype(np.float16)
              for i, value in enumerate(raw)}
    arrays.update({f"conf_{i}": value.detach().cpu().numpy().astype(np.float16)
                   for i, value in enumerate(confidence)})
    arrays["surface_sha256"] = np.asarray(surface_sha)
    arrays["feature_sha256"] = np.asarray(pixloc_sha)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    result = {"path": str(cache_path), "sha256": sha256_file(cache_path),
              "reused": False, "surface_patches": len(frame["surface_valid"])}
    print(json.dumps({"stage": "global_reference_features", **result}), flush=True)
    return result


def load_surface_frame(path, device):
    with np.load(path, allow_pickle=False) as src:
        frame = {key: src[key] for key in src.files}
    frame["baseline_pose_tensor"] = torch.as_tensor(frame["baseline_pose"], dtype=torch.float64,
                                                     device=device)
    return frame


def estimate_visual_scales(model, training_frames, reference_cache, device):
    ratios = [[] for _ in SCALES]
    frame_ratios = []
    for frame, row in training_frames:
        frame["baseline_pose_tensor"] = torch.as_tensor(frame["baseline_pose"],
                                                        dtype=torch.float64, device=device)
        if len(frame["cameras"]) < 4:
            continue
        with torch.no_grad():
            bundle = build_feature_bundle_from_cache(model, frame, reference_cache,
                                                     row, device)
        lidar = torch.as_tensor(frame["lidar_information"], dtype=torch.float64, device=device)
        scale = torch.as_tensor(POSE_SCALE, dtype=torch.float64, device=device)
        lidar_h = scale[:, None] * lidar * scale[None, :]
        local = []
        for level in range(3):
            visual_h, _, _, valid_counts = normal_equations(
                frame, row, bundle, torch.zeros(6, dtype=torch.float64, device=device),
                level, 1., device, detach_jacobian=True)
            valid_n = int((valid_counts > 0).sum().item())
            ratio = float((torch.trace(lidar_h) /
                           torch.trace(visual_h).clamp_min(1e-12)).detach().cpu())
            ratios[level].append(ratio)
            local.append({"valid_patches": valid_n, "unscaled_lidar_visual_trace_ratio": ratio})
        frame_ratios.append({"frame_id": str(frame["frame_id"]), "scales": local})
        print(json.dumps({"stage": "weight_calibration", "frame": str(frame["frame_id"]),
                          "ratios": [row["unscaled_lidar_visual_trace_ratio"] for row in local]}), flush=True)
    result = [float(np.clip(np.median(values), 1e-6, 1e6)) if values else 1.
              for values in ratios]
    return result, frame_ratios


def build_feature_bundle_from_cache(model, frame, reference_cache, row, device):
    indices = torch.as_tensor(frame["source_indices"], dtype=torch.long, device=device)
    raw_refs = [reference_cache[f"raw_{i}"][indices].to(torch.float32) for i in range(3)]
    ref_conf = [reference_cache[f"conf_{i}"][indices].to(torch.float32) for i in range(3)]
    reference_features = [F.normalize(F.linear(raw, adapter[0].weight[:, :, 0, 0],
                                               adapter[0].bias), dim=-1)
                          for raw, adapter in zip(raw_refs, model.adaptation)]
    query = {}
    views = {int(view["camera"]): view for view in row["views"]}
    for camera in sorted(set(frame["cameras"].tolist())):
        view = views[int(camera)]
        with torch.no_grad():
            pyramid = pre_features(model, load_rgb(view["image"], device))
            conf = [torch.sigmoid(-layer(value))[0, 0]
                    for layer, value in zip(model.uncertainty, pyramid)]
        maps = [F.normalize(adapter(feature), dim=1)[0]
                for adapter, feature in zip(model.adaptation, pyramid)]
        query[int(camera)] = {
            "maps": maps,
            "confidence": conf,
            "mask": camera_mask_tensor(view, device),
            "camera_to_body": torch.as_tensor(view["camera_to_body"], dtype=torch.float64,
                                                device=device),
            "calibration": torch.as_tensor(np.loadtxt(view["calibration"]), dtype=torch.float64,
                                             device=device),
        }
    return {"query": query, "reference_features": reference_features,
            "reference_confidence": ref_conf}


def load_training_record(surface_path, lidar_cache, input_path):
    with np.load(surface_path, allow_pickle=False) as source:
        frame = {key: source[key] for key in source.files}
    with np.load(lidar_cache, allow_pickle=False) as source:
        gt = np.asarray(source["GT"], dtype=np.float64)
    frame["ground_truth"] = gt
    frame["input_sha256"] = str(frame["input_sha256"].item())
    frame["baseline_pose_tensor"] = torch.as_tensor(frame["baseline_pose"], dtype=torch.float64)
    return frame


def train(args):
    torch.set_num_threads(8)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device)
    manifest_path = Path(args.manifest)
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_by_frame = {str(row["frame_id"]): row for row in rows}
    manifest_sha = sha256_file(manifest_path)
    train_dir = Path(args.train_inputs)
    train_rows = [row for row in rows if row["split"] == "train" and
                  (train_dir / (str(row["frame_id"]) + ".npz")).is_file()]
    if len(train_rows) != 47:
        raise RuntimeError(f"expected the frozen 47 training frames, found {len(train_rows)}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = Path(args.reference_map)
    observation_path = Path(args.observation_map)
    checkpoint_path = Path(args.pixloc_checkpoint)
    model, pixloc_checkpoint = load_pixloc(checkpoint_path, device)
    pixloc_sha = sha256_file(checkpoint_path)
    surface_map_path = output_dir / "surface_map.npz"
    surface_map_summary = build_surface_map(observation_path, map_path, args.lidar_cache,
                                            rows_by_frame, surface_map_path)
    surface_map = load_surface_map(surface_map_path)
    ref_cache_path = output_dir / "surface_map_pixloc_refs.npz"
    ref_cache_summary = materialize_global_reference_features(
        model, surface_map_path, ref_cache_path, rows_by_frame, device, pixloc_sha)
    ref_cache = load_reference_feature_cache(ref_cache_path,
                                             sha256_file(surface_map_path), device)
    training_frames, training_inputs = [], []
    for row in train_rows:
        frame_id = str(row["frame_id"])
        path = train_dir / (frame_id + ".npz")
        with np.load(path, allow_pickle=False) as source:
            baseline = np.asarray(source["baseline_pose"], dtype=np.float64)
            lidar_information = np.asarray(source["lidar_information"], dtype=np.float64)
        input_sha = sha256_file(path)
        frame = select_frame_surfaces(surface_map, row, baseline, lidar_information,
                                      input_sha256=input_sha)
        frame["baseline_pose_tensor"] = torch.as_tensor(baseline, dtype=torch.float64, device=device)
        if len(frame["cameras"]) < 4:
            raise RuntimeError("fewer than four finite map surfaces in training frame " + frame_id)
        with np.load(Path(args.lidar_cache) / (frame_id + ".npz"), allow_pickle=False) as lidar_source:
            ground_truth = np.asarray(lidar_source["GT"], dtype=np.float64)
        training_frames.append((frame, row, ground_truth))
        training_inputs.append({"frame_id": frame_id, "baseline_input_sha256": input_sha,
                                "surface_patches": int(len(frame["cameras"])),
                                "surface_samples": int(frame["surface_valid"].sum()),
                                "camera_patch_counts": frame["camera_patch_counts"]})
    with torch.no_grad():
        damping_values = []
        for scale_index in range(3):
            values = pixloc_checkpoint["model"][f"optimizer.{scale_index}.dampingnet.const"]
            damping_values.append((10. ** (-6. + values.sigmoid() * 11.)).clamp(1e-4, 1.).numpy())
    damping_values = [value.astype(np.float64) for value in damping_values]
    visual_scales, weight_rows = estimate_visual_scales(
        model, [(frame, row) for frame, row, _ in training_frames], ref_cache, device)
    optimizer = torch.optim.Adam([parameter for layer in model.adaptation
                                  for parameter in layer.parameters()], lr=5e-6)
    history = []
    for epoch in range(1, args.epochs + 1):
        order = np.random.default_rng(SEED + epoch).permutation(len(train_rows))
        frame_losses, pose_terms, used = [], [], 0
        for position in order:
            frame, row, gt = training_frames[int(position)]
            frame_id = str(frame["frame_id"])
            bundle = build_feature_bundle_from_cache(model, frame, ref_cache, row, device)
            perturb_rng = np.random.default_rng(SEED + epoch * 1009 + int(frame_id[-4:]))
            initial_y = torch.as_tensor(perturb_rng.uniform(-.4, .4, size=6),
                                        dtype=torch.float64, device=device)
            initial_y = initial_y.detach()
            y, _, _ = solve_frame(frame, row, bundle, visual_scales, damping_values, device,
                                  TRAIN_STEPS_PER_SCALE, initial_y, training=True)
            loss, trans_component, rot_component = pose_loss(y, frame["baseline_pose"], gt, device)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite pose loss in training frame " + frame_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for layer in model.adaptation
                                            for parameter in layer.parameters()], 1.)
            optimizer.step()
            frame_losses.append(float(loss.detach().cpu()))
            pose_terms.append([float(trans_component.cpu()), float(rot_component.cpu())])
            used += 1
            del bundle, y, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
        record = {"epoch": epoch, "train_frames": used,
                  "mean_normalized_pose_loss": float(np.mean(frame_losses)),
                  "mean_normalized_translation_error": float(np.mean(np.asarray(pose_terms)[:, 0])),
                  "mean_normalized_rotation_error": float(np.mean(np.asarray(pose_terms)[:, 1]))}
        history.append(record)
        checkpoint = {
            "adapter_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()
                                   if key.startswith("adaptation.")},
            "config": {"method": "LiDAR-bounded 3D surface reprojection with PixLoc multi-scale direct feature alignment",
                       "epochs": args.epochs, "fixed_final_epoch": True,
                       "optimizer": "Adam", "learning_rate": 5e-6,
                       "gradient_clip": 1., "random_initial_pose_box_normalized": .4,
                       "training_steps_per_scale": TRAIN_STEPS_PER_SCALE,
                       "inference_steps_per_scale": INFERENCE_STEPS_PER_SCALE,
                       "validation_gt_used_for_training": False,
                       "checkpoint_selection": "fixed final epoch; no development split selection",
                       "pose_scale": POSE_SCALE.tolist(), "visual_scales_coarse_to_fine": visual_scales,
                       "patch_pixels": PATCH_PIXELS.tolist(),
                       "surface_neighbor_count": SURFACE_NEIGHBOR_COUNT,
                       "surface_max_neighbor_distance_m": SURFACE_MAX_NEIGHBOR_DISTANCE_M,
                       "surface_finite_radius_m": SURFACE_MAX_PATCH_RADIUS_M,
                       "surface_nearest_map_support_m": SURFACE_MAX_SUPPORT_DISTANCE_M,
                       "pixloc_original_epoch": int(pixloc_checkpoint["epoch"])},
            "manifest_sha256": manifest_sha,
            "pixloc_checkpoint_sha256": pixloc_sha,
            "training_inputs": training_inputs,
            "weight_calibration": weight_rows,
            "damping": [value.tolist() for value in damping_values],
            "history": history,
        }
        epoch_path = output_dir / "training_state.pt"
        epoch_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, epoch_path)
        print(json.dumps({"stage": "train", **record,
                          "checkpoint_sha256": sha256_file(epoch_path)}), flush=True)
    final_path = output_dir / "pixloc_surface_adapters.pt"
    checkpoint["config"].update({"training_frames": len(train_rows),
                                 "training_surface_patches": int(sum(row["surface_patches"]
                                                                      for row in training_inputs)),
                                 "training_surface_samples": int(sum(row["surface_samples"]
                                                                      for row in training_inputs)),
                                 "surface_map_sha256": sha256_file(map_path),
                                 "surface_observation_sha256": sha256_file(observation_path),
                                 "surface_map_cache_sha256": sha256_file(surface_map_path),
                                 "reference_feature_cache_sha256": sha256_file(ref_cache_path),
                                 "initial_checkpoint_sha256": pixloc_sha,
                                 "manifest_path_sha256": manifest_sha})
    torch.save(checkpoint, final_path)
    report = {
        "protocol": "train-only PixLoc multi-scale feature adaptation with LiDAR-fitted finite patches from the complete train visual map and a single shared six-DoF pose",
        "training_split": "manifest split=train only; 47 train rows; query GT loaded only by pose loss",
        "validation_gt_used_for_training": False,
        "training_frames": len(train_rows),
        "surface_materialization": {"map_sha256": sha256_file(map_path),
                                    "visual_observation_map_sha256": sha256_file(observation_path),
                                    "surface_map_cache_sha256": sha256_file(surface_map_path),
                                    "surface_map_cache": surface_map_summary,
                                    "reference_feature_cache": ref_cache_summary,
                                    "neighbor_count": SURFACE_NEIGHBOR_COUNT,
                                    "max_neighbor_distance_m": SURFACE_MAX_NEIGHBOR_DISTANCE_M,
                                    "finite_patch_radius_m": SURFACE_MAX_PATCH_RADIUS_M,
                                    "nearest_map_support_m": SURFACE_MAX_SUPPORT_DISTANCE_M,
                                    "max_plane_rmse_m": SURFACE_MAX_PLANE_RMSE_M,
                                    "patch_pixel_offsets": PATCH_PIXELS.tolist(),
                                    "training_patch_total": checkpoint["config"]["training_surface_patches"],
                                    "training_sample_total": checkpoint["config"]["training_surface_samples"],
                                    "per_frame": training_inputs},
        "pretrained_pixloc": {"source": "official CVG PixLoc MegaDepth checkpoint",
                              "official_repository_commit": "6f7a943afc34183654754f9c4e90672e491a629b",
                              "checkpoint_sha256": pixloc_sha,
                              "original_epoch": int(pixloc_checkpoint["epoch"]),
                              "trainable": "the three official feature adaptation layers only; VGG19 encoder, decoder and confidence layers frozen"},
        "optimizer": {"name": "Adam", "lr": 5e-6, "epochs": args.epochs,
                      "fixed_final_epoch": True, "clip_grad_norm": 1.},
        "pose_alignment": {"multiscale_coarse_to_fine": [16, 4, 1],
                           "training_steps_per_scale": TRAIN_STEPS_PER_SCALE,
                           "inference_steps_per_scale": INFERENCE_STEPS_PER_SCALE,
                           "random_initial_pose_perturbation_normalized": [-.4, .4],
                           "pose_bounds": "±0.1 m and ±1 degree about the LEADER pose",
                           "feature_loss": "PixLoc normalized descriptor residual with official scaled Barron alpha=0, c=0.1 form and frozen PixLoc uncertainty maps",
                           "surface_patch_normalization": "feature samples are averaged per patch so sample count does not multiply patch weight",
                           "lidar_prior": "single existing frame-level quadratic information prior; shared across all cameras and scales",
                           "visual_scales": visual_scales,
                           "learned_pixloc_damping": [value.tolist() for value in damping_values]},
        "history": history,
        "training_inputs": training_inputs,
        "checkpoint_path": str(final_path),
        "checkpoint_sha256": sha256_file(final_path),
    }
    Path(args.output_dir, "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"stage": "training_complete", "report": str(Path(args.output_dir, "training_report.json")),
                      "checkpoint_sha256": report["checkpoint_sha256"]}), flush=True)


def inference(args):
    torch.set_num_threads(8)
    device = torch.device(args.device)
    manifest_path = Path(args.manifest)
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_by_frame = {str(row["frame_id"]): row for row in rows}
    manifest_sha = sha256_file(manifest_path)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["manifest_sha256"] != manifest_sha:
        raise RuntimeError("training checkpoint and inference manifest differ")
    if checkpoint["config"].get("validation_gt_used_for_training") is not False:
        raise RuntimeError("checkpoint does not certify validation GT exclusion")
    pixloc_path = Path(args.pixloc_checkpoint)
    pixloc_sha = sha256_file(pixloc_path)
    if pixloc_sha != checkpoint["pixloc_checkpoint_sha256"]:
        raise RuntimeError("official PixLoc checkpoint SHA-256 mismatch")
    model, _ = load_pixloc(pixloc_path, device)
    state = model.state_dict()
    state.update(checkpoint["adapter_state_dict"])
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    surface_map_path = Path(args.output_dir) / "surface_map.npz"
    if sha256_file(surface_map_path) != checkpoint["config"]["surface_map_cache_sha256"]:
        raise RuntimeError("training and inference surface-map caches differ")
    surface_map = load_surface_map(surface_map_path)
    surface_sha = sha256_file(surface_map_path)
    ref_cache_path = Path(args.output_dir) / "surface_map_pixloc_refs.npz"
    if sha256_file(ref_cache_path) != checkpoint["config"]["reference_feature_cache_sha256"]:
        raise RuntimeError("PixLoc reference feature cache changed since training")
    reference_cache = load_reference_feature_cache(ref_cache_path, surface_sha, device)
    damping = [np.asarray(value, dtype=np.float64) for value in checkpoint["damping"]]
    visual_scales = [float(value) for value in checkpoint["config"]["visual_scales_coarse_to_fine"]]
    peak_path = Path(args.reference_peak_run)
    peak_run = json.loads(peak_path.read_text(encoding="utf-8"))
    peak_records = {str(record["frame_id"]): record for record in peak_run["records"]}
    match_dir = Path(args.validation_match_cache)
    validation_rows = [row for row in rows if row["split"] in ("val", "validation") and
                       (match_dir / (str(row["frame_id"]) + ".npz")).is_file()]
    if len(validation_rows) != 32:
        raise RuntimeError(f"expected 32 validation frames, found {len(validation_rows)}")
    records_out = []
    for row in validation_rows:
        frame_id = str(row["frame_id"])
        peak_record = peak_records.get(frame_id)
        if peak_record is None:
            raise RuntimeError("frozen peak baseline is missing frame " + frame_id)
        match_path = match_dir / (frame_id + ".npz")
        input_sha = sha256_file(match_path)
        if input_sha != peak_record.get("match_cache_sha256"):
            raise RuntimeError("frozen matcher input hash differs for frame " + frame_id)
        baseline = np.asarray(peak_record["baseline_pose"], dtype=np.float64)
        lidar_information = np.asarray(peak_record["lidar_information"], dtype=np.float64)
        frame = select_frame_surfaces(surface_map, row, baseline, lidar_information,
                                      np.asarray(peak_record["peak_pose"], dtype=np.float64),
                                      input_sha)
        frame["baseline_pose_tensor"] = torch.as_tensor(baseline, dtype=torch.float64,
                                                         device=device)
        bundle = build_feature_bundle_from_cache(model, frame, reference_cache, row, device)
        start = time.perf_counter()
        with torch.no_grad():
            y, pose, solver = solve_frame(frame, row, bundle, visual_scales,
                                          damping, device, INFERENCE_STEPS_PER_SCALE)
        elapsed = time.perf_counter() - start
        records_out.append({
            "frame_id": frame_id,
            "baseline_pose": baseline.tolist(),
            "peak_pose": np.asarray(peak_record["peak_pose"], dtype=np.float64).tolist(),
            "surface_direct_pose": pose.tolist(),
            "surface_solver": solver,
            "surface_patches": int(len(frame["surface_points"])),
            "surface_samples": int(np.asarray(frame["surface_valid"]).sum()),
            "surface_candidates_in_fov": int(frame["input_count"]),
            "camera_patch_counts": frame["camera_patch_counts"],
            "match_cache_sha256": input_sha,
            "surface_map_sha256": surface_sha,
            "reference_feature_cache_sha256": sha256_file(ref_cache_path),
            "solver_wall_seconds": elapsed,
        })
        print(json.dumps({"stage": "inference", "frame": frame_id,
                          "surface_patches": len(frame["surface_points"]),
                          "success": solver["success"],
                          "valid_patches": solver.get("valid_patches", 0),
                          "seconds": elapsed}), flush=True)
        del bundle, frame
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(records_out) != 32:
        raise RuntimeError("validation denominator changed during inference")
    output = {
        "protocol": {
            "name": "train-map LiDAR surface patches with pretrained PixLoc multi-scale direct feature alignment",
            "dataset_note": "32 validation frames from one repeatedly used development route; not an independent test sequence",
            "baseline": "same frozen SC2-PCR plus two-stage full-pool-refinement initial pose; compared against frozen peak_then_pose",
            "shared_pose": True,
            "ground_truth_in_runner": False,
            "validation_ground_truth_used_for_training": False,
            "candidate_policy": "one direct shared pose per frame; insufficient visual support minimizes the single LiDAR-prior objective",
            "map_observations": "all supported observations from the train-only visual map; no RoMa peak or correspondence offsets used by the candidate",
            "feature_model": "official PixLoc MegaDepth VGG19 U-Net; encoder, decoder and uncertainty frozen; original multi-scale feature adaptation layers fine-tuned on train split only",
            "scales_coarse_to_fine": [16, 4, 1],
            "feature_loss": "normalized learned feature residual, PixLoc scaled Barron alpha=0 c=0.1 form and frozen uncertainty maps",
            "patch_support": "local PCA plane from 16 nearest train-map LiDAR points, 3x3 reference pixels at offsets -4/0/4, convex-hull support, nearest-map support <=0.25 m, maximum sample radius <=0.5 m",
            "map_selection": "one reference view per map point, selected by baseline-viewpoint angular similarity; z-buffered and spatially sampled to one patch per 18x13 image grid cell per query camera",
            "surface_patch_weighting": "each surface patch normalized by its currently valid sample count",
            "lidar_prior": "single existing frame-level quadratic pose prior included once",
            "reference_pose_source": "training-split LiDAR GT used only to lift known train-map reference pixels onto the map surface",
        },
        "settings": {
            "manifest_sha256": manifest_sha,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "pixloc_checkpoint_sha256": pixloc_sha,
            "visual_map_observation_sha256": checkpoint["config"]["surface_observation_sha256"],
            "reference_map_sha256": sha256_file(args.reference_map),
            "surface_map_cache_sha256": surface_sha,
            "reference_feature_cache_sha256": sha256_file(ref_cache_path),
            "reference_peak_run_sha256": sha256_file(peak_path),
            "expected_frames": 32,
            "actual_frames": len(records_out),
            "visual_scales": visual_scales,
            "damping": checkpoint["damping"],
            "inference_steps_per_scale": INFERENCE_STEPS_PER_SCALE,
            "total_solver_wall_seconds": float(sum(record["solver_wall_seconds"] for record in records_out)),
            "device": args.device,
        },
        "frames": len(records_out),
        "records": records_out,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"stage": "run_frozen", "path": str(output_path),
                      "sha256": sha256_file(output_path), "frames": len(records_out)}), flush=True)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(len(values)), "mean": float(np.mean(values)),
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9))}


def block_interval(values, block_size=4, samples=20000):
    values = np.asarray(values, dtype=np.float64)
    blocks = [np.arange(i, min(i + block_size, len(values)))
              for i in range(0, len(values), block_size)]
    rng = np.random.default_rng(SEED)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        estimates[index] = values[np.concatenate([blocks[i] for i in chosen])].mean()
    return [float(x) for x in np.quantile(estimates, [.025, .975])]


def pose_error(pose, target):
    relative = pose[:3, :3].T @ target[:3, :3]
    angle = math.degrees(math.acos(float(np.clip((np.trace(relative) - 1.) * .5, -1., 1.))))
    return float(np.linalg.norm(pose[:3, 3] - target[:3, 3])), angle


def evaluate(args):
    run_path = Path(args.run)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run["frames"] != 32 or len(run["records"]) != 32:
        raise RuntimeError("frozen run does not contain the complete 32-frame denominator")
    if run["protocol"].get("ground_truth_in_runner") is not False:
        raise RuntimeError("runner did not certify GT exclusion")
    if any("GT" in record or "ground_truth" in record for record in run["records"]):
        raise RuntimeError("frozen prediction record contains a GT field")
    rows = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows_by_frame = {str(row["frame_id"]): row for row in rows}
    methods = {"geometry_only_control": [], "peak_then_pose": [], "pixloc_surface_direct": []}
    failures, per_frame = 0, []
    for record in run["records"]:
        frame_id = str(record["frame_id"])
        if rows_by_frame[frame_id]["split"] not in ("val", "validation"):
            raise RuntimeError("non-validation row in frozen output")
        with np.load(Path(args.lidar_cache) / (frame_id + ".npz"), allow_pickle=False) as src:
            gt = np.asarray(src["GT"], dtype=np.float64)
        baseline = np.asarray(record["baseline_pose"], dtype=np.float64)
        peak = np.asarray(record["peak_pose"], dtype=np.float64)
        candidate = np.asarray(record["surface_direct_pose"], dtype=np.float64)
        geometry_error = pose_error(baseline, gt)
        peak_error = pose_error(peak, gt)
        candidate_error = pose_error(candidate, gt)
        methods["geometry_only_control"].append(geometry_error)
        methods["peak_then_pose"].append(peak_error)
        methods["pixloc_surface_direct"].append(candidate_error)
        solver = record["surface_solver"]
        failures += int(not solver.get("success", False))
        per_frame.append({"frame_id": frame_id,
                          "geometry_mpe_m": geometry_error[0], "geometry_moe_deg": geometry_error[1],
                          "peak_mpe_m": peak_error[0], "peak_moe_deg": peak_error[1],
                          "candidate_mpe_m": candidate_error[0], "candidate_moe_deg": candidate_error[1],
                          "delta_vs_peak_mpe_m": candidate_error[0] - peak_error[0],
                          "delta_vs_peak_moe_deg": candidate_error[1] - peak_error[1],
                          "both_improved_vs_peak": bool(candidate_error[0] < peak_error[0] and
                                                         candidate_error[1] < peak_error[1]),
                          "surface_patches": int(record["surface_patches"]),
                          "solver_success": bool(solver.get("success", False))})
    stats = {name: {"mpe_m": summarize(np.asarray(values)[:, 0]),
                    "moe_deg": summarize(np.asarray(values)[:, 1])}
             for name, values in methods.items()}
    delta_t = np.asarray([item["delta_vs_peak_mpe_m"] for item in per_frame])
    delta_r = np.asarray([item["delta_vs_peak_moe_deg"] for item in per_frame])
    result = {
        "protocol": "separate post-freeze evaluator; validation query GT loaded only after prediction JSON was frozen",
        "dataset_note": "all 32 rows belong to one repeatedly used 2012-02-18 development route; not an independent sequence test",
        "run_sha256": sha256_file(run_path),
        "denominator": 32,
        "metrics": stats,
        "pixloc_surface_direct_vs_peak_then_pose": {
            "mean_delta_mpe_m": float(delta_t.mean()), "mean_delta_moe_deg": float(delta_r.mean()),
            "mpe_95ci_block4": block_interval(delta_t),
            "moe_95ci_block4": block_interval(delta_r),
            "mpe_improved_frames": int((delta_t < 0).sum()),
            "mpe_damaged_frames": int((delta_t > 0).sum()),
            "moe_improved_frames": int((delta_r < 0).sum()),
            "moe_damaged_frames": int((delta_r > 0).sum()),
            "both_improved_frames": int(sum(item["both_improved_vs_peak"] for item in per_frame)),
        },
        "solver_failures_with_finite_pose": failures,
        "patch_coverage": {"mean": float(np.mean([x["surface_patches"] for x in per_frame])),
                           "minimum": int(min(x["surface_patches"] for x in per_frame)),
                           "maximum": int(max(x["surface_patches"] for x in per_frame))},
        "total_solver_wall_seconds": float(run["settings"]["total_solver_wall_seconds"]),
        "interval_note": "descriptive paired 95% bootstrap over blocks of four adjacent frames; the route remains a single development sequence",
        "per_frame": per_frame,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"stage": "evaluation_complete", "path": str(output),
                      "sha256": sha256_file(output), "metrics": stats,
                      "comparison": result["pixloc_surface_direct_vs_peak_then_pose"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--manifest", default="/home/zhang/leader-image-gate-multicamera/all_views.json")
    train_parser.add_argument("--train-inputs", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/work/highres_patch_pose_20260923/train_inputs")
    train_parser.add_argument("--lidar-cache", default="/home/zhang/leader-image-gate/lidar")
    train_parser.add_argument("--reference-map", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER/research/prevoxel_multiview/results/reference_map.npz")
    train_parser.add_argument("--observation-map", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER/research/prevoxel_multiview/results/surface_patch_reference_observations.npz")
    train_parser.add_argument("--pixloc-checkpoint", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/work/pixloc-official/outputs/training/pixloc_megadepth/checkpoint_best.tar")
    train_parser.add_argument("--output-dir", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER-num1/research/results/pixloc_surface_direct_20260924")
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--manifest", default="/home/zhang/leader-image-gate-multicamera/all_views.json")
    run_parser.add_argument("--lidar-cache", default="/home/zhang/leader-image-gate/lidar")
    run_parser.add_argument("--validation-match-cache", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER/research/prevoxel_multiview/results/roma_controlled_validation_matches")
    run_parser.add_argument("--reference-map", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER/research/prevoxel_multiview/results/reference_map.npz")
    run_parser.add_argument("--reference-peak-run", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER-num1/research/results/highres_precision_covariance_20260924/validation_run.json")
    run_parser.add_argument("--pixloc-checkpoint", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/work/pixloc-official/outputs/training/pixloc_megadepth/checkpoint_best.tar")
    run_parser.add_argument("--checkpoint", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER-num1/research/results/pixloc_surface_direct_20260924/pixloc_surface_adapters.pt")
    run_parser.add_argument("--output-dir", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER-num1/research/results/pixloc_surface_direct_20260924")
    run_parser.add_argument("--output", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER-num1/research/results/pixloc_surface_direct_20260924/validation_run.json")
    run_parser.add_argument("--device", default="cuda")
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--manifest", default="/home/zhang/leader-image-gate-multicamera/all_views.json")
    eval_parser.add_argument("--lidar-cache", default="/home/zhang/leader-image-gate/lidar")
    eval_parser.add_argument("--run", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER-num1/research/results/pixloc_surface_direct_20260924/validation_run.json")
    eval_parser.add_argument("--output", default="/mnt/c/Users/zhang/Documents/ChatGPT/LEADER/LEADER-num1/research/results/pixloc_surface_direct_20260924/evaluation.json")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "run":
        inference(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
