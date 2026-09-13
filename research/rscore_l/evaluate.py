import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import poselib
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from scrstudio.data.samplers import PQKNN

from .fusion import fuse, project, spatial_budget
from .prepare import save_json
from .train import make_config


def load_model(root, variant):
    config = make_config(root / 'data', root / 'outputs', 'lidar' if variant == 'reliable' else variant)
    model = config.pipeline.model.setup(metadata={'cluster_centers': torch.zeros(50, 3)}).cuda().eval()
    checkpoint = config.get_base_dir() / 'scrstudio_models/head.pt'
    model.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    return model, checkpoint


def predict(model, features, global_features):
    results = []
    hidden = []
    for global_feature in global_features:
        inputs = torch.cat([global_feature[None].expand(len(features), -1), features], dim=1)
        chunks = []
        hidden_chunks = []
        for chunk in inputs.split(4096):
            with torch.inference_mode(), torch.autocast('cuda'):
                out = model({'features': chunk})
            chunks.append(torch.stack([out['sc0'].float(), out['sc'].float()], dim=1))
            hidden_chunks.append(out['features'].float())
        results.append(torch.cat(chunks))
        hidden.append(torch.cat(hidden_chunks))
    return torch.stack(results), torch.stack(hidden)


def export(root, variant, split):
    data = root / 'data'
    rows = json.loads((data / 'manifest.json').read_text())
    model, checkpoint = load_model(root, variant)
    encoding = 'lidar_n2c.pt' if variant == 'reliable' or variant.startswith('lidar') else 'pose_n2c.pt'
    global_features = torch.load(data / 'train' / encoding, weights_only=True)['model.embedding.weight'].cuda().float()
    queries = np.load(data / split / 'netvlad_feats.npy').astype(np.float32)
    with (data / 'train/netvlad_feats_pq.pkl').open('rb') as file:
        pq, codes = pickle.load(file)
    retriever = PQKNN(pq, codes, n_neighbors=10)
    meta = json.loads((data / 'scene_meta.json').read_text())
    destination = root / 'exports' / variant / split
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = [checkpoint, data / 'train' / encoding, data / 'proc/pcad3LB_128.pth', data / split / 'netvlad_feats.npy', data / 'train/netvlad_feats.npy',
        data / 'train/netvlad_feats_pq.pkl', data / 'scene_meta.json']
    if variant == 'reliable':
        artifacts.append(root / 'reliability/head.pt')
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    reliability = None
    database = None
    if variant == 'reliable':
        from .reliability import ReliabilityHead, reliability_inputs
        saved = torch.load(root / 'reliability/head.pt', weights_only=True)
        reliability = ReliabilityHead().cuda().eval()
        reliability.load_state_dict(saved['model'])
        database = np.load(data / 'train/netvlad_feats.npy').astype(np.float32)
    for i, row in enumerate(tqdm(rows[split], desc='Export ' + variant + ' ' + split)):
        target = destination / (row['frame_id'] + '.npz')
        feature_path = data / 'proc' / ('features_' + split) / target.name
        feature_hash = hashlib.sha256(feature_path.read_bytes()).hexdigest()
        if target.exists():
            with np.load(target) as existing:
                if str(existing['model_sha256']) == digest and 'feature_sha256' in existing and str(existing['feature_sha256']) == feature_hash:
                    continue
        cached = np.load(feature_path)
        features = torch.from_numpy(cached['features'].astype(np.float32)).cuda()
        torch.cuda.synchronize()
        started = time.perf_counter()
        indices = retriever.kneighbors(queries[i])
        coordinates, hidden = predict(model, features, global_features[indices])
        if reliability is None:
            probability = torch.full(coordinates.shape[:2], .5, device='cuda')
        else:
            detector = torch.from_numpy(cached['scores']).cuda()
            similarity = torch.from_numpy(queries[i] @ database[indices.cpu().numpy()].T).cuda()
            with torch.inference_mode():
                inputs = reliability_inputs(hidden, coordinates, detector, similarity)
                probability = torch.sigmoid(reliability(inputs) / saved['temperature'])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        np.savez_compressed(target, uv=cached['uv'], xyz=coordinates[:, :, 1].cpu().numpy(), sc0=coordinates[:, :, 0].cpu().numpy(),
            reliability=probability.cpu().numpy(), K=cached['K'], image_size_hw=cached['image_size_hw'],
            T_BC=np.array(meta['T_BC_camera_to_body']), hypothesis_train_ids=indices.cpu().numpy(), frame_id=row['frame_id'],
            camera_id=row['camera_id'], image_timestamp=row['image_timestamp'], world_frame='NCLT_world',
            pixel_convention='DeDoDe continuous pixel centers; K scaled to input raster', model_version=variant,
            model_sha256=digest, feature_sha256=feature_hash, regression_retrieval_seconds=elapsed)
    save_json(destination / 'manifest.json', dict(frames=len(rows[split]), hypotheses=10, keypoints=5000,
        model_sha256=digest, artifact_sha256=hashes, inference_reads_reference_poses=False, timing_excludes_cached_encoder_and_netvlad=True))


def pose_error(T, reference):
    translation = np.linalg.norm(T[:3, 3] - reference[:3, 3])
    rotation = np.degrees(Rotation.from_matrix(T[:3, :3] @ reference[:3, :3].T).magnitude())
    return [float(translation), float(rotation)]


def summary(errors):
    errors = np.asarray(errors)
    result = dict(frames=len(errors), mean=errors.mean(0).tolist(), median=np.median(errors, axis=0).tolist(),
        p95=np.percentile(errors, 95, axis=0).tolist(), p99=np.percentile(errors, 99, axis=0).tolist())
    for m, deg in ((.25, 2), (.5, 1), (1, 2), (1, 5), (5, 10)):
        result[f'success_{m}m_{deg}deg'] = int(((errors[:, 0] < m) & (errors[:, 1] < deg)).sum())
    return result


def visual_pose(correspondence, budget=0):
    K, size = correspondence['K'], correspondence['image_size_hw']
    camera = dict(model='PINHOLE', width=int(size[1]), height=int(size[0]), params=[float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])])
    xyz = correspondence['xyz']
    if xyz.ndim == 2:
        xyz = xyz[None]
    indices = spatial_budget(correspondence['uv'], size, budget) if budget else np.arange(len(correspondence['uv']))
    best = None
    best_count = -1
    for points in xyz:
        pose, info = poselib.estimate_absolute_pose(correspondence['uv'][indices].astype(float), points[indices].astype(float), camera,
            dict(max_reproj_error=10., max_iterations=10000, seed=2089), {})
        count = info['num_inliers']
        if count > best_count:
            T_CW = np.eye(4)
            T_CW[:3, :3], T_CW[:3, 3] = pose.R, pose.t
            best = np.linalg.inv(T_CW)
            best_count = count
    return best, best_count


def evaluate(root, bundle, variant, split):
    data = root / 'data'
    rows = json.loads((data / 'manifest.json').read_text())[split]
    reference = np.load(data / split / 'poses.npy')
    E = np.asarray(json.loads((data / 'scene_meta.json').read_text())['T_BC_camera_to_body'])
    destination = root / 'evaluation' / variant / split
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    test_metadata = {row['image']: row for row in json.loads((bundle / 'data/test_rows.json').read_text())} if split == 'test' else {}
    for i, row in enumerate(tqdm(rows, desc='Evaluate ' + variant + ' ' + split)):
        if variant == 'glace':
            cache = bundle / 'cache' / ('validation_selected' if split == 'val' else 'selected') / 'coordinates'
            camera = dict(np.load(cache / (row['frame_id'] + '.npz')))
            if 'image_size_hw' not in camera:
                original = np.load(data / 'proc' / ('features_' + split) / (row['frame_id'] + '.npz'))
                stored_mask = np.load(data / split / 'valid_mask.npy')
                original_K = np.load(data / split / 'calibration.npy')[i]
                scale_x, scale_y = camera['K'][0, 0] / original_K[0, 0], camera['K'][1, 1] / original_K[1, 1]
                camera['image_size_hw'] = np.rint(np.array(stored_mask.shape) * [scale_y, scale_x]).astype(int)
            camera['reliability'] = np.full(camera['xyz'].shape[:-1], .5)
            camera.pop('GT', None)
        else:
            camera = dict(np.load(root / 'exports' / variant / split / (row['frame_id'] + '.npz')))
        started = time.perf_counter()
        estimated, count = visual_pose(camera)
        budget_pose, budget_count = visual_pose(camera, budget=256)
        record = dict(frame_id=row['frame_id'], session_id=row['session_id'], camera_error=pose_error(estimated, reference[i]),
            camera_error_256=pose_error(budget_pose, reference[i]), pnp_inliers_256=budget_count,
            visual_body_error=pose_error(estimated @ np.linalg.inv(E), reference[i] @ np.linalg.inv(E)), pnp_inliers=count,
            pnp_seconds=time.perf_counter() - started)
        grouped = camera['xyz'] if camera['xyz'].ndim == 3 else camera['xyz'][None]
        direction = []
        for points in grouped:
            predicted_uv, depth = project(reference[i] @ np.linalg.inv(E), points, camera['K'], E)
            error = np.linalg.norm(predicted_uv - camera['uv'], axis=1)
            valid = (depth > 0) & np.isfinite(error)
            direction.append(dict(inliers_10px=int(((error < 10) & valid).sum()), total=len(points)))
        record['direction_by_hypothesis'] = direction
        if split == 'test':
            record.update({key: test_metadata[row['frame_id']][key] for key in ('in_orientation_support', 'previous_probe')})
            pool = dict(np.load(bundle / 'cache/lidar_pools' / (row['frame_id'] + '.npz')))
            inference_pool = {k: v for k, v in pool.items() if k != 'GT'}
            pose, diagnostic = fuse(inference_pool, camera, E)
            control_folder = root / 'controls/lidar_only'
            control_folder.mkdir(parents=True, exist_ok=True)
            control_path = control_folder / (row['frame_id'] + '.npy')
            if control_path.exists():
                control = np.load(control_path)
            else:
                control, _ = fuse(inference_pool, camera, E, enable_visual=False)
                np.save(control_path, control)
            record.update(baseline_error=pose_error(pool['v1_two_stage'], pool['GT']), fused_error=pose_error(pose, pool['GT']), fusion=diagnostic)
            record['lidar_only_refined_error'] = pose_error(control, pool['GT'])
            pool_errors = np.array([pose_error(T, pool['GT']) for T in np.concatenate([pool['v1_two_stage'][None], pool['candidate_T_WB']]) if np.isfinite(T).all()])
            record['candidate_pool_has_1m_2deg'] = bool(((pool_errors[:, 0] < 1) & (pool_errors[:, 1] < 2)).any())
        records.append(record)
        save_json(destination / 'progress.json', dict(completed=len(records), total=len(rows)))
    save_json(destination / 'records.json', records)
    report = dict(camera=summary([r['camera_error'] for r in records]), camera_256=summary([r['camera_error_256'] for r in records]), body=summary([r['visual_body_error'] for r in records]),
        pnp_mean_seconds=float(np.mean([r['pnp_seconds'] for r in records])),
        test_status='Previously used local development split', backend='same grouped mixture score, 256 spatial points per hypothesis, checkerboard holdout')
    if split == 'test':
        baseline = np.array([r['baseline_error'] for r in records])
        fused = np.array([r['fused_error'] for r in records])
        before = (baseline[:, 0] < 1) & (baseline[:, 1] < 2)
        after = (fused[:, 0] < 1) & (fused[:, 1] < 2)
        report.update(baseline=summary(baseline), fused=summary(fused), rescue=int((~before & after).sum()),
            damage=int((before & ~after).sum()), accepted=sum(r['fusion']['accepted'] for r in records),
            lidar_only_refined=summary([r['lidar_only_refined_error'] for r in records]),
            candidate_pool_success_available=sum(r['candidate_pool_has_1m_2deg'] for r in records))
        report['orientation_groups'] = {}
        for name, supported in [('supported', True), ('outside', False)]:
            subset = [r for r in records if r['in_orientation_support'] == supported]
            if subset:
                report['orientation_groups'][name] = dict(camera=summary([r['camera_error'] for r in subset]),
                    baseline=summary([r['baseline_error'] for r in subset]), fused=summary([r['fused_error'] for r in subset]))
    save_json(destination / 'summary.json', report)
