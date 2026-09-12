import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import MinkowskiEngine as ME
import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from .glace_adapter import GLACEAdapter, deit_global_feature_fn
from .joint_solver import JointProblem, JointSolverConfig, solve, pose_distance
from .pose_boundary import solver_pose
from .make_glace_scene import calibration_chain
from .nclt_camera import camera_rows, stored_intrinsics, preprocess_image, TEST_DATES

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from eval_clean_ablation import IndexedSubset, collate_samples
from full_pool_robust_v1 import full_pool_refine
from data.NCLTVelodyne_datagenerator_mink import NCLT_mink
from models.model_mink import LEADER
from models.sc2pcr import Matcher
from utils.pose_util import polar_expansion_to_cartesian


def errors(pose, gt):
    if pose is None:
        return None
    translation, rotation = pose_distance(pose, gt)
    return [float(translation), float(np.rad2deg(rotation))]


def summarize(records, method):
    values = np.asarray([r['errors'][method] for r in records if r['errors'][method] is not None])
    count = len(records)
    if not len(values):
        return {'frames': count, 'poses': 0, 'coverage': 0., 'recall_1m_2deg': 0.}
    return {'frames': count, 'poses': len(values), 'coverage': len(values) / count,
            'mean_t': float(values[:, 0].mean()), 'mean_r': float(values[:, 1].mean()),
            'median_t': float(np.median(values[:, 0])), 'median_r': float(np.median(values[:, 1])),
            'p95_t': float(np.quantile(values[:, 0], .95)), 'p95_r': float(np.quantile(values[:, 1], .95)),
            'recall_0.5m_1deg': float(np.sum((values[:, 0] < .5) & (values[:, 1] < 1)) / count),
            'recall_1m_2deg': float(np.sum((values[:, 0] < 1) & (values[:, 1] < 2)) / count)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--head', required=True, type=Path)
    parser.add_argument('--quality-report', required=True, type=Path)
    parser.add_argument('--vendor', required=True)
    parser.add_argument('--deit-checkpoint', required=True)
    parser.add_argument('--camera-root', required=True, type=Path)
    parser.add_argument('--dataset-folder', default='/root/rivermind-data/datasets')
    parser.add_argument('--checkpoint', default='/root/rivermind-data/LEADER/checkpoints/checkpoint_epoch_49', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--sequence', choices=TEST_DATES, default='2012-02-12')
    parser.add_argument('--max-sync-ms', type=float, default=10.)
    args = parser.parse_args()
    quality = json.loads(args.quality_report.read_text())
    if not 1 <= args.limit <= 128 or not 0 < args.max_sync_ms <= 50:
        raise SystemExit('Diagnostic requires 1-128 frames and synchronization within 50 ms')
    head_hash = hashlib.sha256(args.head.read_bytes()).hexdigest()
    if quality.get('head_sha256') != head_hash:
        raise SystemExit('Quality report does not match the requested checkpoint')
    torch.set_num_threads(4)
    torch.manual_seed(2089)
    rows = camera_rows(args.camera_root, 5)
    camera = {date: [r for r in rows if r['sequence'] == date] for date in TEST_DATES}
    times = {date: np.asarray([r['timestamp_us'] for r in selected]) for date, selected in camera.items()}
    K_raw, E = calibration_chain(args.camera_root, 5, '0.035,0.002,-1.23,-179.93,-0.23,0.50', None)
    config = JointSolverConfig()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {'head_sha256': head_hash, 'solver': config.__dict__, 'limit': args.limit,
                'camera_root': str(args.camera_root), 'checkpoint': str(args.checkpoint),
                'gt': 'same original LEADER scan GT for every method',
                'missing_or_rejected_policy': 'native v2 abstains; deployed fusion falls back to v1-two-stage',
                'max_sync_delta_s': args.max_sync_ms / 1000,
                'scope': 'diagnostic only; failed head explicitly retained to measure incremental value',
                'sequence': args.sequence, 'quality_passed': bool(quality.get('ready_for_test'))}
    manifest_path = args.out / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit('Existing results have a different configuration')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    records_path = args.out / 'records.jsonl'
    records = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    dataset = NCLT_mink(args.dataset_folder, train=False, voxel_size=.2, horizontal_res=1024)
    eligible = []
    timestamps = times[args.sequence]
    for index, filename in enumerate(dataset.pcs):
        path = Path(filename)
        if path.parent.parent.name != args.sequence or not len(timestamps):
            continue
        ts = int(path.stem)
        j = int(np.searchsorted(timestamps, ts))
        neighbors = [k for k in [j-1, j] if 0 <= k < len(timestamps)]
        if min(abs(int(timestamps[k])-ts) for k in neighbors) <= args.max_sync_ms*1000:
            eligible.append(index)
    if len(eligible) < args.limit:
        raise SystemExit(f'Only {len(eligible)} synchronized frames available')
    indices = [eligible[j] for j in np.linspace(0,len(eligible)-1,args.limit,dtype=int)]
    (args.out / 'selected_indices.json').write_text(json.dumps({'eligible':len(eligible),'indices':indices}))
    total = len(indices)
    loader = DataLoader(IndexedSubset(dataset, indices[len(records):]), batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_samples, num_workers=4, pin_memory=True)
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, width=1024)
    model.load_state_dict(load_file(str(args.checkpoint / 'model.safetensors')), strict=True)
    model.cuda().eval()
    center = torch.tensor(json.loads((args.checkpoint / 'extra.json').read_text())['center_t'], device='cuda')
    matcher = Matcher(inlier_threshold=2., d_thre=2, num_iterations=10, ratio=.15,
                      nms_radius=.1, max_points=3000, k1=30)
    feature_fn = deit_global_feature_fn(args.vendor, args.deit_checkpoint)
    adapter = GLACEAdapter(args.vendor, args.head, T_BC=E, global_feature_fn=feature_fn)
    started = time.time()
    methods = ['leader', 'v1_two_stage', 'camera', 'v2_native', 'fusion_with_v1_fallback']
    with records_path.open('a', buffering=1) as output, torch.inference_mode():
        for batch in loader:
            encoded = model.encoder(ME.SparseTensor(batch['feats'].cuda(), batch['coords'].cuda()))
            pred = model.decoder(encoded.F).float()
            batch_index = encoded.C[:, 0].long()
            stride = torch.tensor(encoded.tensor_stride, device='cuda', dtype=torch.float32)
            local = polar_expansion_to_cartesian((encoded.C[:, 1:].float() + stride / 2) * .2, 204.8)
            for position, index in enumerate(batch['indices']):
                torch.manual_seed(2089 + int(index))
                np.random.seed(2089 + int(index))
                mask = batch_index == position
                source, target, reliability = local[mask], pred[mask, :3], pred[mask, 3]
                keep = max(min(50, reliability.numel()), int(.5 * reliability.numel()))
                top = torch.topk(reliability, keep).indices
                initial, seeds, _ = matcher.estimator(source[top][None], target[top][None], return_hypotheses=True)
                initial = initial[0]
                refined, _ = full_pool_refine(initial, source, target)
                correction = batch['T_corr'][position].cuda().float()
                gt = batch['T'][position].numpy()
                def world_pose(T):
                    T = T.clone()
                    T[..., :3, 3] += center
                    return (T @ correction).cpu().numpy()
                baseline, v1 = world_pose(initial), world_pose(refined)
                path = Path(dataset.pcs[index])
                date, ts = path.parent.parent.name, int(path.stem)
                timestamps = times[date]
                j = int(np.searchsorted(timestamps, ts))
                neighbors = [k for k in [j - 1, j] if 0 <= k < len(timestamps)]
                j = min(neighbors, key=lambda k: abs(int(timestamps[k]) - ts)) if neighbors else None
                delta = None if j is None else (int(timestamps[j]) - ts) / 1e6
                camera_pose = joint_pose = None
                status = 'NO_SYNCHRONIZED_IMAGE'
                if delta is not None and abs(delta) <= args.max_sync_ms / 1000:
                    row = camera[date][j]
                    image_path = args.camera_root / row['saved_path']
                    K, _ = stored_intrinsics(K_raw, row, image_path)
                    image, K = preprocess_image(image_path, K, 616)
                    camera_output = adapter.infer(image, K)
                    camera_pose = camera_output.T_WB
                    if not records:
                        np.savez(args.out / 'first_frame_pose_inputs.npz', v1=v1, baseline=baseline, correction=correction.cpu().numpy())
                    C = correction.cpu().numpy()
                    body = (source.cpu().numpy() - C[:3, 3]) @ C[:3, :3]
                    world = target.cpu().numpy() + center.cpu().numpy()
                    problem = JointProblem(body, world, reliability.cpu().numpy(), camera_output.uv,
                                           camera_output.xyz_world, K, E, config)
                    seed_poses = world_pose(seeds[0])
                    candidates = np.concatenate([baseline[None], seed_poses])
                    candidates = np.asarray([solver_pose(T) for T in candidates])
                    result = solve(problem, solver_pose(v1), camera_pose, candidates, mode='joint_refine')
                    joint_pose, status = result.pose, result.status
                poses = [baseline, v1, camera_pose, joint_pose, joint_pose if joint_pose is not None else v1]
                record = {'index': int(index), 'sequence': date, 'scan_timestamp': ts, 'sync_delta_s': delta,
                          'status': status,
                          'camera_support_at_v1': float(problem.support(v1)['camera_ratio']),
                          'camera_support_at_scan_gt': float(problem.support(gt)['camera_ratio']),
                          'solver_seconds': result.diagnostics.get('elapsed_s'),
                          'errors': {name: errors(T, gt) for name, T in zip(methods, poses)}}
                output.write(json.dumps(record) + '\n')
                records.append(record)
            progress = {'frames': len(records), 'total': total, 'session_seconds': time.time() - started,
                        'status_counts': dict(Counter(r['status'] for r in records))}
            (args.out / 'progress.json').write_text(json.dumps(progress, indent=2))
            print(json.dumps(progress), flush=True)
    report = {'frames': len(records), 'complete': len(records) == total, 'manifest': manifest,
              'all_frames': {name: summarize(records, name) for name in methods},
              'paired_frames': {name: summarize([r for r in records if r['status'] != 'NO_SYNCHRONIZED_IMAGE'], name) for name in methods},
              'sequences': {date: {name: summarize([r for r in records if r['sequence'] == date], name)
                                    for name in methods} for date in TEST_DATES},
              'status_counts': dict(Counter(r['status'] for r in records))}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
