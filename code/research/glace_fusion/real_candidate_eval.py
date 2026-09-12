import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def pose_errors(poses, gt):
    translation = np.linalg.norm(poses[:, :3, 3] - gt[:3, 3], axis=1)
    matrices = np.concatenate([poses[:, :3, :3], gt[None, :3, :3]])
    if not np.isfinite(matrices).all() or np.max(np.abs(matrices.transpose(0, 2, 1) @ matrices - np.eye(3))) > .001:
        raise ValueError('Rotation metric requires near-orthonormal matrices')
    u, _, vh = np.linalg.svd(matrices)
    correction = np.broadcast_to(np.eye(3), matrices.shape).copy()
    correction[:, 2, 2] = np.linalg.det(u @ vh)
    rigid = u @ correction @ vh
    relative = rigid[:-1].transpose(0, 2, 1) @ rigid[-1]
    skew = np.column_stack([relative[:, 2, 1] - relative[:, 1, 2],
        relative[:, 0, 2] - relative[:, 2, 0], relative[:, 1, 0] - relative[:, 0, 1]])
    rotation = np.degrees(np.arctan2(np.linalg.norm(skew, axis=1) / 2,
        np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return np.column_stack([translation, rotation])


def score_camera(poses_body, E, xyz, uv, K):
    scores = []
    for start in range(0, len(poses_body), 32):
        poses = poses_body[start:start + 32] @ E
        camera = np.einsum('bnj,bjk->bnk', xyz[None] - poses[:, None, :3, 3], poses[:, :3, :3])
        projection = camera @ K.T
        with np.errstate(divide='ignore', invalid='ignore'):
            squared = np.sum((projection[:, :, :2] / projection[:, :, 2:] - uv[None]) ** 2, axis=2)
        error = np.minimum(squared / 100, 1)
        error[(camera[:, :, 2] <= 0) | ~np.isfinite(error)] = 1
        scores.extend(error.mean(1).tolist())
    return np.asarray(scores)


def dominance_counts(errors, scores):
    dominates = ((errors[:, None, 0] + .05 < errors[None, :, 0]) &
                 (errors[:, None, 1] + .1 < errors[None, :, 1]))
    delta = scores[:, None] - scores[None, :]
    return dict(pairs=int(dominates.sum()), correct=int((dominates & (delta < -1e-8)).sum()),
        ties=int((dominates & (np.abs(delta) <= 1e-8)).sum()))


def summarize(records, methods):
    groups = dict(all_spatial=lambda r: True,
        orientation_supported=lambda r: r['in_orientation_support'],
        new_supported=lambda r: r['in_orientation_support'] and not r['previous_probe'],
        previous_probe=lambda r: r['previous_probe'],
        outside_orientation_support=lambda r: not r['in_orientation_support'])
    result = {}
    for group, keep in groups.items():
        rows = [r for r in records if keep(r)]
        values = dict(frames=len(rows), methods={})
        for method in methods:
            e = np.asarray([r['errors'][method] for r in rows])
            if not len(e):
                continue
            values['methods'][method] = dict(mean_t=float(e[:, 0].mean()), median_t=float(np.median(e[:, 0])),
                mean_r_deg=float(e[:, 1].mean()), median_r_deg=float(np.median(e[:, 1])),
                p95_t=float(np.quantile(e[:, 0], .95)), p95_r_deg=float(np.quantile(e[:, 1], .95)),
                recall_05m_1deg=float(np.mean((e[:, 0] < .5) & (e[:, 1] < 1))),
                recall_1m_2deg=float(np.mean((e[:, 0] < 1) & (e[:, 1] < 2))),
                recall_2m_5deg=float(np.mean((e[:, 0] < 2) & (e[:, 1] < 5))))
            if method.startswith('camera_'):
                baseline = np.asarray([r['errors']['v1_two_stage'] for r in rows])
                baseline_success = (baseline[:, 0] < 1) & (baseline[:, 1] < 2)
                success = (e[:, 0] < 1) & (e[:, 1] < 2)
                values['methods'][method].update(rescued=int((~baseline_success & success).sum()),
                    harmed=int((baseline_success & ~success).sum()),
                    unchanged_pose=int(sum(r['selected'][method] == 0 for r in rows)))
        for label in ['stage1', 'improved']:
            counts = {k: sum(r['dominance'][label][k] for r in rows) for k in ['pairs', 'correct', 'ties']}
            counts['accuracy'] = counts['correct'] / counts['pairs'] if counts['pairs'] else None
            counts['half_credit_accuracy'] = (counts['correct'] + .5 * counts['ties']) / counts['pairs'] if counts['pairs'] else None
            values['dominance_' + label] = counts
        result[group] = values
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--stage1-coordinates', type=Path, required=True)
    parser.add_argument('--improved-coordinates', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-folder', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=10)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    (args.out / 'pools').mkdir()
    import MinkowskiEngine as ME
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader
    from .pose_boundary import solver_pose
    from data.NCLTVelodyne_datagenerator_mink import NCLT_mink
    from models.model_mink import LEADER
    from models.sc2pcr import Matcher
    from utils.pose_util import polar_expansion_to_cartesian
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
    from eval_clean_ablation import IndexedSubset, collate_samples
    from full_pool_robust_v1 import full_pool_refine
    torch.set_num_threads(4)
    torch.manual_seed(2089)
    np.random.seed(2089)
    rows = json.loads(args.rows.read_text())
    E = np.asarray(json.loads((args.scene / 'scene_meta.json').read_text())['T_BC_camera_to_body'])
    dataset = NCLT_mink(str(args.dataset_folder), train=False, voxel_size=.2, horizontal_res=1024)
    by_scan = {(Path(p).parent.parent.name, Path(p).stem): i for i, p in enumerate(dataset.pcs)}
    matched = {by_scan[r['sequence'], r['image']]: r for r in rows if (r['sequence'], r['image']) in by_scan}
    missing = [r['image'] for r in rows if (r['sequence'], r['image']) not in by_scan]
    coordinate_manifests = {label: json.loads((folder / 'manifest.json').read_text()) for label, folder in
        [('stage1', args.stage1_coordinates), ('improved', args.improved_coordinates)]}
    expected = {r['image'] for r in rows}
    for m in coordinate_manifests.values():
        if {r['image'] for r in m['samples']} != expected:
            raise ValueError('Camera inference frame manifest differs')
    manifest = dict(camera_frames=len(rows), paired_frames=len(matched), missing_exact_scans=missing,
        region_selection='fixed training-region GT coverage for evaluation only; no GT used to rank candidates',
        synchronization='exact camera/scan timestamp only', checkpoint_sha256=hashlib.sha256((args.checkpoint / 'model.safetensors').read_bytes()).hexdigest(),
        camera_manifests=coordinate_manifests,
        candidate_pool='v1-two-stage, original LEADER, all valid SC2 seedwise poses; duplicates rounded at 1e-5 removed',
        score='mean(min(L2 reprojection squared / 100,1)); invalid depth=1; no fitted weights',
        ties='within 1e-8 of minimum prefer v1; otherwise first candidate',
        dominance='better by at least 0.05m and 0.1deg; GT only used for evaluation labels',
        oracle='best available max(translation/1m, rotation/2deg), diagnostic only',
        seed='2089 + original dataset index for SC2', v1_thresholds_m=[1.2,.6])
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    model = LEADER(in_channels=3, out_channels=4, feat_channels=512, width=1024)
    model.load_state_dict(load_file(str(args.checkpoint / 'model.safetensors')), strict=True)
    model.cuda().eval()
    center = torch.tensor(json.loads((args.checkpoint / 'extra.json').read_text())['center_t'], device='cuda')
    matcher = Matcher(inlier_threshold=2., d_thre=2, num_iterations=10, ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    loader = DataLoader(IndexedSubset(dataset, sorted(matched)), batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_samples, num_workers=2, pin_memory=True)
    records = []
    methods = ['leader', 'v1_two_stage', 'camera_stage1', 'camera_improved', 'oracle']
    started = time.time()
    with (args.out / 'records.jsonl').open('w', buffering=1) as output, torch.inference_mode():
        for batch in loader:
            encoded = model.encoder(ME.SparseTensor(batch['feats'].cuda(), batch['coords'].cuda()))
            pred = model.decoder(encoded.F).float()
            batch_index = encoded.C[:, 0].long()
            stride = torch.tensor(encoded.tensor_stride, device='cuda', dtype=torch.float32)
            local = polar_expansion_to_cartesian((encoded.C[:, 1:].float() + stride / 2) * .2, 204.8)
            for position, index in enumerate(batch['indices']):
                row = matched[int(index)]
                torch.manual_seed(2089 + int(index))
                np.random.seed(2089 + int(index))
                selected = batch_index == position
                source, target, reliability = local[selected], pred[selected, :3], pred[selected, 3]
                keep = max(min(50, reliability.numel()), int(.5 * reliability.numel()))
                top = torch.topk(reliability, keep).indices
                initial, seeds, fitness = matcher.estimator(source[top][None], target[top][None], return_hypotheses=True)
                refined, _ = full_pool_refine(initial[0], source, target)
                correction = batch['T_corr'][position].cuda().float()
                gt = batch['T'][position].numpy().astype(np.float64)
                def world_pose(T):
                    T = T.clone()
                    T[..., :3, 3] += center
                    return (T @ correction).cpu().numpy().astype(np.float64)
                leader, v1 = world_pose(initial[0]), world_pose(refined)
                raw_poses = np.concatenate([v1[None], leader[None], world_pose(seeds[0])])
                candidates, invalid = [], 0
                for pose in raw_poses:
                    try:
                        candidates.append(solver_pose(pose))
                    except ValueError:
                        invalid += 1
                        if len(candidates) < 2:
                            raise
                candidates = np.asarray(candidates)
                _, unique = np.unique(np.round(candidates.reshape(len(candidates), -1), 5), axis=0, return_index=True)
                candidates = candidates[np.sort(unique)]
                errors = pose_errors(candidates, gt)
                record = dict(index=int(index), image=row['image'], sequence=row['sequence'],
                    in_orientation_support=row['in_orientation_support'], previous_probe=row['previous_probe'],
                    distance_from_anchor_m=row['distance_from_anchor_m'], angle_from_anchor_deg=row['angle_from_anchor_deg'],
                    candidates=len(candidates), invalid_candidates=invalid,
                    errors=dict(leader=pose_errors(leader[None], gt)[0].tolist(), v1_two_stage=pose_errors(v1[None], gt)[0].tolist()),
                    selected={}, dominance={})
                score_arrays = {}
                reference = None
                for label, folder in [('stage1', args.stage1_coordinates), ('improved', args.improved_coordinates)]:
                    with np.load(folder / 'coordinates' / (row['image'] + '.npz')) as data:
                        xyz, uv, K, camera_gt = [data[k] for k in ['xyz', 'uv', 'K', 'GT']]
                    if reference is not None:
                        for a, b in zip([uv, K, camera_gt], reference):
                            np.testing.assert_array_equal(a, b)
                    reference = [uv, K, camera_gt]
                    scores = score_camera(candidates, E, xyz, uv, K)
                    best = int(np.flatnonzero(scores <= scores.min() + 1e-8)[0])
                    name = 'camera_' + label
                    record['selected'][name] = best
                    record['errors'][name] = errors[best].tolist()
                    record['dominance'][label] = dominance_counts(errors, scores)
                    score_arrays[label + '_scores'] = scores
                oracle = int(np.argmin(np.maximum(errors[:, 0], errors[:, 1] / 2)))
                record['errors']['oracle'] = errors[oracle].tolist()
                record['gt_alignment_error'] = pose_errors((camera_gt @ np.linalg.inv(E))[None], gt)[0].tolist()
                np.savez_compressed(args.out / 'pools' / (row['image'] + '.npz'),
                    candidate_T_WB=candidates, candidate_errors=errors, GT=gt, leader=leader, v1_two_stage=v1,
                    c_local_all=source.cpu().numpy(), c_pred_all=target.cpu().numpy(), u_pred_all=reliability.cpu().numpy(),
                    center_t=center.cpu().numpy(), T_corr=correction.cpu().numpy(), **score_arrays)
                records.append(record)
                output.write(json.dumps(record) + '\n')
            progress = dict(completed=len(records), total=len(matched), elapsed_s=time.time()-started)
            (args.out / 'progress.json').write_text(json.dumps(progress, indent=2))
            print(json.dumps(progress), flush=True)
    report = dict(complete=len(records)==len(matched), manifest=manifest, groups=summarize(records, methods))
    (args.out / 'report.json').write_text(json.dumps(report, indent=2))
    print('REAL_CANDIDATES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
