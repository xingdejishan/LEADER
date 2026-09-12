import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .lidar_supervision import camera_targets
from .pose_boundary import solver_pose
from .real_candidate_eval import pose_errors
from .replay_geometry import (accept_on_holdout, camera_scores, candidates_from_pool, grid_cells,
    lidar_scores, matched_subset, point_errors, refine_selected, sample_pixels, spatial_holdout)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def supervision_from_scan(scan, uv, K, gt, extrinsic, shape):
    dtype = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('intensity', 'u1'), ('ring', 'u1')])
    if Path(scan).stat().st_size % dtype.itemsize:
        raise ValueError('Incomplete NCLT scan')
    raw = np.fromfile(scan, dtype=dtype)
    body = np.column_stack([raw[k] for k in ('x', 'y', 'z')]).astype(float) * .005 - 100
    T_WB = gt @ np.linalg.inv(extrinsic)
    world = body @ T_WB[:3, :3].T + T_WB[:3, 3]
    target, supported = camera_targets(world, uv, K, np.linalg.inv(gt), *shape, 3 * shape[0] / 480)
    return target, supported > 0


def diagnostic_metrics(errors, uv, shape):
    result = dict(count=len(uv))
    for name in ('angle_deg', 'perpendicular_m', 'along_error_m', 'range_error_m'):
        if name not in errors:
            continue
        values = errors[name]
        values = values[np.isfinite(values)]
        result[name] = dict(median=float(np.median(values)), median_abs=float(np.median(np.abs(values))),
            p90_abs=float(np.quantile(np.abs(values), .9))) if len(values) else None
    finite = errors['positive'] & np.isfinite(errors['signed_px']).all(1)
    signed = errors['signed_px'][finite]
    result['positive_fraction'] = float(errors['positive'].mean()) if len(uv) else None
    result['q10'] = float((errors['squared_px'] < 100).mean()) if len(uv) else None
    if len(signed):
        lengths = np.linalg.norm(signed, axis=1)
        result.update(median_signed_px=np.median(signed, axis=0).tolist(),
            mean_signed_px=np.mean(signed, axis=0).tolist(),
            residual_direction_coherence=float(np.linalg.norm(np.mean(signed / np.maximum(lengths[:, None], 1e-9), axis=0))))
    cells = grid_cells(uv, shape)
    result['by_image_block'] = {str(cell): dict(count=int((cells == cell).sum()),
        q10=float(np.mean(errors['squared_px'][cells == cell] < 100))) for cell in np.unique(cells)}
    return result


def reference_input(folder, stem, uv):
    path = folder / (stem + '.npz')
    if not path.exists():
        return None
    with np.load(path) as data:
        if data['uv'].shape != uv.shape or not np.allclose(data['uv'], uv, atol=1e-6, rtol=0):
            raise ValueError('Reference pixels must match the full cached grid in the same order')
        xyz, valid = data['xyz'].astype(float), data['valid'].astype(bool)
        if xyz.shape != (len(uv), 3) or valid.shape != (len(uv),):
            raise ValueError('Reference dimensions differ')
        if not np.isfinite(xyz[valid]).all():
            raise ValueError('Nonfinite verified reference')
        return xyz, valid


def replay_arm(poses, pool, xyz, uv, K, extrinsic, shape, cached_lidar):
    legacy = camera_scores(poses, xyz, uv, K, extrinsic, shape)
    strict = camera_scores(poses, xyz, uv, K, extrinsic, shape, strict=True)
    joint = (cached_lidar + legacy) / 2
    camera_index, joint_index = int(np.argmin(legacy)), int(np.argmin(joint))
    refined, fit = refine_selected(poses[joint_index], pool, xyz, uv, K, extrinsic, shape)
    selection, verification = spatial_holdout(uv, shape)
    blocks = camera_scores(poses, xyz[selection], uv[selection], K, extrinsic, shape, block_weights=True, strict=True)
    block_index = int(np.argmin((cached_lidar + blocks) / 2)) if selection.any() else 0
    block_refined, block_fit = refine_selected(poses[block_index], pool, xyz[selection], uv[selection], K, extrinsic, shape, True)
    gated = {}
    reasons = {}
    for name, proposal in [('selected', poses[block_index]), ('refined', block_refined)]:
        accepted, reason = accept_on_holdout(poses[0], proposal, pool, xyz[verification], uv[verification], K, extrinsic, shape)
        gated[name] = proposal if accepted else poses[0]
        reasons[name] = dict(accepted=accepted, reason=reason)
    results = dict(camera_selected=poses[camera_index], joint_selected=poses[joint_index], joint_refined=refined,
        block_selected=poses[block_index], block_refined=block_refined,
        block_accepted=gated['selected'], block_refined_accepted=gated['refined'])
    info = dict(points=len(uv), cells=len(np.unique(grid_cells(uv, shape))),
        selection_points=int(selection.sum()), verification_points=int(verification.sum()),
        selected_indices=dict(camera=camera_index, joint=joint_index, block=block_index),
        legacy_strict_score_max_difference=float(np.max(np.abs(legacy - strict))),
        fit=fit, block_fit=block_fit, acceptance=reasons)
    return results, info, dict(camera=legacy, strict_camera=strict, joint=joint, block_camera=blocks)


def aggregate(records):
    groups = dict(all=records)
    for date in sorted({r['sequence'] for r in records}):
        groups[date] = [r for r in records if r['sequence'] == date]
    groups['orientation_supported'] = [r for r in records if r['in_orientation_support']]
    reference_rows = [r for r in records if 'reference_common' in r['arms']]
    if reference_rows:
        groups['reference_available'] = reference_rows
    output = {}
    for group, rows in groups.items():
        if not rows:
            continue
        methods = sorted(set.intersection(*(set(r['errors']) for r in rows)))
        report = dict(frames=len(rows), reachable_1m_2deg=sum(r['reachable']['1m_2deg'] for r in rows),
            reachable_05m_1deg=sum(r['reachable']['05m_1deg'] for r in rows), methods={})
        baseline = np.array([r['errors']['baseline'] for r in rows])
        base_ok = (baseline[:, 0] < 1) & (baseline[:, 1] < 2)
        blocks = np.array([r['time_block'] for r in rows])
        for method in methods:
            errors = np.array([r['errors'][method] for r in rows])
            ok = (errors[:, 0] < 1) & (errors[:, 1] < 2)
            rescued, harmed = int((ok & ~base_ok).sum()), int((~ok & base_ok).sum())
            block_changes = {str(b): float((ok[blocks == b].astype(float) - base_ok[blocks == b]).mean()) for b in np.unique(blocks)}
            best = np.array([r['best_candidate_error'] for r in rows])
            report['methods'][method] = dict(success_1m_2deg=int(ok.sum()),
                success_05m_1deg=int(((errors[:, 0] < .5) & (errors[:, 1] < 1)).sum()),
                median_t_m=float(np.median(errors[:, 0])), median_r_deg=float(np.median(errors[:, 1])),
                p95_t_m=float(np.quantile(errors[:, 0], .95)), p95_r_deg=float(np.quantile(errors[:, 1], .95)),
                rescued=rescued, harmed=harmed, rescue_fraction_all=rescued / len(rows), harm_fraction_all=harmed / len(rows),
                rescue_given_baseline_failure=rescued / int((~base_ok).sum()) if (~base_ok).any() else None,
                harm_given_baseline_success=harmed / int(base_ok.sum()) if base_ok.any() else None,
                mean_extra_translation_vs_minimax_candidate_m=float((errors[:, 0] - best[:, 0]).mean()),
                mean_extra_rotation_vs_minimax_candidate_deg=float((errors[:, 1] - best[:, 1]).mean()),
                net_success_change_by_60s_block=block_changes)
        output[group] = report
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', choices=['selected', 'balanced', 'previous'], default='selected')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    root, out = args.bundle.resolve(), args.out.resolve()
    reference_manifest = None
    if args.reference:
        reference_manifest = json.loads((args.reference / 'manifest.json').read_text())
        if reference_manifest.get('verification') not in ('manual_physical_features', 'independent_multiview') or not reference_manifest.get('provenance'):
            raise ValueError('References require independent verification and documented provenance')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'frames').mkdir()
    meta = json.loads((root / 'data/train_scene/scene_meta.json').read_text())
    E = solver_pose(np.asarray(meta['T_BC_camera_to_body']))
    rows = json.loads((root / 'data/test_rows.json').read_text())[:args.limit or None]
    protocol = dict(variant=args.variant, seed=2089, pixel_budget=256, grid=4,
        candidates='unchanged cached v1_two_stage + original leader + all cached SC2 hypotheses; no GT, no new candidates',
        baseline='v1_two_stage; original leader reported separately',
        input_interface='2D pixel reprojection of world-coordinate predictions at T_WC=T_WB@T_BC',
        supervision='existing sparse LiDAR target constructor; same-time scan placed by camera GT; circular diagnostic, NOT independent reference',
        matched_arms='supervision_common, prediction_common and shuffled_common share exact pixels/count/coverage',
        oracle='prediction oracle q10 subset and same-block/count random control; both OFFLINE diagnostic',
        common_mask='depends on GT-generated supervision support; common-mask arms are diagnostic, not deployable',
        inference_only_arm='prediction_full; pixel sampling and spatial acceptance use no GT',
        scales=dict(lidar_m=.3, camera_px=10), visual_weight=.5,
        refinement='existing JointProblem.refine, max_nfev=20, one selected candidate; report raw output even if worse',
        holdout='4x4 checkerboard: even blocks select/refine, odd blocks verify; candidate-independent blocks',
        holdout_acceptance='>=6 points in >=3 blocks; >=20% support in >=3 blocks; heldout visual score improves and LiDAR score does not worsen',
        calibration_status='inherited fixed scales and uncalibrated conservative gates, no tuning on this test',
        invalid_points='legacy keeps negative/nonfinite depth in denominator; strict also assigns full cost outside raster',
        best_candidate='min max(translation/1m, rotation/2deg); signed per-axis regrets may be negative',
        split='2012-02-12, already used for development; not final independent evaluation',
        reference_status='user-declared independent reference, pending independent provenance review' if args.reference else 'unavailable; reference arm skipped',
        reference_manifest=reference_manifest,
        source_hashes={name: digest(Path(__file__).parent / name) for name in
            ('correspondence_replay.py', 'replay_geometry.py', 'joint_solver.py', 'lidar_supervision.py')},
        coordinate_manifest_sha256=digest(root / 'cache' / args.variant / 'manifest.json'))
    dump(out / 'protocol.json', protocol)
    started = time.perf_counter()
    records = []
    with (out / 'records.jsonl').open('w', encoding='utf-8', buffering=1) as log:
        for row in rows:
            frame_started = time.perf_counter()
            stem = row['image']
            pool_path = root / 'cache/lidar_pools' / (stem + '.npz')
            camera_path = root / 'cache' / args.variant / 'coordinates' / (stem + '.npz')
            with np.load(pool_path) as data:
                pool = {k: data[k] for k in data.files}
            with np.load(camera_path) as data:
                camera = {k: data[k] for k in data.files}
            poses, names = candidates_from_pool(pool)
            xyz, uv, K, gt = [camera[k] for k in ('xyz', 'uv', 'K', 'GT')]
            shape = (480, round(row['stored_size_hw'][1] * 480 / row['stored_size_hw'][0]))
            scan = root / 'data/scans' / row['sequence'] / 'velodyne_sync' / (stem + '.bin')
            targets, support = supervision_from_scan(scan, uv, K, gt, E, shape)
            reference = reference_input(args.reference, stem, uv) if args.reference else None
            common = support.copy()
            if reference is not None:
                common &= reference[1]
            available = np.flatnonzero(common)
            indices = available[sample_pixels(uv[available], shape, 256, 2089)]
            full = sample_pixels(uv, shape, 256, 2089)
            common_uv, common_xyz = uv[indices], xyz[indices]
            target_world = targets[indices] @ gt[:3, :3].T + gt[:3, 3]
            point_diagnostics = point_errors(common_xyz, common_uv, K, gt, shape, targets[indices])
            reliable = point_diagnostics['positive'] & (point_diagnostics['squared_px'] < 100)
            matched = matched_subset(common_uv, reliable, shape, 2090)
            permutation = np.random.default_rng(2091).permutation(len(indices))
            arms = dict(prediction_full=(xyz[full], uv[full]),
                prediction_common=(common_xyz, common_uv), supervision_common=(target_world, common_uv),
                shuffled_common=(common_xyz[permutation], common_uv),
                oracle_prediction=(common_xyz[reliable], common_uv[reliable]),
                oracle_matched_random=(common_xyz[matched], common_uv[matched]))
            if reference is not None:
                arms['reference_common'] = (reference[0][indices], common_uv)
            cached_lidar = lidar_scores(poses, pool)
            errors = pose_errors(poses, pool['GT'])
            best_index = int(np.argmin(np.max(errors / [1, 2], axis=1)))
            record = dict(image=stem, sequence=row['sequence'], time_block=str(int(stem) // 60000000),
                in_orientation_support=row['in_orientation_support'], candidates=len(poses),
                reachable={'1m_2deg': bool(((errors[:, 0] < 1) & (errors[:, 1] < 2)).any()),
                           '05m_1deg': bool(((errors[:, 0] < .5) & (errors[:, 1] < 1)).any())},
                best_candidate_index=best_index, best_candidate_error=errors[best_index].tolist(),
                componentwise_min_candidate_error=errors.min(0).tolist(),
                errors=dict(baseline=errors[0].tolist(), original_leader=errors[1].tolist(),
                            lidar_rescore=errors[np.argmin(cached_lidar)].tolist()),
                common_points=len(indices), sparse_support=int(support.sum()),
                point_metrics=diagnostic_metrics(point_diagnostics, common_uv, shape), arms={},
                camera_scan_gt_difference=pose_errors((solver_pose(gt) @ np.linalg.inv(E))[None], pool['GT'])[0].tolist(),
                hashes=dict(pool=digest(pool_path), coordinates=digest(camera_path), scan=digest(scan),
                    reference=digest(args.reference / (stem + '.npz')) if reference is not None else None))
            saved = dict(candidate_T_WB=poses, candidate_names=np.array(names), candidate_errors=errors,
                common_indices=indices, full_indices=full, common_uv=common_uv, prediction_xyz=common_xyz,
                supervision_xyz=target_world, reliable_mask=reliable, matched_random_indices=matched,
                permutation=permutation, lidar_scores=cached_lidar, K=K, T_BC=E, shape_hw=np.array(shape),
                signed_residual_px=point_diagnostics['signed_px'], angle_deg=point_diagnostics['angle_deg'],
                along_error_m=point_diagnostics['along_error_m'], perpendicular_m=point_diagnostics['perpendicular_m'],
                GT_evaluation_only=pool['GT'])
            for name, (arm_xyz, arm_uv) in arms.items():
                output, info, scores = replay_arm(poses, pool, arm_xyz, arm_uv, K, E, shape, cached_lidar)
                record['arms'][name] = info
                for stage, pose in output.items():
                    record['errors'][name + '/' + stage] = pose_errors(pose[None], pool['GT'])[0].tolist()
                    saved[name + '__' + stage] = pose
                for label, score in scores.items():
                    saved[name + '__score_' + label] = score
            record['elapsed_seconds'] = time.perf_counter() - frame_started
            np.savez_compressed(out / 'frames' / (stem + '.npz'), **saved)
            log.write(json.dumps(record, allow_nan=False) + '\n')
            records.append(record)
            if len(records) % 10 == 0:
                print(json.dumps(dict(frames=len(records), total=len(rows), elapsed_seconds=time.perf_counter() - started)), flush=True)
    report = aggregate(records)
    dump(out / 'summary.json', report)
    dump(out / 'complete.json', dict(frames=len(records), elapsed_seconds=time.perf_counter() - started,
        reference_frames=sum('reference_common' in r['arms'] for r in records),
        reference_required_for_root_cause_claim=True))
    print(json.dumps(dict(frames=len(records), reachable=report['all']['reachable_1m_2deg'], output=str(out))))


if __name__ == '__main__':
    main()
