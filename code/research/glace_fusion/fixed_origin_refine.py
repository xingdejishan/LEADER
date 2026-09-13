import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .correspondence_replay import digest
from .joint_solver import JointProblem, JointSolverConfig
from .pose_boundary import solver_pose
from .real_candidate_eval import pose_errors
from .replay_geometry import lidar_arrays, matched_subset


def oracle_mask(reprojection_mask, prediction, target, quality):
    if quality == 'reprojection':
        return reprojection_mask.copy()
    if quality != 'joint3d':
        raise ValueError('Unknown oracle criterion')
    distance = np.linalg.norm(prediction - target, axis=1)
    return reprojection_mask & np.isfinite(distance) & (distance < 1.)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8', newline='\n')


def objective_parts(problem, pose, support_lidar, support_camera):
    lidar, camera, _ = problem.residuals(pose)
    def part(residual, weights, mask):
        squared = np.sum(residual[mask] ** 2, axis=1)
        return float(weights[mask] @ np.log1p(squared))
    return dict(lidar=part(lidar, problem.w_L, support_lidar),
                camera=part(camera, problem.w_C, support_camera))


def run_refinement(pool, initial, extrinsic, K, arms, cfg):
    body, world, reliability = lidar_arrays(pool)
    control = JointProblem(body, world, reliability, np.empty((0, 2)), np.empty((0, 3)), K, extrinsic, cfg)
    frozen_lidar = control.support(initial)['lidar_inlier_mask']
    output, info, masks = {}, {}, {}
    for name, (xyz, uv) in arms.items():
        problem = JointProblem(body, world, reliability, uv, xyz, K, extrinsic, cfg)
        np.testing.assert_array_equal(problem.w_L, control.w_L)
        np.testing.assert_array_equal(problem.support(initial)['lidar_inlier_mask'], frozen_lidar)
        camera_weight = 0. if name == 'lidar_only' else 1.
        camera_support = problem.support(initial)['camera_inlier_mask'] if camera_weight else np.zeros(len(uv), bool)
        before = objective_parts(problem, initial, frozen_lidar, camera_support)
        start = time.perf_counter()
        pose, diagnostics = problem.refine(initial, camera_weight=camera_weight, lidar_support=frozen_lidar,
            require_camera_support=False)
        elapsed = time.perf_counter() - start
        pose = solver_pose(pose)
        after = objective_parts(problem, pose, frozen_lidar, camera_support)
        output[name] = pose
        info[name] = dict(diagnostics, elapsed_seconds=elapsed, input_camera_points=len(uv),
            fixed_objective_before=before, fixed_objective_after=after,
            pose_update=pose_errors(pose[None], initial)[0].tolist(),
            truncated_score_before=problem.score(initial), truncated_score_after=problem.score(pose))
        masks[name] = camera_support
    return output, info, frozen_lidar, masks


def success(errors, t=1., r=2.):
    errors = np.asarray(errors)
    return (errors[..., 0] < t) & (errors[..., 1] < r)


def summarize(records):
    report = {}
    groups = dict(all=records, supported=[r for r in records if r['in_orientation_support']])
    for group, rows in groups.items():
        if not rows:
            continue
        baseline = np.array([r['errors']['baseline'] for r in rows])
        lidar = np.array([r['errors']['lidar_only'] for r in rows])
        base_ok, lidar_ok = success(baseline), success(lidar)
        methods = {}
        for name in rows[0]['errors']:
            error = np.array([r['errors'][name] for r in rows])
            ok = success(error)
            harmed = base_ok & ~ok
            extra_harmed = lidar_ok & ~ok
            extra_rescued = ~lidar_ok & ok
            transition = {''.join(map(str, state)): int(np.sum((base_ok == state[0]) & (lidar_ok == state[1]) & (ok == state[2])))
                for state in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]}
            item = dict(success_1m_2deg=int(ok.sum()), success_05m_1deg=int(success(error, .5, 1).sum()),
                rescued_vs_v1=int((~base_ok & ok).sum()), harmed_vs_v1=int(harmed.sum()),
                new_harm_vs_lidar_only=int(extra_harmed.sum()), rescue_vs_lidar_only=int(extra_rescued.sum()),
                shared_v1_harm_with_lidar=int((base_ok & ~lidar_ok & ~ok).sum()),
                new_v1_harm_where_lidar_stable=int((base_ok & lidar_ok & ~ok).sum()),
                recovered_backend_harm=int((base_ok & ~lidar_ok & ok).sum()),
                median_t_m=float(np.median(error[:, 0])), median_r_deg=float(np.median(error[:, 1])),
                p95_t_m=float(np.quantile(error[:, 0], .95)), p95_r_deg=float(np.quantile(error[:, 1], .95)),
                transition_counts_v1_lidar_visual=transition,
                harmed_frame_ids=[r['image'] for r, flag in zip(rows, harmed) if flag],
                added_harm_frame_ids=[r['image'] for r, flag in zip(rows, extra_harmed) if flag],
                net_change_vs_lidar_by_60s_block={block: float(np.mean([int(a)-int(b) for a,b,r in zip(ok,lidar_ok,rows) if r['time_block']==block])) for block in sorted({r['time_block'] for r in rows})})
            if name != 'baseline':
                item.update(solver_called=sum(r['solver'][name]['solver_called'] for r in rows),
                    solver_converged=sum(r['solver'][name]['success'] for r in rows),
                    total_residual_calls=sum(r['solver'][name]['residual_calls'] for r in rows),
                    skipped_frame_ids=[r['image'] for r in rows if not r['solver'][name]['solver_called']],
                    median_pose_difference_vs_lidar=np.median(np.array([r['difference_vs_lidar'][name] for r in rows]), axis=0).tolist())
            methods[name] = item
        report[group] = dict(frames=len(rows), methods=methods)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--variant', choices=['selected', 'balanced'], default='selected')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--oracle-quality', choices=['reprojection', 'joint3d'], default='reprojection')
    args = parser.parse_args()
    root, out = args.bundle.resolve(), args.out.resolve()
    previous = root / 'outputs' / ('correspondence-replay-final-' + args.variant)
    prior_rows = {r['image']: r for r in map(json.loads, (previous / 'records.jsonl').read_text().splitlines())}
    rows = json.loads((root / 'data/test_rows.json').read_text())[:args.limit or None]
    cfg = JointSolverConfig(camera_scale_px=10.)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'frames').mkdir()
    write_json(out / 'protocol.json', dict(variant=args.variant, config=asdict(cfg),
        origin='cached v1_two_stage for every arm; no candidate selection, GT origin, acceptance or fallback',
        lidar='identical full pools, weights, scales and support mask computed once at v1, frozen during optimization',
        optimizer='same JointProblem.refine LM and Cauchy residuals, max_nfev=20; actual calls and convergence logged',
        visual='coefficient 0 for lidar_only, 1 otherwise; LiDAR coefficient never rescaled',
        controls='prediction_full is no-GT deployment input; prediction_common included only to pair existing oracle with its existing matched random subset',
        oracle='existing camera-GT q10/positive-depth subset' + (' intersect sparse-supervision 3D distance <1m' if args.oracle_quality == 'joint3d' else ''),
        oracle_quality=args.oracle_quality,
        oracle_limitation='offline diagnostic only; sparse supervision is not independently verified 3D truth',
        support='visual support computed once at v1 separately per arm; diagnostic mode still optimizes with 0-2 camera inliers when LiDAR has >=3, avoiding a hidden no-op; production default guard unchanged',
        ground_truth='camera cache reference for existing oracle; scan pool reference for main errors; both reported on harmed frames',
        scope='already-used development dates; no new training, threshold tuning or independent-reference prerequisite',
        prior_protocol_sha256=digest(previous / 'protocol.json'),
        source_hashes={name: digest(Path(__file__).parent / name) for name in ('fixed_origin_refine.py', 'joint_solver.py', 'replay_geometry.py')}))
    records = []
    started = time.perf_counter()
    with (out / 'records.jsonl').open('w', encoding='utf-8', buffering=1) as log:
        for row in rows:
            stem = row['image']
            pool_path = root / 'cache/lidar_pools' / (stem + '.npz')
            camera_path = root / 'cache' / args.variant / 'coordinates' / (stem + '.npz')
            for label, path in [('pool', pool_path), ('coordinates', camera_path)]:
                if digest(path) != prior_rows[stem]['hashes'][label]:
                    raise ValueError('Inputs changed since fixed replay: ' + stem)
            with np.load(pool_path) as saved:
                pool = {k: saved[k] for k in saved.files}
            with np.load(camera_path) as camera, np.load(previous / 'frames' / (stem + '.npz')) as saved:
                initial = solver_pose(pool['v1_two_stage'])
                full, common = saved['full_indices'], saved['common_indices']
                reliable, matched = saved['reliable_mask'], saved['matched_random_indices']
                K, E = saved['K'], saved['T_BC']
                xyz, uv = camera['xyz'][common], camera['uv'][common]
                reliable = oracle_mask(reliable, xyz, saved['supervision_xyz'], args.oracle_quality)
                if args.oracle_quality == 'joint3d':
                    matched = matched_subset(uv, reliable, saved['shape_hw'], 2090)
                arms = dict(lidar_only=(np.empty((0, 3)), np.empty((0, 2))),
                    prediction_full=(camera['xyz'][full], camera['uv'][full]), prediction_common=(xyz, uv),
                    oracle_prediction=(xyz[reliable], uv[reliable]), oracle_matched_random=(xyz[matched], uv[matched]))
                outputs, diagnostics, frozen_lidar, masks = run_refinement(pool, initial, E, K, arms, cfg)
                camera_gt_body = solver_pose(camera['GT']) @ np.linalg.inv(E)
                all_poses = dict(baseline=initial, **outputs)
                errors = {name: pose_errors(pose[None], pool['GT'])[0].tolist() for name, pose in all_poses.items()}
                camera_errors = {name: pose_errors(pose[None], camera_gt_body)[0].tolist() for name, pose in all_poses.items()}
                source = prior_rows[stem]
                record = dict(image=stem, time_block=str(int(stem)//60000000),
                    in_orientation_support=row['in_orientation_support'], errors=errors, camera_reference_errors=camera_errors,
                    camera_scan_gt_difference=pose_errors(camera_gt_body[None], pool['GT'])[0].tolist(),
                    solver=diagnostics, difference_vs_lidar={name: pose_errors(pose[None], outputs['lidar_only'])[0].tolist() for name, pose in outputs.items()},
                    input_hashes=source['hashes'], lidar_support_count=int(frozen_lidar.sum()))
                harmed = bool(success(errors['baseline']) and not success(errors['oracle_prediction']))
                record['oracle_harm_reference_check'] = dict(harmed_under_scan_reference=harmed,
                    harmed_under_camera_reference=bool(success(camera_errors['baseline']) and not success(camera_errors['oracle_prediction'])),
                    oracle_passes_camera_reference=bool(success(camera_errors['oracle_prediction'])))
                if reliable.any():
                    record['oracle_geometry'] = dict(points=int(reliable.sum()),
                        median_signed_px=np.median(saved['signed_residual_px'][reliable], axis=0).tolist(),
                        median_abs_along_error_m=float(np.median(np.abs(saved['along_error_m'][reliable]))),
                        median_perpendicular_m=float(np.median(saved['perpendicular_m'][reliable])))
                arrays = dict(initial=initial, frozen_lidar_support=frozen_lidar,
                    full_indices=full, common_indices=common, reliable_mask=reliable, matched_random_indices=matched,
                    GT_scan_evaluation_only=pool['GT'], GT_camera_body_evaluation_only=camera_gt_body,
                    **{name + '_pose': pose for name, pose in outputs.items()},
                    **{name + '_initial_camera_support': mask for name, mask in masks.items()})
                np.savez_compressed(out / 'frames' / (stem + '.npz'), **arrays)
                log.write(json.dumps(record, allow_nan=False) + '\n')
                records.append(record)
            if len(records) % 25 == 0:
                print(json.dumps(dict(frames=len(records), total=len(rows))), flush=True)
    write_json(out / 'summary.json', summarize(records))
    write_json(out / 'oracle_harmed_frames.json', [r for r in records if r['oracle_harm_reference_check']['harmed_under_scan_reference']])
    write_json(out / 'complete.json', dict(frames=len(records), elapsed_seconds=time.perf_counter()-started))
    print(json.dumps(summarize(records)['all'], indent=2))


if __name__ == '__main__':
    main()
