"""Joint LiDAR (LEADER) + Camera (GLACE) evaluation.

Stage A (LEADER side) is run separately and offline:
    python run_mink.py --mode test --dataset NCLT --export_fusion_pool <DIR> ...
which stores, per query scan and WITHOUT changing LEADER's own pipeline:
    the full pre-top-50% correspondence pool (c_local_all, c_pred_all, u_pred_all),
    T_corr / center_t (frame bookkeeping), the final T_WB, GT T_WB, scan timestamp,
    and (by default) ALL valid SC2-PCR seedwise hypotheses in T_WB form.

Stage B (this script) attaches the GLACE camera branch and the shared pose
solver (v2): every frame goes through
    candidates -> joint scoring -> joint refinement -> acceptance.
--backend compare additionally reports the 'select' (per-modality scoring) and
'joint' (no refinement) baselines per frame, so the source of any gain can be
attributed to multimodal evidence vs. extra candidates vs. joint refinement.
--backend fallback keeps the v1 confidence-gated fusion module for reference.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    from .lidar_camera_fusion import FusionConfig, localize
    from .packet import (IsotonicCalibrator, camera_support_rate, lidar_pool_from_export,
                         lidar_support_rate, make_evidence_stamps, make_fusion_evidence)
    from .inference_contract import InferenceSession, resolve_contract, RGB_PROTOCOL
    from .glace_adapter import GLACEAdapter, deit_global_feature_fn
    from .nclt_camera import (TEST_DATES, camera_rows, stored_intrinsics, preprocess_image,
                              NCLTTrajectory, trajectory_path)
    from .joint_solver import JointProblem, JointSolverConfig, pose_distance, solve
except ImportError:  # running with the package directory itself on sys.path
    from lidar_camera_fusion import FusionConfig, localize
    from packet import (IsotonicCalibrator, camera_support_rate, lidar_pool_from_export,
                        lidar_support_rate, make_evidence_stamps, make_fusion_evidence)
    from inference_contract import InferenceSession, resolve_contract, RGB_PROTOCOL
    from glace_adapter import GLACEAdapter, deit_global_feature_fn
    from nclt_camera import (TEST_DATES, camera_rows, stored_intrinsics, preprocess_image,
                             NCLTTrajectory, trajectory_path)
    from joint_solver import JointProblem, JointSolverConfig, pose_distance, solve


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool_dir', required=True, help='export_fusion_pool directory')
    parser.add_argument('--vendor_dir', required=True, help='ACE/GLACE vendor directory')
    parser.add_argument('--glace_head', required=True)
    parser.add_argument('--glace_encoder', default='')
    parser.add_argument('--deit_checkpoint', default='',
                        help='required when the head consumes global features')
    parser.add_argument('--camera_root', required=True, help='NCLT_camera_v1 root')
    parser.add_argument('--camera_number', type=int, default=5)
    parser.add_argument('--body_to_lb3_ssc_deg', default='0.035,0.002,-1.23,-179.93,-0.23,0.50')
    parser.add_argument('--image_resolution', type=int, default=None)
    parser.add_argument('--feature_split', default=None, help='RGB scene/test cache; otherwise extract from RGB paths online')
    parser.add_argument('--pose_backend', choices=['opencv', 'dsacstar', 'none'], default='opencv')
    parser.add_argument('--dataset_folder', required=True, help='Root containing NCLT/ GT trajectories')
    parser.add_argument('--allow_partial', action='store_true', help='Allow missing test camera sequences')
    parser.add_argument('--lidar_threshold_m', type=float, default=0.3,
                        help='s_L for the diagnostic support rate')
    parser.add_argument('--camera_threshold_px', type=float, default=4.0,
                        help='s_C for the diagnostic support rate')
    parser.add_argument('--solver_config', default='', help='JointSolverConfig.save() json')
    parser.add_argument('--backend', default='joint', choices=['joint', 'compare', 'fallback'],
                        help='joint: v2 shared solver (default); compare: joint + select/joint '
                             'baselines; fallback: v1 confidence-gated fusion module')
    parser.add_argument('--mode', default='joint_refine',
                        choices=['select', 'joint', 'joint_refine'],
                        help='solver mode when --backend joint')
    parser.add_argument('--max_sync_delta_s', type=float, default=0.05)
    parser.add_argument('--limit', type=int, default=0, help='evaluate only the first N frames')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--eps_t_m', type=float, default=0.5)
    parser.add_argument('--eps_R_deg', type=float, default=2.0)
    parser.add_argument('--coordinate_precision', choices=['amp', 'fp32_head'], default=None)
    parser.add_argument('--valid_mask', default='', help='Geometric valid-FOV mask for correspondence output')
    parser.add_argument('--correspondence_confidence', default='', help='Head-matched confidence model folder')
    return parser.parse_args()


def calibration_chain(calibration_root, camera_number, body_to_lb3_ssc_deg, image_size_hw):
    try:
        from .make_glace_scene import calibration_chain as _chain
    except ImportError:
        from make_glace_scene import calibration_chain as _chain
    return _chain(calibration_root, camera_number, body_to_lb3_ssc_deg, image_size_hw)


def pose_errors(T, gt):
    if T is None:
        return None, None
    dt, dr = pose_distance(T, gt)
    return float(dt), float(np.rad2deg(dr))


def summarize(records, key, eps_t_m, eps_R_deg):
    """coverage / failure rate among accepted / success rate / error stats."""
    n = len(records)
    with_pose = [r for r in records if r[key]['has_pose']]
    errs = np.array([r[key]['err'] for r in with_pose], dtype=float)
    failures = int(np.sum((errs[:, 0] >= eps_t_m) | (errs[:, 1] >= eps_R_deg))) if len(errs) else 0
    agg = None
    if len(errs):
        agg = {'mean_t': float(errs[:, 0].mean()), 'median_t': float(np.median(errs[:, 0])),
               'mean_r_deg': float(errs[:, 1].mean()), 'median_r_deg': float(np.median(errs[:, 1]))}
    return {
        'n_frames': n, 'n_accepted': len(with_pose),
        'coverage': len(with_pose) / n if n else None,
        'failure_rate_accepted': failures / len(with_pose) if len(with_pose) else None,
        'localization_success_rate': (len(with_pose) - failures) / n if n else None,
        'errors': agg,
    }


def main():
    args = parse_args()
    mask_path = getattr(args, 'valid_mask', '')
    confidence_folder = getattr(args, 'correspondence_confidence', '')
    if confidence_folder and not mask_path:
        raise ValueError('Confidence filtering requires the geometric FOV mask')
    if mask_path and args.pose_backend != 'none':
        raise ValueError('Filtered correspondence requires --pose_backend none')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    K_raw, T_BC = calibration_chain(args.camera_root, args.camera_number,
                                    args.body_to_lb3_ssc_deg, None)
    rows = [r for r in camera_rows(args.camera_root, args.camera_number)
            if r['sequence'] in TEST_DATES]
    missing = set(TEST_DATES) - {r['sequence'] for r in rows}
    if missing and not args.allow_partial:
        raise SystemExit('Missing test camera sequences: ' + ', '.join(sorted(missing))
                         + '; use --allow_partial to report a partial experiment')
    if not rows:
        raise SystemExit('No test camera images available')
    cam_ts = np.array([r['timestamp_us'] for r in rows], dtype=np.int64)
    trajectories = {date: NCLTTrajectory(trajectory_path(args.dataset_folder, date))
                    for date in {r['sequence'] for r in rows}}
    pool_files = sorted(Path(args.pool_dir).rglob('*.npz'))
    if args.limit:
        pool_files = pool_files[:args.limit]
    if not pool_files:
        raise SystemExit('No pool exports found under ' + str(args.pool_dir))

    contract = resolve_contract(args.glace_head, args.image_resolution)
    args.image_resolution = contract['image_resolution']
    rgb_session = None
    if contract['global_feature_protocol'] == RGB_PROTOCOL or args.pose_backend != 'opencv' or args.coordinate_precision is not None:
        rgb_session = InferenceSession(args.vendor_dir, args.glace_head, args.deit_checkpoint,
            split=args.feature_split, encoder=args.glace_encoder or None, T_BC=T_BC,
            resolution=args.image_resolution, pose_backend=args.pose_backend, coordinate_precision=args.coordinate_precision)
        contract = rgb_session.contract
    else:
        feature_fn = None
        if args.deit_checkpoint:
            feature_fn = deit_global_feature_fn(args.vendor_dir, args.deit_checkpoint)
        adapter = GLACEAdapter(args.vendor_dir, args.glace_head,
                               encoder_path=args.glace_encoder or None, T_BC=T_BC,
                               global_feature_fn=feature_fn)

    correspondence_filter = None
    if mask_path:
        from .correspondence_filter import CorrespondenceFilter
        if confidence_folder and contract['coordinate_precision'] != 'fp32_head':
            raise ValueError('Confidence filtering requires FP32 head inference')
        correspondence_filter = CorrespondenceFilter(mask_path, args.glace_head, confidence_folder or None)
    solver_cfg = JointSolverConfig.load(args.solver_config) if args.solver_config else JointSolverConfig()
    fusion_cfg = replace(FusionConfig(), max_sync_delta_s=args.max_sync_delta_s)
    fallback_lidar_conf = IsotonicCalibrator()
    fallback_camera_conf = IsotonicCalibrator()

    modes = ['select', 'joint', 'joint_refine'] if args.backend == 'compare' else (
        ['fallback'] if args.backend == 'fallback' else [args.mode])
    records = []
    skipped_sync = 0
    skipped_gt = 0
    all_lidar_records = []

    for pool_file in pool_files:
        export = dict(np.load(pool_file, allow_pickle=False))
        scan_ts = int(export['scan_timestamp_us'])
        all_lidar_records.append({'result': {'has_pose': True, 'status': 'BASELINE',
            'err': list(pose_errors(export['T_WB'], export['T_WB_gt']))}})
        j = int(np.argmin(np.abs(cam_ts - scan_ts)))
        sync_delta_s = (int(cam_ts[j]) - scan_ts) / 1e6
        if abs(sync_delta_s) > args.max_sync_delta_s:
            skipped_sync += 1
            continue

        row = rows[j]
        trajectory = trajectories[row['sequence']]
        try:
            gt, camera_gt = trajectory.at([scan_ts, int(cam_ts[j])])
        except ValueError:
            skipped_gt += 1
            continue
        pool_l = lidar_pool_from_export(export)
        T_L = np.asarray(export['T_WB'], dtype=float)
        q_L = lidar_support_rate(T_L, pool_l['p_body'], pool_l['p_world'], args.lidar_threshold_m)
        q_C = None

        image_path = Path(args.camera_root) / row['saved_path']
        K_stored, _ = stored_intrinsics(K_raw, row, image_path)
        if rgb_session is not None:
            glace = rgb_session.infer(image_path, K_stored)
            K = glace.K
        else:
            image, K = preprocess_image(image_path, K_stored, args.image_resolution)
            glace = adapter.infer(image, K)
        if correspondence_filter is not None:
            glace = correspondence_filter.apply(glace)
        T_C = glace.T_WB
        if T_C is not None:
            q_C = camera_support_rate(T_C, T_BC, K, glace.uv, glace.xyz_world,
                                      args.camera_threshold_px)

        frame_id = pool_file.stem
        record = {
            'frame': frame_id, 'scan_ts': scan_ts, 'image_ts': int(cam_ts[j]),
            'sync_delta_s': sync_delta_s,
            'q_L': q_L, 'q_C': q_C, 'glace_inliers': glace.inlier_count,
            'correspondence_filter': getattr(glace, 'diagnostics', {}).get('correspondence_filter'),
            'lidar_err': list(pose_errors(T_L, gt)),
            'camera_err': list(pose_errors(T_C, camera_gt)),
            'sequence': row['sequence'],
            'evaluation_gt': 'linear_translation_slerp_at_each_sensor_timestamp',
        }

        if args.backend == 'fallback':
            lidar_stamp, camera_stamp = make_evidence_stamps(frame_id, scan_ts / 1e6,
                                                             int(cam_ts[j]) / 1e6)
            evidence = make_fusion_evidence(pool_l['p_body'], pool_l['p_world'], glace.uv,
                                            glace.xyz_world, K, T_BC, lidar_stamp,
                                            camera_stamp, camera_inlier_mask=glace.inlier_mask)
            c_L = float(np.asarray(fallback_lidar_conf(q_L)))
            c_C = None if q_C is None else float(np.asarray(fallback_camera_conf(q_C)))
            result = localize(T_L, c_L, T_C, c_C, evidence=evidence, config=fusion_cfg)
            e_F, r_F = pose_errors(result.pose, gt)
            record['result'] = {'status': result.status, 'source': result.source,
                                'reason': result.reason, 'has_pose': result.pose is not None,
                                'err': [e_F, r_F]}
        else:
            problem = JointProblem(pool_l['p_body'], pool_l['p_world'], pool_l['u'],
                                   glace.uv, glace.xyz_world, K, T_BC, solver_cfg)
            seedwise = export.get('seedwise_T_WB')
            per_mode = {}
            for mode in modes:
                result = solve(problem, T_L, T_C, seedwise, mode=mode)
                e_F, r_F = pose_errors(result.pose, gt)
                per_mode[mode] = {'status': result.status, 'source': result.source,
                                  'modality': result.modality, 'reason': result.reason,
                                  'has_pose': result.pose is not None, 'err': [e_F, r_F]}
            record['modes'] = per_mode
            if len(modes) == 1:
                record['result'] = per_mode[modes[0]]

        records.append(record)

    # ---- report ----------------------------------------------------------
    report = {
        'inference_contract': contract,
        'correspondence_filter': None if correspondence_filter is None else dict(
            mask_sha256=correspondence_filter.mask_hash, confidence_protocol=correspondence_filter.protocol),
        'n_frames': len(records), 'n_skipped_sync': skipped_sync,
        'n_input_lidar_frames': len(pool_files), 'n_skipped_gt_range': skipped_gt,
        'synchronized_fraction': len(records) / len(pool_files),
        'missing_camera_sequences': sorted(missing),
        'evaluation_scope': 'synchronized_camera_subset',
        'max_sync_delta_s': args.max_sync_delta_s,
        'motion_model': 'simultaneous_within_gate; no GT motion compensation in solver',
        'image_resolution': args.image_resolution,
        'backend': args.backend, 'modes': modes,
        'success_eps': [args.eps_t_m, args.eps_R_deg],
        'lidar_threshold_m': args.lidar_threshold_m,
        'camera_threshold_px': args.camera_threshold_px,
        'solver_config': {k: getattr(solver_cfg, k) for k in solver_cfg.__dataclass_fields__},
    }

    def errors_agg(records, key):
        errs = np.array([r[key]['err'] for r in records if r[key]['err'][0] is not None])
        return None if not len(errs) else {
            'mean_t': float(errs[:, 0].mean()), 'median_t': float(np.median(errs[:, 0])),
            'mean_r_deg': float(errs[:, 1].mean()), 'median_r_deg': float(np.median(errs[:, 1]))}

    def summarize_errors(records, key):
        n = len(records)
        accepted = [r for r in records if r[key]['err'][0] is not None]
        errs = np.array([r[key]['err'] for r in accepted])
        failures = int(np.sum((errs[:, 0] >= args.eps_t_m) | (errs[:, 1] >= args.eps_R_deg))) \
            if len(errs) else 0
        statuses = [r[key]['status'] for r in records]
        return {
            'n_frames': n, 'n_accepted': len(accepted),
            'coverage': len(accepted) / n if n else None,
            'failure_rate_accepted': failures / len(accepted) if len(accepted) else None,
            'localization_success_rate': (len(accepted) - failures) / n if n else None,
            'errors': errors_agg(records, key),
            'status_counts': {s: statuses.count(s) for s in sorted(set(statuses))},
        }

    report['leader_all_input_export_gt'] = summarize(
        all_lidar_records, 'result', args.eps_t_m, args.eps_R_deg)
    report['leader_baseline'] = summarize_errors(
        [{'lidar_err': {'err': r['lidar_err'], 'status': 'BASELINE'}} for r in records],
        'lidar_err')
    report['glace_baseline'] = summarize_errors(
        [{'camera_err': {'err': r['camera_err'], 'status': 'BASELINE'}} for r in records],
        'camera_err')

    if args.backend == 'compare':
        for mode in modes:
            report[mode] = summarize_errors(
                [{'result': r['modes'][mode]} for r in records],
                'result')
    else:
        report['solver'] = summarize_errors(
            [{'result': {'err': r['result']['err'], 'status': r['result']['status']}}
             for r in records],
            'result')

    (out_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out_dir / 'records.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
