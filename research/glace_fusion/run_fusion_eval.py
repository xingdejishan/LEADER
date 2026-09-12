"""Joint LiDAR (LEADER) + Camera (GLACE) fusion evaluation.

Stage A (LEADER side) is run separately and offline:
    python run_mink.py --mode test --dataset NCLT --export_fusion_pool <DIR> ...
which stores, per query scan and WITHOUT changing LEADER's own pipeline:
    the full pre-top-50% correspondence pool (c_local_all, c_pred_all, u_pred_all),
    T_corr / center_t (frame bookkeeping), the final T_WB, GT T_WB, scan timestamp,
    and (optionally) pose-diverse SC2-PCR seedwise hypotheses in T_WB form.

Stage B (this script) attaches the GLACE camera branch and the fixed fusion
module, then reports per-modality and fused accuracy:
    python -m research.glace_fusion.run_fusion_eval --pool_dir <DIR> ...
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    from .lidar_camera_fusion import FusionConfig, evaluate_results, localize, pose_distance
    from .packet import (IsotonicCalibrator, camera_support_rate, diverse_poses,
                         lidar_pool_from_export, lidar_support_rate, make_evidence_stamps,
                         make_fusion_evidence)
    from .glace_adapter import GLACEAdapter, deit_global_feature_fn
except ImportError:  # running with the package directory itself on sys.path
    from lidar_camera_fusion import FusionConfig, evaluate_results, localize, pose_distance
    from packet import (IsotonicCalibrator, camera_support_rate, diverse_poses,
                        lidar_pool_from_export, lidar_support_rate, make_evidence_stamps,
                        make_fusion_evidence)
    from glace_adapter import GLACEAdapter, deit_global_feature_fn


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
    parser.add_argument('--image_size', type=int, nargs=2, default=(616, 808))
    parser.add_argument('--lidar_threshold_m', type=float, default=0.3)
    parser.add_argument('--camera_threshold_px', type=float, default=4.0)
    parser.add_argument('--lidar_conf', default='', help='isotonic calibration json for q_L')
    parser.add_argument('--camera_conf', default='', help='isotonic calibration json for q_C')
    parser.add_argument('--fusion_config', default='', help='FusionConfig.save() json')
    parser.add_argument('--max_sync_delta_s', type=float, default=0.05)
    parser.add_argument('--use_extra_hypotheses', action='store_true',
                        help='feed pose-diverse SC2-PCR seedwise poses into the fallback')
    parser.add_argument('--seedwise_max', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='evaluate only the first N frames')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--eps_t_m', type=float, default=0.5)
    parser.add_argument('--eps_R_deg', type=float, default=2.0)
    return parser.parse_args()


def camera_index(camera_root, camera_number):
    import csv
    rows = []
    with open(Path(camera_root) / 'all_images.csv') as handle:
        for row in csv.DictReader(handle):
            if row['camera'] == 'Cam%d' % camera_number:
                rows.append((int(row['group_target_timestamp']), row['saved_path']))
    rows.sort()
    return np.array([r[0] for r in rows]), [r[1] for r in rows]


def calibration_chain(calibration_root, camera_number, body_to_lb3_ssc_deg, image_size_hw):
    try:
        from .make_glace_scene import calibration_chain as _chain
    except ImportError:
        from make_glace_scene import calibration_chain as _chain
    return _chain(calibration_root, camera_number, body_to_lb3_ssc_deg, image_size_hw)


def load_image(path, image_size_hw):
    from PIL import Image
    with Image.open(path) as im:
        if im.size != (image_size_hw[1], image_size_hw[0]):
            im = im.resize((image_size_hw[1], image_size_hw[0]), Image.BILINEAR)
        return np.asarray(im.convert('L'), dtype=np.float32) / 255.0


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = FusionConfig.load(args.fusion_config) if args.fusion_config else FusionConfig()
    config = replace(config, max_sync_delta_s=args.max_sync_delta_s)
    lidar_conf = IsotonicCalibrator.load(args.lidar_conf) if args.lidar_conf else IsotonicCalibrator()
    camera_conf = IsotonicCalibrator.load(args.camera_conf) if args.camera_conf else IsotonicCalibrator()

    image_size_hw = tuple(args.image_size)
    K, T_BC = calibration_chain(args.camera_root, args.camera_number,
                                args.body_to_lb3_ssc_deg, image_size_hw)

    feature_fn = None
    if args.deit_checkpoint:
        feature_fn = deit_global_feature_fn(args.vendor_dir, args.deit_checkpoint, image_size_hw)
    adapter = GLACEAdapter(args.vendor_dir, args.glace_head,
                           encoder_path=args.glace_encoder or None, T_BC=T_BC,
                           global_feature_fn=feature_fn)

    cam_ts, cam_paths = camera_index(args.camera_root, args.camera_number)
    pool_files = sorted(Path(args.pool_dir).rglob('*.npz'))
    if args.limit:
        pool_files = pool_files[:args.limit]
    if not pool_files:
        raise SystemExit('No pool exports found under ' + str(args.pool_dir))

    records = []
    results, lidar_poses, camera_poses, gt_poses = [], [], [], []
    skipped_sync = 0
    for pool_file in pool_files:
        export = dict(np.load(pool_file, allow_pickle=False))
        scan_ts = int(export['scan_timestamp_us'])
        j = int(np.argmin(np.abs(cam_ts - scan_ts)))
        sync_delta_s = (int(cam_ts[j]) - scan_ts) / 1e6
        if abs(sync_delta_s) > args.max_sync_delta_s:
            skipped_sync += 1
            continue

        pool_l = lidar_pool_from_export(export)
        T_L = np.asarray(export['T_WB'], dtype=float)
        q_L = lidar_support_rate(T_L, pool_l['p_body'], pool_l['p_world'], args.lidar_threshold_m)
        c_L = float(np.asarray(lidar_conf(q_L)))

        image = load_image(Path(args.camera_root) / cam_paths[j], image_size_hw)
        glace = adapter.infer(image, K)
        T_C = glace.T_WB
        if T_C is not None:
            q_C = camera_support_rate(T_C, T_BC, K, glace.uv, glace.xyz_world,
                                      args.camera_threshold_px)
            c_C = float(camera_conf(np.array([q_C]))[0])
        else:
            q_C, c_C = None, None

        frame_id = pool_file.stem
        lidar_stamp, camera_stamp = make_evidence_stamps(
            frame_id, scan_ts / 1e6, int(cam_ts[j]) / 1e6)
        camera_mask = glace.inlier_mask
        evidence = make_fusion_evidence(
            pool_l['p_body'], pool_l['p_world'], glace.uv, glace.xyz_world, K, T_BC,
            lidar_stamp, camera_stamp, camera_inlier_mask=camera_mask)

        extras = None
        if args.use_extra_hypotheses and 'seedwise_T_WB' in export:
            extras = diverse_poses(export['seedwise_T_WB'], args.seedwise_max)

        result = localize(T_L, c_L, T_C, c_C, evidence=evidence, config=config,
                          extra_hypotheses=extras)

        gt = np.asarray(export['T_WB_gt'], dtype=float)
        def err(T):
            if T is None:
                return None, None
            dt, dr = pose_distance(T, gt)
            return float(dt), float(np.rad2deg(dr))
        e_L, r_L = err(T_L)
        e_C, r_C = err(T_C)
        e_F, r_F = err(result.pose)

        records.append({
            'frame': frame_id, 'scan_ts': scan_ts, 'image_ts': int(cam_ts[j]),
            'sync_delta_s': sync_delta_s,
            'q_L': q_L, 'c_L': c_L, 'q_C': q_C, 'c_C': c_C,
            'glace_inliers': glace.inlier_count,
            'lidar_err': [e_L, r_L], 'camera_err': [e_C, r_C], 'fusion_err': [e_F, r_F],
            'fusion_status': result.status, 'fusion_source': result.source,
            'fusion_reason': result.reason,
            'e_t': e_F if e_F is not None else e_L, 'e_r': r_F if r_F is not None else r_L,
            'has_pose': result.pose is not None,
        })
        results.append(result)
        lidar_poses.append(T_L)
        camera_poses.append(T_C if T_C is not None else np.eye(4))
        gt_poses.append(gt)

    summary = evaluate_results(results, gt_poses, eps_t_m=args.eps_t_m,
                               eps_R_rad=np.deg2rad(args.eps_R_deg))
    lidar_errs = np.array([r['lidar_err'] for r in records if r['lidar_err'][0] is not None])
    camera_errs = np.array([r['camera_err'] for r in records if r['camera_err'][0] is not None])
    fusion_errs = np.array([r['fusion_err'] for r in records if r['fusion_err'][0] is not None])

    def agg(a):
        return None if not len(a) else {
            'mean_t': float(a[:, 0].mean()), 'median_t': float(np.median(a[:, 0])),
            'mean_r_deg': float(a[:, 1].mean()), 'median_r_deg': float(np.median(a[:, 1]))}

    report = {
        'n_frames': len(records), 'n_skipped_sync': skipped_sync,
        'lidar_threshold_m': args.lidar_threshold_m,
        'camera_threshold_px': args.camera_threshold_px,
        'confidence_calibrated': {'lidar': lidar_conf.calibrated, 'camera': camera_conf.calibrated},
        'success_eps': [args.eps_t_m, args.eps_R_deg],
        'leader_baseline': agg(lidar_errs),
        'glace_baseline': agg(camera_errs),
        'fusion': agg(fusion_errs),
        'fusion_evaluate_results': summary,
        'status_counts': {},
    }
    for r in records:
        report['status_counts'][r['fusion_status']] = report['status_counts'].get(r['fusion_status'], 0) + 1

    (out_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez_compressed(out_dir / 'records.npz',
                        lidar=np.array(lidar_poses), camera=np.array(camera_poses),
                        gt=np.array(gt_poses))
    (out_dir / 'records.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
