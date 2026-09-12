import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .joint_solver import JointProblem, JointSolverConfig, solve
from .pose_boundary import solver_pose
from .real_candidate_eval import pose_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinates', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    base = Path('/root/rivermind-data')
    stage3 = base / 'glace_stage3_region_full_20260912'
    rows_path = args.bundle / 'data/test_rows.json' if args.bundle else stage3 / 'rows.json'
    meta_path = args.bundle / 'data/train_scene/scene_meta.json' if args.bundle else base / 'glace_nclt_rgb_large_20260912/scene/scene_meta.json'
    pool_dir = args.bundle / 'cache/lidar_pools' if args.bundle else stage3 / 'real_candidates/pools'
    rows = json.loads(rows_path.read_text())[:args.limit or None]
    meta = json.loads(meta_path.read_text())
    E = solver_pose(np.asarray(meta['T_BC_camera_to_body']))
    cfg = JointSolverConfig(camera_scale_px=10.0)
    (args.out / 'protocol.json').write_text(json.dumps(dict(config=asdict(cfg),
        candidates='v1-two-stage + cached real SC2 seeds + fixed 256 P3P samples',
        acceptance='unchanged existing shared solver; rejected or single-modal outputs fall back to v1',
        weights='existing modality-balanced objective; uniform per-camera-point weights',
        gt_usage='error reporting only; never input to solver',
        frozen_before_new_head_test=True), indent=2))
    records = []
    started = time.time()
    with (args.out / 'records.jsonl').open('w', buffering=1) as output:
        for row in rows:
            pool = np.load(pool_dir / (row['image'] + '.npz'))
            camera = np.load(args.coordinates / (row['image'] + '.npz'))
            Q = pool['T_corr']
            local = pool['c_local_all'].astype(float)
            body = (local - Q[:3, 3]) @ Q[:3, :3]
            world = pool['c_pred_all'].astype(float) + pool['center_t']
            initial = solver_pose(pool['v1_two_stage'])
            problem = JointProblem(body, world, pool['u_pred_all'], camera['uv'], camera['xyz'], camera['K'], E, cfg)
            result = solve(problem, T_L=initial,
                seedwise_T_WB=np.array([solver_pose(T) for T in pool['candidate_T_WB']]), mode='joint_refine')
            accepted = result.status == 'JOINT' and result.pose is not None
            pose = result.pose if accepted else initial
            record = dict(row, accepted=accepted, status=result.status, source=result.source,
                reason=result.reason, baseline_error=pose_errors(initial[None], pool['GT'])[0].tolist(),
                error=pose_errors(pose[None], pool['GT'])[0].tolist(),
                diagnostics=result.diagnostics)
            records.append(record)
            output.write(json.dumps(record) + '\n')
            if len(records) % 10 == 0:
                print(json.dumps(dict(frames=len(records), total=len(rows), elapsed=time.time()-started)), flush=True)
    groups = dict(all=lambda r: True, supported=lambda r: r['in_orientation_support'],
        outside=lambda r: not r['in_orientation_support'],
        new_supported=lambda r: r['in_orientation_support'] and not r['previous_probe'])
    report = {}
    for group, keep in groups.items():
        subset = [r for r in records if keep(r)]
        entry = dict(frames=len(subset), accepted=sum(r['accepted'] for r in subset))
        if not subset:
            report[group] = entry
            continue
        for key in ['baseline_error', 'error']:
            error = np.array([r[key] for r in subset])
            entry[key] = dict(median_t=float(np.median(error[:, 0])), median_r=float(np.median(error[:, 1])),
                mean_t=float(error[:, 0].mean()), mean_r=float(error[:, 1].mean()),
                success_1m_2deg=int(((error[:, 0] < 1) & (error[:, 1] < 2)).sum()),
                success_05m_1deg=int(((error[:, 0] < .5) & (error[:, 1] < 1)).sum()))
        baseline_success = np.array([r['baseline_error'][0] < 1 and r['baseline_error'][1] < 2 for r in subset])
        success = np.array([r['error'][0] < 1 and r['error'][1] < 2 for r in subset])
        entry.update(rescued=int((~baseline_success & success).sum()), harmed=int((baseline_success & ~success).sum()))
        report[group] = entry
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
