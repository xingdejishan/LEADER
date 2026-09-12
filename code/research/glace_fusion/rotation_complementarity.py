import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .joint_solver import lidar_reliability_weights
from .pose_boundary import solver_pose
from .real_candidate_eval import pose_errors, score_camera


def rotation_candidates(initial):
    offsets = [(0, 0, 0)] + [v for v in itertools.product((-2, -1, 0, 1, 2), repeat=3) if any(v)]
    poses = np.repeat(initial[None], len(offsets), axis=0)
    poses[:, :3, :3] = initial[:3, :3] @ Rotation.from_rotvec(np.deg2rad(offsets)).as_matrix()
    return poses


def evidence(poses, pool, camera, extrinsic):
    correction = pool['T_corr']
    body = (pool['c_local_all'].astype(float) - correction[:3, 3]) @ correction[:3, :3]
    world = pool['c_pred_all'].astype(float) + pool['center_t']
    weights = lidar_reliability_weights(pool['u_pred_all'])
    lidar = np.array([np.minimum(np.sum((body @ p[:3, :3].T + p[:3, 3] - world) ** 2, axis=1) / .3 ** 2, 1) @ weights for p in poses])
    xyz, uv, K = [camera[k] for k in ('xyz', 'uv', 'K')]
    all_camera = score_camera(poses, extrinsic, xyz, uv, K)
    initial_camera = poses[0] @ extrinsic
    depth = ((xyz - initial_camera[:3, 3]) @ initial_camera[:3, :3])[:, 2]
    positive = np.isfinite(depth) & (depth > 0)
    far = positive & (depth >= np.median(depth[positive])) if positive.any() else positive
    far_camera = score_camera(poses, extrinsic, xyz[far], uv[far], K) if far.sum() >= 6 else all_camera.copy()
    return dict(lidar=lidar, camera=all_camera, joint=(lidar + all_camera) / 2,
                far_camera=far_camera, far_joint=(lidar + far_camera) / 2), int(far.sum())


def summary(records):
    result = {}
    for group, subset in [('all', records), ('supported', [r for r in records if r['in_orientation_support']]),
                          ('outside', [r for r in records if not r['in_orientation_support']])]:
        if not subset:
            continue
        base = np.array([r['errors']['baseline'] for r in subset])
        base_ok = (base[:, 0] < 1) & (base[:, 1] < 2)
        methods = {}
        for name in subset[0]['errors']:
            errors = np.array([r['errors'][name] for r in subset])
            ok = (errors[:, 0] < 1) & (errors[:, 1] < 2)
            methods[name] = dict(success_1m_2deg=int(ok.sum()),
                success_05m_1deg=int(((errors[:, 0] < .5) & (errors[:, 1] < 1)).sum()),
                median_rotation_deg=float(np.median(errors[:, 1])),
                mean_rotation_deg=float(errors[:, 1].mean()),
                rescued=int((~base_ok & ok).sum()), harmed=int((base_ok & ~ok).sum()))
        result[group] = dict(frames=len(subset), methods=methods)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--variant', choices=['selected', 'balanced', 'previous'], default='selected')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = dict(candidate_offsets_deg=[-2, -1, 0, 1, 2], candidate_count=125,
        rotation_convention='right body-frame rotation-vector increment; body translation fixed',
        scales=dict(lidar_m=.3, camera_px=10), modality_weights=[.5, .5],
        far_selection='upper half of positive predicted depths at baseline; frozen across candidates',
        gt_usage='evaluation and explicitly labelled oracle only; not candidate construction or selection',
        scope='previously inspected test set; fixed diagnostic ablations, not deployment acceptance or tuning',
        selection='minimum truncated score; ties retain baseline first',
        variant=args.variant, changes_to_frontend=False)
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows = json.loads((args.bundle / 'data/test_rows.json').read_text())[:args.limit or None]
    meta = json.loads((args.bundle / 'data/train_scene/scene_meta.json').read_text())
    extrinsic = solver_pose(np.asarray(meta['T_BC_camera_to_body']))
    records = []
    with (args.out / 'records.jsonl').open('w', buffering=1) as output:
        for row in rows:
            with np.load(args.bundle / 'cache/lidar_pools' / (row['image'] + '.npz')) as pool, np.load(
                    args.bundle / 'cache' / args.variant / 'coordinates' / (row['image'] + '.npz')) as camera:
                poses = rotation_candidates(solver_pose(pool['v1_two_stage']))
                scores, far_count = evidence(poses, pool, camera, extrinsic)
                selected = dict(baseline=0, **{k: int(np.argmin(v)) for k, v in scores.items()})
                errors = pose_errors(poses, pool['GT'])
                selected['oracle_rotation_only'] = int(np.argmin(errors[:, 1]))
                record = dict(image=row['image'], in_orientation_support=row['in_orientation_support'],
                    selected=selected, errors={k: errors[i].tolist() for k, i in selected.items()},
                    far_count=far_count, camera_count=len(camera['uv']))
                output.write(json.dumps(record) + '\n')
                records.append(record)
                if len(records) % 20 == 0:
                    print(json.dumps(dict(frames=len(records), total=len(rows))), flush=True)
    report = summary(records)
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['all'], indent=2))


if __name__ == '__main__':
    main()
