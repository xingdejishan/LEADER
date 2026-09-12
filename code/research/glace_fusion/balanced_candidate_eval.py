import argparse
import json
from pathlib import Path

import numpy as np

from .joint_solver import lidar_reliability_weights
from .packet import lidar_pool_from_export
from .real_candidate_eval import pose_errors, summarize


def lidar_scores(poses, body, world, weights):
    scores = []
    for start in range(0, len(poses), 32):
        p = poses[start:start + 32]
        transformed = np.einsum('nj,bkj->bnk', body, p[:, :3, :3]) + p[:, None, :3, 3]
        residual = np.sum((transformed - world[None]) ** 2, axis=2) / .3 ** 2
        scores.extend((np.minimum(residual, 1) @ weights).tolist())
    return np.asarray(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    protocol = dict(lidar_scale_m=.3, camera_scale_px=10, modality_weights=[.5,.5],
        lidar_weights='existing joint_solver.lidar_reliability_weights, trr_scale=10',
        score='0.5 * clipped weighted LiDAR squared residual + 0.5 * fixed camera S_C',
        methods=['v1_two_stage', 'lidar_rank', 'camera_improved', 'balanced_stage1', 'balanced_improved', 'oracle'],
        purpose='fixed candidate-scoring ablation, not a new refinement or calibrated deployment policy', tuning=False)
    declaration = args.root / 'balanced_protocol.json'
    if declaration.exists():
        if json.loads(declaration.read_text()) != protocol:
            raise ValueError('Balanced protocol changed')
    elif args.declare_only:
        declaration.write_text(json.dumps(protocol, indent=2))
    else:
        raise ValueError('Declare the fixed scoring protocol before evaluating results')
    if args.declare_only:
        return
    root = args.root / 'real_candidates'
    report = json.loads((root / 'report.json').read_text())
    if not report['complete']:
        raise ValueError('Real candidate export is incomplete')
    records = [json.loads(line) for line in (root / 'records.jsonl').read_text().splitlines()]
    for row in records:
        with np.load(root / 'pools' / (row['image'] + '.npz')) as data:
            pool = lidar_pool_from_export(data)
            scores = lidar_scores(data['candidate_T_WB'], pool['p_body'], pool['p_world'],
                lidar_reliability_weights(data['u_pred_all']))
            alternatives = dict(lidar_rank=scores,
                balanced_stage1=.5 * (scores + data['stage1_scores']),
                balanced_improved=.5 * (scores + data['improved_scores']))
            errors = pose_errors(data['candidate_T_WB'], data['GT'])
            for name, values in alternatives.items():
                best = int(np.flatnonzero(values <= values.min() + 1e-8)[0])
                row['selected'][name] = best
                row['errors'][name] = errors[best].tolist()
    report['balanced_protocol'] = protocol
    methods = ['leader', 'v1_two_stage', 'lidar_rank', 'camera_stage1', 'camera_improved',
               'balanced_stage1', 'balanced_improved', 'oracle']
    report['groups'] = summarize(records, methods)
    (args.root / 'balanced_report.json').write_text(json.dumps(report, indent=2))
    (args.root / 'balanced_records.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(report['groups'], indent=2))


if __name__ == '__main__':
    main()
