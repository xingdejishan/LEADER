import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .balanced_candidate_eval import lidar_scores
from .joint_solver import lidar_reliability_weights
from .packet import lidar_pool_from_export
from .real_candidate_eval import dominance_counts, pose_errors, score_camera, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    output = args.out_dir or args.root / 'same_v1_refinement'
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
    from full_pool_robust_v1 import full_pool_refine
    E = np.asarray(json.loads((args.scene / 'scene_meta.json').read_text())['T_BC_camera_to_body'])
    root = args.root / 'real_candidates'
    records = [json.loads(line) for line in (root / 'records.jsonl').read_text().splitlines()]
    manifest = dict(protocol='rerank raw SC2 candidates, then apply the original v1 full-pool refinement once',
        raw_pool='original LEADER pose first plus seedwise poses; no appended already-refined v1 pose',
        refinement_thresholds_m=[1.2,.6], camera_score_threshold_px=10, lidar_score_threshold_m=.3,
        joint_weights=[.5,.5], tie_preference='original LEADER raw pose',
        nature='follow-up diagnostic after direct reranking; no parameter search or training',
        oracle='best refined pose among all raw seeds, GT only used to compute this diagnostic bound')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    methods = ['v1_two_stage', 'v1_reproduced', 'lidar_rank_then_v1', 'camera_stage1_then_v1',
               'camera_improved_then_v1', 'balanced_stage1_then_v1', 'balanced_improved_then_v1', 'oracle_refined']
    max_translation_delta, max_rotation_element_delta = 0., 0.
    with torch.inference_mode():
        for row in records:
            with np.load(root / 'pools' / (row['image'] + '.npz')) as data:
                raw = np.concatenate([data['leader'][None], data['candidate_T_WB'][1:]])
                _, unique = np.unique(np.round(raw.reshape(len(raw), -1), 5), axis=0, return_index=True)
                raw = raw[np.sort(unique)]
                pool = lidar_pool_from_export(data)
                scores_l = lidar_scores(raw, pool['p_body'], pool['p_world'], lidar_reliability_weights(data['u_pred_all']))
                source = torch.from_numpy(data['c_local_all']).float().cuda()
                target = torch.from_numpy(data['c_pred_all']).float().cuda()
                correction = data['T_corr'].astype(np.float64)
                center = data['center_t'].astype(np.float64)
                refined = []
                for pose in raw:
                    local = pose @ np.linalg.inv(correction)
                    local[:3, 3] -= center
                    transformed, _ = full_pool_refine(torch.from_numpy(local).float().cuda(), source, target)
                    world = transformed.cpu().numpy().astype(np.float64)
                    world[:3, 3] += center
                    refined.append(world @ correction)
                refined = np.asarray(refined)
                delta_t = float(np.linalg.norm(refined[0, :3, 3] - data['v1_two_stage'][:3, 3]))
                delta_R = float(np.max(np.abs(refined[0, :3, :3] - data['v1_two_stage'][:3, :3])))
                max_translation_delta = max(max_translation_delta, delta_t)
                max_rotation_element_delta = max(max_rotation_element_delta, delta_R)
                if delta_t > .001 or delta_R > 1e-4:
                    raise ValueError('Original v1 refinement did not reproduce within floating-point tolerance')
                gt = data['GT']
                errors = pose_errors(refined, gt)
                raw_errors = pose_errors(raw, gt)
                row['errors']['v1_reproduced'] = errors[0].tolist()
                choices = dict(lidar_rank_then_v1=scores_l)
                for label in ['stage1', 'improved']:
                    with np.load(args.root / label / 'coordinates' / (row['image'] + '.npz')) as camera:
                        scores_c = score_camera(raw, E, camera['xyz'], camera['uv'], camera['K'])
                    choices['camera_' + label + '_then_v1'] = scores_c
                    choices['balanced_' + label + '_then_v1'] = .5 * (scores_l + scores_c)
                    row['dominance'][label] = dominance_counts(raw_errors, scores_c)
                for name, scores in choices.items():
                    chosen = int(np.flatnonzero(scores <= scores.min() + 1e-8)[0])
                    row['selected'][name] = chosen
                    row['errors'][name] = errors[chosen].tolist()
                oracle = int(np.argmin(np.maximum(errors[:, 0], errors[:, 1] / 2)))
                row['errors']['oracle_refined'] = errors[oracle].tolist()
                np.savez_compressed(output / (row['image'] + '.npz'), raw=raw, refined=refined, errors=errors, **choices)
    report = dict(manifest=manifest, groups=summarize(records, methods), complete=True,
        v1_reproduction_max_translation_delta_m=max_translation_delta,
        v1_reproduction_max_rotation_element_delta=max_rotation_element_delta)
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    (output / 'records.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
