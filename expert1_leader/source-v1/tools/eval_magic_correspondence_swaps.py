import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from models.sc2pcr import Matcher
from utils.full_pool_robust_v1 import full_pool_refine


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def load_run(path):
    run = json.loads((path / 'predictions.json').read_text(encoding='utf-8'))
    if not run.get('correspondence_cache'):
        raise ValueError(f'Run has no correspondence cache: {path}')
    for row in run['predictions']:
        cache_path = path / row['correspondences_file']
        if digest(cache_path) != row['correspondences_sha256']:
            raise ValueError(f'Cache hash mismatch: {cache_path}')
    return run


def load_cache(path, row):
    with np.load(path / row['correspondences_file'], allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def alignment(original, changed):
    source = original['voxel_coordinates']
    target = changed['voxel_coordinates']
    if len(source) != len(target) or len({tuple(row) for row in source}) != len(source):
        raise ValueError('Voxel sets are not unique and equal size')
    lookup = {tuple(row): index for index, row in enumerate(target)}
    try:
        indices = np.asarray([lookup[tuple(row)] for row in source], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f'Voxel identity missing: {error}') from error
    if np.max(np.abs(original['input_local_xyz'] - changed['input_local_xyz'][indices])) > 1e-5:
        raise ValueError('Input local coordinates differ after voxel alignment')
    return indices


def solve(matcher, original, changed, order, use_changed_coords, use_changed_reliability,
          center, seed):
    local = torch.from_numpy(original['input_local_xyz']).cuda()
    coords = changed if use_changed_coords else original
    reliability = changed if use_changed_reliability else original
    predicted = torch.from_numpy(coords['predicted_centered_xyz'][order]
                                 if use_changed_coords else coords['predicted_centered_xyz']).cuda()
    scores = torch.from_numpy(reliability['predicted_reliability'][order]
                             if use_changed_reliability else reliability['predicted_reliability']).cuda()
    count = max(min(50, len(scores)), int(0.5 * len(scores)))
    if count < 3:
        raise ValueError('Insufficient correspondences')
    selected = scores.topk(count).indices
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    initial = matcher.estimator(local[selected][None], predicted[selected][None])[0]
    initial_world = initial.clone()
    refined = full_pool_refine(initial, local[selected], predicted[selected])
    refined = refined.clone()
    initial_world[:3, 3] += center
    refined[:3, 3] += center
    if not torch.isfinite(refined).all():
        raise FloatingPointError('Nonfinite swapped pose')
    return initial_world.cpu().tolist(), refined.cpu().tolist(), selected.cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--center_checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    base = load_run(args.baseline)
    candidate = load_run(args.candidate)
    if base['split_sha256'] != candidate['split_sha256'] or base['subset'] != candidate['subset']:
        raise ValueError('Runs use different test denominators')
    if [row['scan'] for row in base['predictions']] != [row['scan'] for row in candidate['predictions']]:
        raise ValueError('Runs do not have identical frame order')
    center_state = torch.load(args.center_checkpoint, map_location='cpu')
    center = torch.tensor(center_state['center_t'], dtype=torch.float32, device='cuda')
    matcher = Matcher(inlier_threshold=2.0, d_thre=2, num_iterations=10, ratio=0.15,
                      nms_radius=0.1, max_points=3000, k1=30)
    modes = {'L0_coordinates_L0_reliability': (False, False),
             'new_coordinates_L0_reliability': (True, False),
             'L0_coordinates_new_reliability': (False, True),
             'new_coordinates_new_reliability': (True, True)}
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    rows = {name: [] for name in modes}
    started = time.perf_counter()
    with torch.no_grad():
        for index, (base_row, new_row) in enumerate(zip(base['predictions'], candidate['predictions'])):
            if base_row['status'] != 'ok' or new_row['status'] != 'ok':
                for name in modes:
                    rows[name].append({'scan': base_row['scan'], 'status': 'failed',
                                       'error': 'source_prediction_failed'})
                continue
            original = load_cache(args.baseline, base_row)
            changed = load_cache(args.candidate, new_row)
            order = alignment(original, changed)
            for name, (use_coords, use_reliability) in modes.items():
                try:
                    initial, refined, selected = solve(
                        matcher, original, changed, order, use_coords,
                        use_reliability, center, 20260924 + index)
                    row = {'scan': base_row['scan'], 'status': 'ok',
                           'T_initial_world_body': initial,
                           'T_world_body': refined,
                           'selected_indices': selected}
                except Exception as error:
                    row = {'scan': base_row['scan'], 'status': 'failed',
                           'error': type(error).__name__, 'detail': str(error)}
                rows[name].append(row)
            print(f'{index + 1}/{len(base["predictions"])} {base_row["scan"]}', flush=True)
    for name, predictions in rows.items():
        folder = args.out / name
        folder.mkdir()
        payload = {'protocol': 'local905_gt_isolated_online_v1',
                   'source': 'voxel_identity_aligned_coordinate_reliability_swap_v1',
                   'baseline_predictions_sha256': digest(args.baseline / 'predictions.json'),
                   'candidate_predictions_sha256': digest(args.candidate / 'predictions.json'),
                   'checkpoint_sha256': candidate['checkpoint_sha256'],
                   'split_sha256': base['split_sha256'], 'subset': base['subset'],
                   'expected_frames': len(predictions),
                   'elapsed_seconds': time.perf_counter() - started,
                   'predictions': predictions}
        path = folder / 'predictions.json'
        temporary = folder / 'predictions.tmp'
        temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, path)
        path.with_suffix('.sha256').write_text(digest(path) + '\n', encoding='ascii')


if __name__ == '__main__':
    main()
