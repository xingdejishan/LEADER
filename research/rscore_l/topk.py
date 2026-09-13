import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import poselib

from .evaluate import pose_error, summary
from .fusion import camera_score, lidar_score, project
from .prepare import save_json


RATIOS = (.1, .2, .5, 1.)


def select_points(uv, probability, size, count, mode, seed):
    if count >= len(uv):
        return np.arange(len(uv)), len(uv)
    h, w = size
    cells = np.clip((uv / [w, h] * 4).astype(int), 0, 3)
    blocks = cells[:, 0] + 4*cells[:, 1]
    populations = np.bincount(blocks, minlength=16)
    if mode == 'uniform':
        order = np.random.default_rng(seed).permutation(len(uv))
        bins = [order[blocks[order] == b] for b in range(16)]
        selected = []
        for rank in range(int(populations.max())):
            for group in bins:
                if rank < len(group):
                    selected.append(int(group[rank]))
                    if len(selected) == count:
                        return np.sort(selected), 0
    cap = max(1, int(np.ceil(2*count/16)))
    while np.minimum(populations, cap).sum() < count:
        cap += 1
    order = np.argsort(-probability, kind='stable')
    counts = np.zeros(16, int)
    selected = []
    for index in order:
        block = blocks[index]
        if counts[block] < cap:
            selected.append(int(index))
            counts[block] += 1
        if len(selected) == count:
            return np.sort(selected), cap
    raise RuntimeError('Could not fill point budget')


def candidate_scores(pool, camera, selections):
    E, K, size = camera['T_BC'], camera['K'], camera['image_size_hw']
    candidates = np.concatenate([pool['v1_two_stage'][None], pool['candidate_T_WB']])
    candidates = candidates[np.isfinite(candidates).all(axis=(1, 2))]
    Q = pool['T_corr']
    body = (pool['c_local_all'].astype(float) - Q[:3, 3]) @ Q[:3, :3]
    world = pool['c_pred_all'].astype(float) + pool['center_t']
    weights = np.exp(np.log(10)/np.pi*np.arctan(np.clip(pool['u_pred_all'].reshape(-1), -10*np.pi, 10*np.pi)))
    weights /= weights.sum()
    lidar = np.array([lidar_score(T, body, world, weights) for T in candidates])
    visual = np.full((2, len(candidates)), np.inf)
    for hypothesis, indices in enumerate(selections):
        uv = camera['uv'][indices]
        cells = np.clip((uv / size[::-1] * 4).astype(int), 0, 3)
        fit = cells.sum(1) % 2 == 0
        for partition, keep in enumerate((fit, ~fit)):
            scores = np.array([camera_score(T, uv, camera['xyz'][hypothesis, indices], camera['reliability'][hypothesis, indices],
                K, E, size, keep) for T in candidates])
            visual[partition] = np.minimum(visual[partition], scores)
    joint = .5*lidar[None] + .5*visual/np.log(np.prod(size))
    return candidates, lidar, visual, joint


def margins(scores, correct):
    chosen = int(np.argmin(scores))
    result = dict(top1_correct=bool(correct[chosen]), winner=chosen)
    if correct.any() and (~correct).any():
        result['margin'] = float(scores[~correct].min() - scores[correct].min())
    else:
        result['margin'] = None
    return result


def evaluate_frame(job):
    root, bundle, split, index, row, reference, digest = job
    root, bundle = Path(root), Path(bundle)
    source = root / 'exports/reliable' / split / (row['frame_id'] + '.npz')
    pool_path = bundle / 'cache/lidar_pools' / source.name
    artifact_hash = hashlib.sha256(source.read_bytes() + (pool_path.read_bytes() if split == 'test' else b'')).hexdigest()
    destination = root / 'topk' / split / 'frames' / (row['frame_id'] + '.json')
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing['protocol_sha256'] == digest and existing['input_sha256'] == artifact_hash:
            return existing
    camera = dict(np.load(source))
    size, K = camera['image_size_hw'], camera['K']
    solver_camera = dict(model='PINHOLE', width=int(size[1]), height=int(size[0]),
        params=[float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])])
    pool = dict(np.load(pool_path)) if split == 'test' else None
    inference_pool = {k: v for k, v in pool.items() if k != 'GT'} if pool else None
    arms = {}
    for ratio in RATIOS:
        count = int(round(len(camera['uv'])*ratio))
        for mode in ('uniform', 'top_grid'):
            key = f'{mode}_{int(ratio*100)}'
            if ratio == 1 and mode == 'top_grid':
                arms[key] = dict(arms['uniform_100'])
                continue
            selections, coverage, direction = [], [], []
            best_count, best_pose, best_hypothesis = -1, None, None
            started = time.perf_counter()
            for hypothesis, probability in enumerate(camera['reliability']):
                selected, cap = select_points(camera['uv'], probability, size, count, mode, 2089+index)
                selections.append(selected)
                uv, xyz = camera['uv'][selected], camera['xyz'][hypothesis, selected]
                pose, info = poselib.estimate_absolute_pose(uv.astype(float), xyz.astype(float), solver_camera,
                    dict(max_reproj_error=10., max_iterations=10000, seed=2089), {})
                if info['num_inliers'] > best_count:
                    T_CW = np.eye(4)
                    T_CW[:3, :3], T_CW[:3, 3] = pose.R, pose.t
                    best_pose, best_count, best_hypothesis = np.linalg.inv(T_CW), int(info['num_inliers']), hypothesis
                cells = np.clip((uv / size[::-1]*4).astype(int), 0, 3)
                populations = np.bincount(cells[:, 0] + 4*cells[:, 1], minlength=16)
                coverage.append(dict(occupied=int((populations > 0).sum()), maximum_cell_fraction=float(populations.max()/count), cap=cap))
                projected, depth = project(reference @ np.linalg.inv(camera['T_BC']), xyz, K, camera['T_BC'])
                valid = (depth > 0) & np.isfinite(projected).all(1)
                direction.append(int((valid & (np.linalg.norm(projected-uv, axis=1) < 10)).sum()))
            arm = dict(points_per_hypothesis=count, camera_error=pose_error(best_pose, reference), pnp_inliers=best_count,
                selected_hypothesis=best_hypothesis, reprojection_inliers_by_hypothesis=direction,
                correspondence_fraction=float(np.sum(direction)/(count*len(direction))), coverage=coverage,
                pnp_and_selection_seconds=time.perf_counter()-started)
            if pool:
                candidates, lidar, visual, joint = candidate_scores(inference_pool, camera, selections)
                errors = np.array([pose_error(T, pool['GT']) for T in candidates])
                correct = (errors[:, 0] < 1) & (errors[:, 1] < 2)
                arm['candidates'] = dict(total=len(correct), correct=int(correct.sum()), lidar=margins(lidar, correct),
                    camera_fit=margins(visual[0], correct), camera_holdout=margins(visual[1], correct),
                    joint_fit=margins(joint[0], correct), joint_holdout=margins(joint[1], correct))
            arms[key] = arm
    record = dict(frame_id=row['frame_id'], protocol_sha256=digest, input_sha256=artifact_hash, arms=arms)
    save_json(destination, record)
    return record


def aggregate(records):
    results = {}
    for key in records[0]['arms']:
        arms = [r['arms'][key] for r in records]
        result = dict(camera=summary([a['camera_error'] for a in arms]),
            correspondence_fraction=float(np.mean([a['correspondence_fraction'] for a in arms])),
            mean_occupied_cells=float(np.mean([h['occupied'] for a in arms for h in a['coverage']])),
            mean_maximum_cell_fraction=float(np.mean([h['maximum_cell_fraction'] for a in arms for h in a['coverage']])))
        if 'candidates' in arms[0]:
            result['candidates'] = {}
            for score in ('lidar', 'camera_fit', 'camera_holdout', 'joint_fit', 'joint_holdout'):
                values = [a['candidates'][score] for a in arms]
                eligible = [v for v in values if v['margin'] is not None]
                result['candidates'][score] = dict(top1_correct=sum(v['top1_correct'] for v in values),
                    evaluable=len(eligible), positive_margin=sum(v['margin'] > 0 for v in eligible),
                    mean_margin=float(np.mean([v['margin'] for v in eligible])) if eligible else None,
                    median_margin=float(np.median([v['margin'] for v in eligible])) if eligible else None)
        results[key] = result
    return results


def run_topk(root, bundle, split):
    destination = root / 'topk' / split
    (destination / 'frames').mkdir(parents=True, exist_ok=True)
    rows = json.loads((root / 'data/manifest.json').read_text())[split]
    references = np.load(root / 'data' / split / 'poses.npy')
    protocol = dict(ratios=RATIOS, methods=['top_grid', 'uniform'], grid=[4, 4], cap='ceil(2*K/16), minimally relaxed only if insufficient capacity',
        uniform='Seeded random within cells; round-robin across cells', selection_order='Original index order after subset selection',
        pnp=dict(max_reproj_error=10, max_iterations=10000, seed=2089, hypotheses=10),
        candidate_success='translation <1m AND rotation <2deg', margin='min wrong cost minus min correct cost; positive favors correct',
        scoring='Original probability-weighted Gaussian/uniform mixture and LiDAR score; whole hypotheses only; fit and checkerboard holdout separately',
        candidate_scope='Fixed candidates including original LEADER; ranking diagnostic only, no refinement or acceptance gate',
        reference_usage='Evaluation only, never selection or score computation', split=split,
        reference_sha256=hashlib.sha256(references.tobytes()).hexdigest(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    save_json(destination / 'protocol.json', protocol)
    jobs = [(str(root), str(bundle), split, i, row, references[i], digest) for i, row in enumerate(rows)]
    records = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = [executor.submit(evaluate_frame, job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            elapsed = time.perf_counter()-started
            save_json(destination / 'progress.json', dict(completed=len(records), total=len(rows), elapsed_seconds=elapsed,
                estimated_remaining_seconds=elapsed/len(records)*(len(rows)-len(records))))
    indexed = {r['frame_id']: r for r in records}
    records = [indexed[row['frame_id']] for row in rows]
    save_json(destination / 'records.json', records)
    save_json(destination / 'summary.json', aggregate(records))
