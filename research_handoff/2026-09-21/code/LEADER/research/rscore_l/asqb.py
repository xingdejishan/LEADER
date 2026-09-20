import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import poselib
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

from .evaluate import pose_error
from .prepare import save_json
from .topk import aggregate, candidate_scores, margins, select_points


def rotation_jacobian(uv, K):
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
    x, y = (rays[:, :2] / rays[:, 2:]).T
    jacobian = np.empty((len(uv), 2, 3))
    jacobian[:, 0] = np.c_[-x*y, 1+x*x, -y] * K[0, 0]
    jacobian[:, 1] = np.c_[-1-y*y, x*y, x] * K[1, 1]
    return jacobian


def geometry_modes(uv, K):
    J = rotation_jacobian(uv, K)
    fisher = np.einsum('nki,nkj->nij', J, J)
    features = np.c_[fisher[:, 0, 0], fisher[:, 1, 1], fisher[:, 2, 2],
        np.sqrt(2)*fisher[:, 0, 1], np.sqrt(2)*fisher[:, 0, 2], np.sqrt(2)*fisher[:, 1, 2]]
    features /= np.linalg.norm(features, axis=1, keepdims=True).clip(1e-12)
    count = min(64, len(np.unique(features, axis=0)))
    if count == 1:
        return np.zeros(len(uv), int)
    with threadpool_limits(limits=1):
        return MiniBatchKMeans(n_clusters=count, random_state=2089, n_init=3, batch_size=1024,
            max_iter=50, reassignment_ratio=0).fit_predict(features)


def select_modes(modes, reliability, count):
    if count >= len(modes):
        return np.arange(len(modes))
    order = np.argsort(-reliability, kind='stable')
    groups = [order[modes[order] == mode] for mode in np.unique(modes)]
    selected = []
    for rank in range(max(map(len, groups))):
        for group in groups:
            if rank < len(group):
                selected.append(int(group[rank]))
                if len(selected) == count:
                    return np.sort(selected)
    raise RuntimeError('ASQB could not fill existing-correspondence budget')


def spectrum(H):
    values = np.linalg.eigvalsh((H+H.T)/2)
    values = np.maximum(values, 0)
    rank = int((values > max(values[-1], 1e-30)*1e-9).sum())
    return dict(eigenvalues=values.tolist(), rank=rank,
        condition=float(values[-1]/max(values[0], values[-1]*1e-12, 1e-30)),
        minimum_over_maximum=float(values[0]/max(values[-1], 1e-30)))


def constraint_diagnostic(uv, xyz, K, size, reference):
    q = (xyz-reference[:3, 3]) @ reference[:3, :3]
    projection = q @ K.T
    predicted = projection[:, :2] / projection[:, 2:].clip(1e-8)
    good = (q[:, 2] > 0) & np.isfinite(predicted).all(1) & (np.linalg.norm(predicted-uv, axis=1) < 10)
    points, image_points = q[good], uv[good]
    result = dict(points=len(uv), good_points=int(good.sum()), good_fraction=float(good.mean()))
    if not len(points):
        return dict(result, occupied_cells=0, largest_cell_fraction=None, angular_rms_deg=None, rotation=None, rotation_schur=None)
    cells = np.clip((image_points / size[::-1]*4).astype(int), 0, 3)
    bins = np.bincount(cells[:, 0]+4*cells[:, 1], minlength=16)
    rays = np.c_[image_points, np.ones(len(image_points))] @ np.linalg.inv(K).T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    center = rays.mean(0)
    center /= np.linalg.norm(center)
    result.update(occupied_cells=int((bins>0).sum()), largest_cell_fraction=float(bins.max()/len(points)),
        angular_rms_deg=float(np.sqrt(np.mean(np.degrees(np.arccos(np.clip(rays @ center, -1, 1)))**2))))
    J_r = rotation_jacobian(predicted[good], K)
    x, y, z = points.T
    J_t = np.zeros_like(J_r)
    J_t[:, 0, 0], J_t[:, 0, 2] = K[0, 0]/z, -K[0, 0]*x/z**2
    J_t[:, 1, 1], J_t[:, 1, 2] = K[1, 1]/z, -K[1, 1]*y/z**2
    R, T = J_r.reshape(-1, 3), J_t.reshape(-1, 3)
    H_rr, H_tt, H_rt = R.T @ R/len(points), T.T @ T/len(points), R.T @ T/len(points)
    schur = H_rr - H_rt @ np.linalg.pinv(H_tt, rcond=1e-10) @ H_rt.T
    return dict(result, rotation=spectrum(H_rr), rotation_schur=spectrum(schur))


def evaluate_asqb_frame(job):
    root, bundle, index, row, reference, control, digest = job
    root, bundle = Path(root), Path(bundle)
    source = root / 'exports/reliable/test' / (row['frame_id']+'.npz')
    pool_path = bundle / 'cache/lidar_pools' / source.name
    input_digest = hashlib.sha256(source.read_bytes()+pool_path.read_bytes()).hexdigest()
    if input_digest != control['input_sha256']:
        raise RuntimeError('Frozen Top-K inputs changed: ' + row['frame_id'])
    destination = root / 'asqb/test/frames' / (row['frame_id']+'.json')
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved['protocol_sha256'] == digest and saved['input_sha256'] == input_digest:
            return saved
    camera = dict(np.load(source))
    size, K = camera['image_size_hw'], camera['K']
    started = time.perf_counter()
    modes = geometry_modes(camera['uv'], K)
    grouping_seconds = time.perf_counter()-started
    config = dict(model='PINHOLE', width=int(size[1]), height=int(size[0]),
        params=[float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])])
    pool = dict(np.load(pool_path))
    inference_pool = {k: v for k, v in pool.items() if k != 'GT'}
    arms = {}
    for percent in (10, 20, 50, 100):
        count = int(round(len(modes)*percent/100))
        for method in ('uniform', 'top_grid', 'asqb'):
            key = f'{method}_{percent}'
            if method != 'asqb' or percent == 100:
                arm = dict(control['arms'][f'{method if method != "asqb" else "uniform"}_{percent}'])
                hypothesis = arm['selected_hypothesis']
                selected, _ = select_points(camera['uv'], camera['reliability'][hypothesis], size, count,
                    method if method != 'asqb' else 'uniform', 2089+index)
            else:
                selections, direction, coverage = [], [], []
                best_count, best_pose, hypothesis = -1, None, None
                started = time.perf_counter()
                for h, probability in enumerate(camera['reliability']):
                    take = select_modes(modes, probability, count)
                    selections.append(take)
                    uv, xyz = camera['uv'][take], camera['xyz'][h, take]
                    pose, info = poselib.estimate_absolute_pose(uv.astype(float), xyz.astype(float), config,
                        dict(max_reproj_error=10., max_iterations=10000, seed=2089), {})
                    if info['num_inliers'] > best_count:
                        T = np.eye(4)
                        T[:3, :3], T[:3, 3] = pose.R, pose.t
                        best_pose, best_count, hypothesis = np.linalg.inv(T), int(info['num_inliers']), h
                    diagnostic = constraint_diagnostic(uv, xyz, K, size, reference)
                    direction.append(diagnostic['good_points'])
                    cells = np.clip((uv/size[::-1]*4).astype(int), 0, 3)
                    bins = np.bincount(cells[:, 0]+4*cells[:, 1], minlength=16)
                    coverage.append(dict(occupied=int((bins>0).sum()), maximum_cell_fraction=float(bins.max()/count)))
                selected = selections[hypothesis]
                candidates, lidar, visual, joint = candidate_scores(inference_pool, camera, selections)
                error = np.array([pose_error(T, pool['GT']) for T in candidates])
                correct = (error[:, 0] < 1) & (error[:, 1] < 2)
                arm = dict(points_per_hypothesis=count, camera_error=pose_error(best_pose, reference), pnp_inliers=best_count,
                    selected_hypothesis=hypothesis, correspondence_fraction=float(np.sum(direction)/(count*len(direction))),
                    reprojection_inliers_by_hypothesis=direction, coverage=coverage,
                    selection_pnp_and_diagnostic_seconds=time.perf_counter()-started,
                    candidates=dict(total=len(correct), correct=int(correct.sum()), lidar=margins(lidar, correct),
                        camera_fit=margins(visual[0], correct), camera_holdout=margins(visual[1], correct),
                        joint_fit=margins(joint[0], correct), joint_holdout=margins(joint[1], correct)))
            arm['constraint'] = constraint_diagnostic(camera['uv'][selected], camera['xyz'][hypothesis, selected], K, size, reference)
            arms[key] = arm
    result = dict(frame_id=row['frame_id'], protocol_sha256=digest, input_sha256=input_digest,
        modes=len(np.unique(modes)), grouping_seconds=grouping_seconds, arms=arms)
    save_json(destination, result)
    return result


def run_asqb(root, bundle):
    output = root / 'asqb/test'
    (output / 'frames').mkdir(parents=True, exist_ok=True)
    rows = json.loads((root / 'data/manifest.json').read_text())['test']
    poses = np.load(root / 'data/test/poses.npy')
    controls = json.loads((root / 'topk/test/records.json').read_text())
    assert [r['frame_id'] for r in controls] == [r['frame_id'] for r in rows]
    protocol = dict(method='Local ASQB prototype, no new correspondences', mode_count=64,
        geometry='Normalized symmetric J_rotation.T @ J_rotation signature from observed bearings and K; Frobenius metric',
        clustering='MiniBatchKMeans, seed 2089, n_init 3, batch 1024, max_iter 50',
        representatives='Reliability-descending within each mode, round-robin across modes, exact original-point budget',
        budgets_percent=[10, 20, 50, 100], hypotheses=10, pnp=dict(max_reproj_error=10, max_iterations=10000, seed=2089),
        diagnostics='Reference-pose <10px positive-depth directional inliers, selected PnP hypothesis only; 4x4 coverage, angular spread, per-point rotation Fisher and translation-marginalized Schur complement',
        reference_usage='Post-selection diagnostics and evaluation only; not read by mode generation or sampling',
        diagnostic_warning='Directional consistency does not establish 3D correctness; Fisher/Schur uses predicted depths and is not calibrated uncertainty',
        controls='Reuse identical frozen Top-K diagnostic records with per-frame input hash checks',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        controls_sha256=hashlib.sha256((root / 'topk/test/records.json').read_bytes()).hexdigest())
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    save_json(output / 'protocol.json', protocol)
    jobs = [(str(root), str(bundle), i, row, poses[i], controls[i], digest) for i, row in enumerate(rows)]
    records = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as executor:
        for future in as_completed([executor.submit(evaluate_asqb_frame, job) for job in jobs]):
            records.append(future.result())
            elapsed = time.perf_counter()-started
            save_json(output / 'progress.json', dict(completed=len(records), total=len(rows), elapsed_seconds=elapsed,
                estimated_remaining_seconds=elapsed/len(records)*(len(rows)-len(records))))
    by_frame = {r['frame_id']: r for r in records}
    records = [by_frame[row['frame_id']] for row in rows]
    save_json(output / 'records.json', records)
    save_json(output / 'summary.json', aggregate(records))
