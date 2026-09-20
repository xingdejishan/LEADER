import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from joint_lidar_camera_refinement import frozen_lidar_information, pixel_pose_jacobian
from local_visual_refinement_roma import RoMaField, apply_local_delta, load_match_cache
from oracle_pose_refinement import load_module, pose_error, pose_from_baseline, project_world
from sweep_roma_protected_backend import evaluate_frame


VERSION = 'native_context_pose_v1'
RADIUS = 6
SIDE = RADIUS * 2 + 1


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hypotheses(seed=2089, pairs=128):
    engine = torch.quasirandom.SobolEngine(6, scramble=True, seed=seed)
    values = engine.draw(pairs).double().numpy() * 2 - 1
    for start in (0, 3):
        norms = np.linalg.norm(values[:, start:start + 3], axis=1, keepdims=True)
        values[:, start:start + 3] /= np.maximum(norms, 1)
    values *= np.array([.15] * 3 + [math.radians(1.5)] * 3)
    return np.concatenate((np.zeros((1, 6)), values, -values))


def candidate_offsets():
    y, x = np.mgrid[-RADIUS:RADIUS + 1, -RADIUS:RADIUS + 1]
    return np.stack((x, y), axis=-1).reshape(-1, 2)


class NativeContext:
    def __init__(self, device):
        self.field = RoMaField(device, 'precise')
        self.field.model.requires_grad_(False)
        self.captured = []
        self.grid = None
        self.handle = self.field.model.refiners['1'].warp_head.register_forward_pre_hook(self.capture)

    def capture(self, module, inputs):
        self.captured.append(F.grid_sample(inputs[0].float(), self.grid, align_corners=False,
                                           padding_mode='border')[0, :, :, 0].T.cpu().numpy())

    def pair(self, reference, query, uv, reference_hw, query_hw):
        self.captured.clear()
        grid = torch.as_tensor(uv, dtype=torch.float32, device=self.field.device)
        scale = grid.new_tensor([reference_hw[1], reference_hw[0]])
        self.grid = (2 * grid / scale - 1)[None, :, None]
        start = time.perf_counter()
        result = self.field.model.match(reference, query)
        pixels, overlap, precision = self.field.sample(result, uv, reference_hw, query_hw)
        if not self.captured:
            raise RuntimeError('RoMa native stride-1 context hook was not called')
        context = self.captured[-2 if self.field.model.bidirectional else -1]
        if context.shape != (len(uv), 32):
            raise RuntimeError('unexpected RoMa native context shape')
        torch.cuda.synchronize() if self.field.device.type == 'cuda' else None
        return context, pixels, overlap, precision, time.perf_counter() - start


def build(args):
    from PIL import Image

    rows = json.loads(Path(args.manifest).read_text())
    by_id = {r['frame_id']: r for r in rows}
    matcher_module = load_module('candidate_matcher', HERE.parents[1] / 'models/sc2pcr.py')
    pool = load_module('candidate_pool', args.full_pool)
    matcher = matcher_module.Matcher(inlier_threshold=2., d_thre=2, num_iterations=10,
                                     ratio=.15, nms_radius=.1, max_points=3000, k1=30)
    native = NativeContext(args.device)
    output = Path(args.cache)
    output.mkdir(parents=True, exist_ok=True)
    hyps = hypotheses()
    offsets = candidate_offsets()
    index = []
    for row in rows:
        source_dir = args.train_matches if row['split'] == 'train' else args.development_matches
        match_path = Path(source_dir) / (row['frame_id'] + '.npz')
        if not match_path.exists():
            continue
        path = output / (row['frame_id'] + '.npz')
        metadata = {'version': VERSION, 'manifest_sha256': digest(args.manifest),
                    'matches_sha256': digest(match_path), 'sequence': row['sequence'],
                    'split': 'train' if row['split'] == 'train' else 'development',
                    'frame_id': row['frame_id'], 'seed': 2089,
                    'native_context': 'frozen RoMa precise stride-1 warp_head input, 32 channels',
                    'selection': 'existing match-cache observations; no query GT visibility filtering'}
        if path.exists():
            with np.load(path) as old:
                if json.loads(str(old['metadata'])) != metadata:
                    raise ValueError('cache identity mismatch: ' + str(path))
            index.append(metadata)
            continue
        original = [r['frame_id'] for r in rows if r['split'] == row['split']].index(row['frame_id'])
        pose, gt, _, evidence = pose_from_baseline(row, args.lidar_cache, matcher, pool.full_pool_refine,
                                                  args.device, 2089 + original, True)
        lidar = frozen_lidar_information(pose, evidence)
        points, pixels, refuv, cameras, scores, precisions, anchors, refs, refcams, _ = load_match_cache(match_path)
        n = len(points)
        centres = np.zeros((n, 2))
        covariance = np.zeros((n, 2, 2))
        projected = np.zeros((len(hyps), n, 2))
        valid = np.zeros((n, SIDE * SIDE), bool)
        context = np.zeros((n, 32), np.float32)
        native_pixels = np.zeros((n, 2))
        native_overlap = np.zeros(n)
        native_precision = np.zeros((n, 2, 2))
        image_hw = np.zeros((n, 2))
        feature_seconds = 0.
        for camera in np.unique(cameras):
            keep = cameras == camera
            view = next(v for v in row['views'] if int(v['camera']) == int(camera))
            K = np.loadtxt(view['calibration'])
            ext = np.asarray(view['camera_to_body'])
            centres[keep], jac = pixel_pose_jacobian(points[keep], pose, ext, K)
            covariance[keep] = np.einsum('nai,ij,nbj->nab', jac, lidar['covariance'] * 16, jac) + np.eye(2)
            for h, delta in enumerate(hyps):
                projected[h, keep], depth = project_world(points[keep], apply_local_delta(pose, delta), ext, K)
                projected[h, np.where(keep)[0][depth <= .1]] = 1e6
            with Image.open(view['image']) as im:
                hw = (im.height, im.width)
            image_hw[keep] = hw
            mask = np.load(view['mask']).astype(bool)
            candidates = centres[keep, None] + offsets
            finite = np.isfinite(candidates).all(-1)
            clean = np.nan_to_num(candidates, nan=-1, posinf=-1, neginf=-1)
            ix = np.clip(np.floor(clean[..., 0]).astype(int), 0, hw[1] - 1)
            iy = np.clip(np.floor(clean[..., 1]).astype(int), 0, hw[0] - 1)
            valid[keep] = finite & (clean[..., 0] >= 0) & (clean[..., 0] < hw[1]) & (clean[..., 1] >= 0) & (clean[..., 1] < hw[0]) & mask[iy, ix]
            for ref, refcam in sorted(set(zip(refs[keep], refcams[keep]))):
                subset = keep & (refs == ref) & (refcams == refcam)
                refview = next(v for v in by_id[str(ref)]['views'] if int(v['camera']) == int(refcam))
                with Image.open(refview['image']) as im:
                    refhw = (im.height, im.width)
                ctx, uv, overlap, precision, elapsed = native.pair(refview['image'], view['image'], refuv[subset], refhw, hw)
                context[subset], native_pixels[subset] = ctx, uv
                native_overlap[subset], native_precision[subset] = overlap, precision
                feature_seconds += elapsed
        usable = valid.any(1) & np.isfinite(centres).all(1) & np.isfinite(context).all(1)
        if not usable.any():
            raise RuntimeError('no usable candidate observations for ' + row['frame_id'])
        config = {'name': 'existing_protected', 'sigma_t': .1, 'sigma_r_deg': 1.,
                  'visual_lambda': 1., 'floor_px': 1., 'accept_ratio': .95}
        start = time.perf_counter()
        b = evaluate_frame((pose, gt, row['views'], points, pixels, cameras, precisions, anchors, refs, refcams),
                           config, 2., math.radians(10), 100, 1., 5, 6)
        b_seconds = time.perf_counter() - start
        sigma = np.sqrt(np.diagonal(covariance[usable], axis1=1, axis2=2))
        candidates = centres[usable, None] + offsets
        difference = candidates - native_pixels[usable, None]
        maha = np.einsum('nki,nij,nkj->nk', difference, native_precision[usable], difference)
        visual = np.concatenate((np.broadcast_to(np.tanh(context[usable, None] / 10), (usable.sum(), len(offsets), 32)),
                                 np.clip(difference / RADIUS, -5, 5),
                                 np.log1p(np.maximum(maha, 0))[..., None] / 10,
                                 np.broadcast_to(native_overlap[usable, None, None], (usable.sum(), len(offsets), 1))), -1)
        geometric = np.concatenate((np.broadcast_to(offsets[None] / RADIUS, (usable.sum(), len(offsets), 2)),
                                     np.broadcast_to((np.sum(offsets ** 2, -1) / RADIUS ** 2)[None, :, None], (usable.sum(), len(offsets), 1)),
                                     np.broadcast_to(np.log1p(sigma)[:, None], (usable.sum(), len(offsets), 2)),
                                     np.broadcast_to((centres[usable] / image_hw[usable, ::-1])[:, None], (usable.sum(), len(offsets), 2))), -1)
        arrays = dict(visual=visual.astype(np.float32), geometry=geometric.astype(np.float32), valid=valid[usable],
                      projection_grid=((projected[:, usable] - centres[usable]) / RADIUS).transpose(1, 0, 2).astype(np.float32),
                      hypotheses=hyps, pose=pose, gt=gt, frame_id=row['frame_id'], sequence=row['sequence'],
                      cameras=cameras[usable], points=points[usable], centres=centres[usable],
                      lidar_hessian=lidar['hessian'], b_errors=np.asarray(b['after']), b_accepted=b['accepted'],
                      b_seconds=b_seconds, feature_seconds=feature_seconds,
                      a_errors=np.asarray(pose_error(pose, gt)), metadata=json.dumps(metadata))
        if not np.isfinite(visual).all() or not np.isfinite(geometric).all():
            raise ValueError('nonfinite candidate input')
        np.savez_compressed(path, **arrays)
        index.append(metadata)
        print('cached', row['frame_id'], row['split'], int(usable.sum()), 'native_seconds', round(feature_seconds, 2), flush=True)
        (output / 'index.json').write_text(json.dumps(index, indent=2))
        if args.limit and len(index) >= args.limit:
            break
    (output / 'index.json').write_text(json.dumps(index, indent=2))


class CandidatePose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = torch.nn.Sequential(torch.nn.Linear(43, 48), torch.nn.SiLU(),
                                          torch.nn.Linear(48, 32), torch.nn.SiLU(), torch.nn.Linear(32, 1))
        self.null = torch.nn.Parameter(torch.tensor(-1.))
        self.strength = torch.nn.Parameter(torch.tensor(2.))

    def forward(self, data, zero_visual=False):
        if len(data['visual']) == 0 or not data['valid'].any():
            probability = data['hypotheses'].new_zeros(len(data['hypotheses']))
            probability[0] = 1
            return data['hypotheses'][0] + self.null * 0, probability
        visual = torch.zeros_like(data['visual']) if zero_visual else data['visual']
        logits = self.scorer(torch.cat((visual, data['geometry']), -1)).squeeze(-1)
        logits = logits.masked_fill(~data['valid'], -1e4)
        null = self.null.expand(len(logits), 1)
        probability = torch.softmax(torch.cat((logits, null), -1), -1)
        maps = probability[:, :-1].reshape(-1, 1, SIDE, SIDE)
        sampled = F.grid_sample(maps, data['projection_grid'][:, :, None], align_corners=True,
                                mode='bilinear', padding_mode='zeros')[:, 0, :, 0]
        likelihood = sampled + probability[:, -1:] / (SIDE * SIDE)
        point_score = torch.log(likelihood.clamp_min(1e-9))
        camera_scores = [point_score[data['cameras'] == c].mean(0) for c in torch.unique(data['cameras'])]
        score = torch.stack(camera_scores).mean(0)
        delta = data['hypotheses']
        scale = delta.new_tensor([.08] * 3 + [math.radians(.8)] * 3)
        prior = .5 * ((delta / scale) ** 2).sum(-1)
        hessian = data['lidar_hessian']
        diagonal = torch.sqrt(torch.diagonal(hessian).clamp_min(1e-6))
        correlation = hessian / (diagonal[:, None] * diagonal[None, :])
        z = delta / scale
        prior = prior + .25 * torch.einsum('hi,ij,hj->h', z, correlation, z)
        posterior = torch.softmax(F.softplus(self.strength) * 10 * (score - score[0]) - prior, 0)
        proposed = posterior @ delta
        return proposed, posterior


def rotation_matrix(vector):
    x, y, z = vector.unbind()
    zero = x * 0
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    return torch.matrix_exp(skew)


def final_pose_loss(delta, initial, truth):
    translation = torch.linalg.vector_norm(initial[:3, 3] + delta[:3] - truth[:3, 3])
    rotation = rotation_matrix(delta[3:]) @ initial[:3, :3]
    relative = rotation.T @ truth[:3, :3]
    vee = torch.stack((relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]))
    sine = torch.linalg.vector_norm(vee) / 2
    cosine = (torch.trace(relative) - 1) / 2
    angle = torch.atan2(sine, cosine)
    return translation / .1 + angle / math.radians(1)


def load_frames(cache, device):
    frames = []
    for entry in json.loads((Path(cache) / 'index.json').read_text()):
        with np.load(Path(cache) / (entry['frame_id'] + '.npz')) as raw:
            frame = {k: raw[k].copy() for k in raw.files}
        frame['metadata'] = entry
        frame['tensor'] = {k: torch.as_tensor(frame[k], device=device, dtype=torch.bool if k == 'valid' else
                                            torch.int64 if k == 'cameras' else torch.float32)
                           for k in ('visual', 'geometry', 'valid', 'projection_grid', 'hypotheses', 'pose', 'gt', 'cameras', 'lidar_hessian')}
        frames.append(frame)
    return frames


def metrics(errors, baseline, elapsed):
    errors = np.asarray(errors)
    baseline = np.asarray(baseline)
    return {'frames': len(errors), 'MPE_m': float(errors[:, 0].mean()), 'MOE_deg': float(errors[:, 1].mean()),
            'P90_translation_m': float(np.quantile(errors[:, 0], .9)), 'P90_rotation_deg': float(np.quantile(errors[:, 1], .9)),
            'translation_degraded_fraction': float((errors[:, 0] > baseline[:, 0] + 1e-8).mean()),
            'rotation_degraded_fraction': float((errors[:, 1] > baseline[:, 1] + 1e-8).mean()),
            'either_degraded_fraction': float((errors > baseline + 1e-8).any(1).mean()),
            'mean_measured_added_seconds': float(np.mean(elapsed))}


def measure_geometry(frame, row):
    start = time.perf_counter()
    poses = [apply_local_delta(frame['pose'], delta) for delta in frame['hypotheses']]
    grids = np.empty_like(frame['projection_grid'])
    for camera in np.unique(frame['cameras']):
        keep = frame['cameras'] == camera
        view = next(v for v in row['views'] if int(v['camera']) == int(camera))
        K, ext = np.loadtxt(view['calibration']), np.asarray(view['camera_to_body'])
        for h, pose in enumerate(poses):
            uv, depth = project_world(frame['points'][keep], pose, ext, K)
            uv[depth <= .1] = 1e6
            grids[keep, h] = (uv - frame['centres'][keep]) / RADIUS
    elapsed = time.perf_counter() - start
    if not np.allclose(grids, frame['projection_grid'], atol=1e-5):
        raise ValueError('cached projection does not match current geometry')
    return elapsed


def train(args):
    torch.set_num_threads(4)
    frames = load_frames(args.cache, args.device)
    training = [f for f in frames if f['metadata']['split'] == 'train']
    development = [f for f in frames if f['metadata']['split'] == 'development']
    if not training or not development:
        raise ValueError('both training and development frames are required')
    rows = {r['frame_id']: r for r in json.loads(Path(args.manifest).read_text())}
    for frame in development:
        frame['geometry_seconds'] = measure_geometry(frame, rows[frame['metadata']['frame_id']])
    train_sequences = sorted({f['metadata']['sequence'] for f in training})
    dev_sequences = sorted({f['metadata']['sequence'] for f in development})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = {'protocol': VERSION, 'status': 'training', 'evaluation_status': 'development_only_not_independent_test',
               'train_sequences': train_sequences, 'development_sequences': dev_sequences,
               'sequence_overlap': sorted(set(train_sequences) & set(dev_sequences)),
               'training_frames': len(training), 'development_frames': len(development),
               'settings': vars(args), 'pose_solver': '257 fixed antithetic bounded SE(3) hypotheses; differentiable posterior mean; exact hypothesis projection',
               'runtime': {'torch': torch.__version__, 'training_device': args.device,
                           'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None},
               'loss': 'final translation norm / 0.1 m + final SO(3) geodesic error / 1 degree; no query pixel GT',
               'source_sha256': digest(__file__),
               'limitations': ['same-sequence development only', 'shared preexisting RoMa-selected observations may encode visual selection',
                               'finite pose hypotheses approximate continuous optimization',
                               'native front-end and hypothesis projection times are replay measurements; original observation selection and cache I/O are not included',
                               'D head timing excludes inherited visual observation selection, so it is not an end-to-end image-free runtime'],
               'comparison': {}, 'records': []}
    baseline = [f['a_errors'] for f in development]
    results['comparison']['A_frozen_LEADER'] = metrics(baseline, baseline, [0] * len(development))
    results['comparison']['B_existing_protected'] = metrics([f['b_errors'] for f in development], baseline,
                                                          [float(f['b_seconds'] + f['feature_seconds']) for f in development])
    for zero_visual, name in ((False, 'C_native_candidate_pose'), (True, 'D_retrained_zero_visual')):
        torch.manual_seed(args.seed)
        model = CandidatePose().to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        rng = np.random.default_rng(args.seed)
        history = []
        for epoch in range(args.epochs):
            losses = []
            model.train()
            for idx in rng.permutation(len(training)):
                data = training[idx]['tensor']
                optimizer.zero_grad()
                delta, _ = model(data, zero_visual)
                loss = final_pose_loss(delta, data['pose'], data['gt'])
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite pose training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
            history.append(float(np.mean(losses)))
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                print(name, 'epoch', epoch + 1, 'train_loss', history[-1], flush=True)
        torch.save({'state_dict': model.state_dict(), 'settings': vars(args), 'zero_visual': zero_visual,
                    'version': VERSION, 'training_frame_ids': [f['metadata']['frame_id'] for f in training],
                    'train_loss': history}, output.with_name(name + '.pt'))
        errors, times, records, head_times = [], [], [], []
        model.eval()
        for frame in development:
            start = time.perf_counter()
            with torch.no_grad():
                delta, posterior = model(frame['tensor'], zero_visual)
            torch.cuda.synchronize() if args.device.startswith('cuda') else None
            seconds = time.perf_counter() - start
            head_times.append(seconds)
            correction = delta.cpu().numpy()
            pose = apply_local_delta(frame['pose'], correction)
            error = pose_error(pose, frame['gt'])
            errors.append(error)
            times.append(seconds + frame['geometry_seconds'] + (0 if zero_visual else float(frame['feature_seconds'])))
            records.append({'frame_id': frame['metadata']['frame_id'], 'delta': correction.tolist(),
                            'errors': list(error), 'A_errors': frame['a_errors'].tolist(),
                            'B_errors': frame['b_errors'].tolist(), 'head_seconds': seconds,
                            'hypothesis_geometry_seconds': frame['geometry_seconds'],
                            'native_feature_seconds': float(frame['feature_seconds']),
                            'posterior_zero_probability': float(posterior[0])})
        results['comparison'][name] = metrics(errors, baseline, times)
        results['comparison'][name]['mean_head_seconds'] = float(np.mean(head_times))
        results['comparison'][name]['mean_hypothesis_geometry_seconds'] = float(np.mean([f['geometry_seconds'] for f in development]))
        results['comparison'][name]['mean_native_frontend_seconds'] = 0. if zero_visual else float(np.mean([f['feature_seconds'] for f in development]))
        results['records'].append({'method': name, 'frames': records, 'train_loss': history})
        output.write_text(json.dumps(results, indent=2, allow_nan=False))
    candidate = results['comparison']['C_native_candidate_pose']
    results['development_C_beats_A_B_D_on_both_means'] = all(
        candidate[key] < results['comparison'][name][key]
        for name in ('A_frozen_LEADER', 'B_existing_protected', 'D_retrained_zero_visual')
        for key in ('MPE_m', 'MOE_deg'))
    results['independent_test_success'] = None
    results['status'] = 'completed_development_only'
    output.write_text(json.dumps(results, indent=2, allow_nan=False))
    print(json.dumps(results['comparison'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['build', 'train'])
    parser.add_argument('--manifest', default='/home/zhang/leader-image-gate-multicamera/all_views.json')
    parser.add_argument('--lidar-cache', default='/home/zhang/leader-image-gate/lidar')
    parser.add_argument('--full-pool', default=str(HERE.parents[2] / 'glace-local/code/tools/full_pool_robust_v1.py'))
    parser.add_argument('--train-matches', default=str(HERE / 'results/roma_train_precise_top2_cache_refuv'))
    parser.add_argument('--development-matches', default=str(HERE / 'results/lscr_v1_validation8_matches_refuv'))
    parser.add_argument('--cache', default=str(HERE / 'results/native_candidate_pose_v1'))
    parser.add_argument('--output', default=str(HERE / 'results/native_candidate_pose_development.json'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=2089)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    build(args) if args.command == 'build' else train(args)


if __name__ == '__main__':
    main()
