import argparse
import json
import os
import subprocess
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import e2e_online as E
from online_frontend import OnlineFrontend


SOURCE = Path(__file__).resolve()


def statistics(values):
    import numpy as np

    values = np.asarray(values) * 1000
    return dict(mean_ms=float(values.mean()), p50_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), maximum_ms=float(values.max()))


def initial_batch(front, prepared):
    import MinkowskiEngine as ME
    from data.leader_nclt_manifest import collate_leader_frames
    from models.magic_fusion import polar_voxel_centers
    from tools.magic_nclt_runtime import make_matcher, PoseEstimationFailure

    torch = front.torch
    batch = collate_leader_frames([loaded['sample'] for _, (loaded, _) in prepared])
    sparse = ME.SparseTensor(features=batch['feats'].cuda(), coordinates=batch['coords'].cuda())
    encoded = front.runtime.leader.encoder(sparse)
    points = polar_voxel_centers(encoded.C,
                torch.tensor(encoded.tensor_stride, device='cuda', dtype=torch.float32),
                front.lnm.DEFAULT_VOXEL_SIZE, front.lnm.DEFAULT_HORIZONTAL_RESOLUTION)
    prediction = front.runtime.leader.decoder(encoded.F)
    results = []
    for index, (record, _) in enumerate(prepared):
        mask = encoded.C[:, 0] == index
        values, positions = prediction[mask], points[mask]
        count = max(min(50, len(values)), int(.5 * len(values)))
        if count < 3:
            raise PoseEstimationFailure('insufficient correspondences', len(values), count)
        selected = values[:, 3].topk(count).indices
        source, target = positions[selected], values[selected, :3]
        seed = front.runtime.config['seed'] + record['canonical_index']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        pose = make_matcher(torch.device('cuda:0')).estimator(source[None], target[None])[0]
        if not torch.isfinite(pose).all():
            raise FloatingPointError('nonfinite SC2 pose')
        results.append((pose, source, target, len(values)))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keys-file', required=True)
    parser.add_argument('--scratch-root', type=Path, required=True)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--frames', type=int, choices=(32, 96, 384), default=32)
    parser.add_argument('--mode', choices=('compare', 'batched', 'cooperative'), default='compare')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, choices=(1, 3, 4), default=1)
    parser.add_argument('--gate-file', type=Path)
    args = parser.parse_args()
    config = E.parser().parse_args(['--keys-file', args.keys_file, '--gpu', args.gpu,
                                   '--frontend-workers', '4', '--threads', '1'])
    config.shard_index, config.shard_count = 0, 1
    specs = E.select_inputs(config)
    if not args.frames <= len(specs) <= 384:
        raise ValueError('expected bounded unique input pool')
    specs = [specs[round(i * (len(specs) - 1) / (args.frames - 1))] for i in range(args.frames)]
    if not 0 <= args.shard < args.shards:
        raise ValueError('invalid shard')
    specs = specs[args.shard::args.shards]
    parent = Path('/data2/zhangcy').resolve(strict=True)
    target = args.scratch_root
    if (not target.is_absolute() or target.parent.resolve(strict=True) != parent
            or not target.name.startswith('e2e-live-batch-') or target.exists()
            or target.is_symlink() or parent.stat().st_uid != os.getuid()):
        raise ValueError('scratch must be a new owned isolated data2 directory')
    target.mkdir(mode=0o700)
    (target / 'results').mkdir(mode=0o700)
    E.PROJECT = target
    E.configure(config)
    guard = E.ResourceGuard(config)
    front = None
    try:
        def gpu_snapshot():
            return subprocess.check_output(['nvidia-smi', '-i', args.gpu,
                    '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True).strip()

        before = gpu_snapshot()
        if before and args.shards == 1:
            raise RuntimeError('GPU already has compute processes; benchmark not started')
        initialization_started = time.monotonic()
        front = OnlineFrontend(config)
        torch = front.torch

        def prepare(batch):
            result, arrivals = [], []
            for spec in batch:
                guard.check()
                arrivals.append(time.monotonic())
                if any(E.sha_file(spec[name + '_path']) != spec[name + '_sha256']
                       for name in ('scan', 'image')):
                    raise ValueError('input SHA mismatch')
                _, xyz, intensity = E.read_scan(spec['scan_path'])
                result.append((spec, front.prepare(spec, xyz, intensity)))
            return result, arrivals

        retained = OrderedDict()
        for date in dict.fromkeys(spec['key'].split('/')[0] for spec in specs):
            selected = [spec for spec in specs if spec['key'].split('/')[0] == date]
            prepare(selected[:1])
            manifest_root, manifest = front.sam.manifests[date]
            manifest['frames'] = {spec['key']: manifest['frames'][spec['key']] for spec in selected}
            retained[date] = (manifest_root, manifest)
        front.sam.manifests = retained

        def batched(prepared, geometry_pool, cooperative=False):
            components = {}
            tick = time.monotonic()
            with torch.no_grad():
                initials = initial_batch(front, prepared)
                torch.cuda.synchronize()
                components['leader'] = time.monotonic() - tick
                tick = time.monotonic()
                futures = [geometry_pool.submit(front.runtime.geometry, loaded, initial[1].detach().cpu())
                           for (_, (loaded, _)), initial in zip(prepared, initials)]
                frames = [front.runtime.frame(loaded, *initial, future.result())
                          for (_, (loaded, _)), initial, future in zip(prepared, initials, futures)]
                torch.cuda.synchronize()
                components['geometry_frame'] = time.monotonic() - tick
                tick = time.monotonic()
                if cooperative:
                    from infer_batch_engine_v2 import refine_batch

                    observed = front.model.observation_batch([f['sam'] for f in frames],
                        [f['rgb'] for f in frames], [f['pixels'] for f in frames], [f['hw'] for f in frames])
                    predictions = refine_batch(front.model, frames, observed, front.inference)
                else:
                    predictions = front.parallel.run(frames)
                torch.cuda.synchronize()
                poses = []
                for prediction in predictions:
                    pose = prediction[0].detach().cpu().double().numpy().copy()
                    pose[:3, 3] += front.runtime.center.detach().cpu().double().numpy()
                    poses.append(E.valid_pose(pose).tolist())
                components['v7_output'] = time.monotonic() - tick
            return poses, components

        with ThreadPoolExecutor(max_workers=2) as geometry_pool:
            warm, _ = prepare(specs[:4])
            front.predict_many(warm)
            batched(warm, geometry_pool, args.mode == 'cooperative')
            del warm
            initialization_seconds = time.monotonic() - initialization_started
            (target / 'results' / 'ready.json').write_text(json.dumps(dict(pid=os.getpid())))
            if args.gate_file:
                gate_started = time.monotonic()
                while not args.gate_file.exists():
                    guard.check()
                    if time.monotonic() - gate_started > 600:
                        raise TimeoutError('benchmark start barrier timed out')
                    time.sleep(.2)
            torch.cuda.reset_peak_memory_stats()
            reports, all_rows = {}, {}
            for mode in (('serial', 'batched', 'cooperative') if args.mode == 'compare' else (args.mode,)):
                size = 1 if mode == 'serial' else 4
                batches = [specs[i:i + size] for i in range(0, len(specs), size)]
                rows, component_rows = [], []
                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=1) as prefetch:
                    pending = prefetch.submit(prepare, batches[0])
                    for index, batch in enumerate(batches):
                        guard.check()
                        prepared, arrivals = pending.result()
                        if index + 1 < len(batches):
                            pending = prefetch.submit(prepare, batches[index + 1])
                        tick = time.monotonic()
                        if mode == 'serial':
                            output = front.predict_many(prepared)
                            poses = [E.valid_pose(row[0]).tolist() for row in output]
                            components = output[0][1]
                        else:
                            poses, components = batched(prepared, geometry_pool, mode == 'cooperative')
                        finished = time.monotonic()
                        component_rows.append(components)
                        for spec, pose, arrival in zip(batch, poses, arrivals):
                            rows.append(dict(key=spec['key'], T=pose, input_started_at=arrival,
                                             finished_at=finished, latency_seconds=finished - arrival,
                                             compute_batch_seconds=finished - tick))
                elapsed = time.monotonic() - started
                reports[mode] = dict(frames=len(rows), span_seconds=elapsed,
                    started_at=started, finished_at=started + elapsed,
                    aggregate_ms_per_frame=elapsed * 1000 / len(rows),
                    input_to_T=statistics([row['latency_seconds'] for row in rows]),
                    component_ms_per_frame={name: sum(r[name] for r in component_rows) * 1000 / len(rows)
                                            for name in component_rows[0]})
                all_rows[mode] = rows
                print(json.dumps({mode: reports[mode]}), flush=True)
        import numpy as np
        parity = None
        if args.mode == 'compare':
            parity = {}
            for mode in ('batched', 'cooperative'):
                differences = [float(np.max(np.abs(np.asarray(a['T']) - np.asarray(b['T']))))
                               for a, b in zip(all_rows['serial'], all_rows[mode])]
                parity[mode] = dict(max_pose_element_difference=max(differences),
                                   frames_above_1e_4=sum(d > 1e-4 for d in differences),
                                   differences=differences)
        report = dict(scope='live raw LiDAR + RGB + precomputed SAM to LEADER+V7 world_from_body; no upstream pose cache; no experts',
                      gpu=args.gpu, pid=os.getpid(), gpu_before=before, gpu_after=gpu_snapshot(),
                      benchmark_sha256=E.sha_file(SOURCE), peak_torch_bytes=torch.cuda.max_memory_allocated(),
                      initialization_seconds=initialization_seconds,
                      metadata_cache='verified SAM manifests pruned to selected frame metadata; no upstream inference cache',
                      GT_attempts=len(front.runtime.denied), reports=reports, parity=parity, rows=all_rows)
        (target / 'results' / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)
    finally:
        if front:
            front.close()
        guard.close()


if __name__ == '__main__':
    main()
