#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Accelerated-path sweep: ParallelFrames (N model replicas + N CUDA streams).

Deployment construction (infer_shard_predict_v5.py:73, infer_fast_predict_v4.py:89):
  parallel = ParallelFrames(lambda: r.model(c, config, state['model'], False),
                            c.selfverify.refine, p['inference_method'], workers)
  solve(frames) = parallel.run(frames)        # groups of `workers` cached frames
  + one mandatory serial-initialisation frame before any parallel call.

P0  build 16 frames (4 per dev date) into a throwaway input cache under scratch
P1  one process: workers=2 (8 frames), workers=4 (16 frames), single-frame latency, peaks
P2  two shard processes, workers=4 each, 8 frames each -> aggregate
P3  same + frame loading on 4 prefetch threads per shard (realistic deployment shape)

Writes only /home/zhangcy/.dsh_scratch.  No /data2 writes, no ground truth.
"""
import concurrent.futures
import json
import multiprocessing
import os
import sys
import time
import traceback

SCRATCH = '/home/zhangcy/.dsh_scratch'
CACHE_DIR = SCRATCH + '/accel-input-cache'
EVAL = '/home/zhangcy/projects/leader-v5-eval-throughput-20261003'
V7FULL = '/home/zhangcy/projects/v7-full-20261006'
V7PLAN = '/data2/zhangcy/v7-full-20261006/plan.json'
CKPT = '/data2/zhangcy/v7-scratch-continue-20261009-v1/checkpoints/full-epoch7.pt'
INIT = '/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/initial.npz'
USED = {998, 1007, 1063, 1072, 2269, 2271, 3868, 3869, 3871, 3872}

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.chdir(SCRATCH)

import numpy as np                       # noqa: E402
import torch                             # noqa: E402

assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'

sys.path.insert(0, EVAL)
from runtime import Runtime                               # noqa: E402
import cache_runtime_v2 as crv                            # noqa: E402
sys.path.insert(0, V7FULL)
import v7_runtime as vr                                   # noqa: E402
import infer_batch_engine_v2 as engine                    # noqa: E402

import data.leader_nclt_manifest as _lnm                  # noqa: E402
from pathlib import Path                                  # noqa: E402
from pathlib import PurePosixPath                         # noqa: E402


def _safe_child_allow_symlink(root, date, relative_path):
    value = str(relative_path)
    relative = PurePosixPath(value)
    if ('\\' in value or ':' in value or '\x00' in value or relative.is_absolute()
            or not relative.parts or any(part in ('', '.', '..') for part in relative.parts)):
        raise ValueError('manifest paths must be clean relative paths')
    return Path(root) / date / Path(*relative.parts)


_lnm._safe_child = _safe_child_allow_symlink
SETTINGS = json.load(open(V7PLAN))['inference_method']


class _NoCache(object):
    def __init__(self, *a, **k):
        self.directory = None

    def load(self, *a, **k):
        raise AssertionError('InputCache disabled')

    def read(self, *a, **k):
        return None

    def write(self, *a, **k):
        raise AssertionError('InputCache disabled')


def _runtime(fraction):
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(fraction)
    c, parent, config, adaptive, Loader = vr.imports()
    real = crv.InputCache
    crv.InputCache = _NoCache
    denied = c.deny_test(config)
    test = vr.TestLoader(c, config)
    runtime = Runtime.__new__(Runtime)
    runtime.config = config
    runtime.plan = test.plan
    runtime.datasets = test.datasets
    runtime.records = test.records
    runtime.denied = denied
    crv.InputCache = real
    cache = real(runtime, CACHE_DIR, {'reason': 'dsh accel sweep', 'GT_used': False})
    return c, config, runtime, cache


def _leader(runtime):
    from tools.magic_nclt_runtime import load_pretrained_model
    runtime.leader, saved = load_pretrained_model(runtime.config['base_checkpoint'], False, torch.device('cuda:0'))
    runtime.leader.eval().requires_grad_(False)
    runtime.center = torch.as_tensor(saved['center_t'], device='cuda', dtype=torch.float32).reshape(3)
    del saved


def _raw_frame(runtime, index):
    rec = runtime.records[index]
    torch.cuda.synchronize()
    tick = time.monotonic()
    loaded = runtime.load(rec)
    initial = runtime.initial(loaded)
    geometry = runtime.geometry(loaded, initial[1].detach().cpu())
    frame = runtime.frame(loaded, *initial, geometry)
    torch.cuda.synchronize()
    return frame, time.monotonic() - tick


def _cached_frame(cache, runtime, index):
    rec = runtime.records[index]
    loaded, cached = cache.load(rec)
    arrays, metadata = cached
    frame = {name: torch.from_numpy(value).cuda() for name, value in arrays.items()}
    frame.update(input_sha256=metadata['shared_input_sha256'], global_index=index, key=rec[2],
                 date=rec[0], timestamp_us=int(loaded['sample']['timestamp_us']),
                 rgb=loaded['rgb'].cuda(),
                 sam=loaded['sample']['sam_features'].to(device='cuda', dtype=torch.float32),
                 hw=loaded['sample']['sam_metadata']['resized_size_hw'])
    return frame


def _pick_indices(runtime):
    with np.load(INIT, allow_pickle=False) as archive:
        keys = [str(k) for k in archive['keys']]
    by_date = {}
    for key in keys:
        by_date.setdefault(key.split('/')[0], []).append(key)
    picked = []
    for date in sorted(by_date):
        group = by_date[date]
        start = len(group) // 2
        picked.extend(group[start:start + 4])
    indices = []
    for key in picked:
        index = next(i for i, rec in enumerate(runtime.records) if rec[2] == key)
        if index not in USED:
            indices.append(index)
    return indices[:16]


def _shard_worker(indices, workers, prefetch, go, out):
    try:
        c, config, runtime, cache = _runtime(0.18)
        state = torch.load(CKPT, map_location='cpu')
        parallel = engine.ParallelFrames(lambda: vr.model(c, config, state['model'], False),
                                         c.selfverify.refine, SETTINGS, workers)
        del state
        torch.cuda.reset_peak_memory_stats()
        warm = _cached_frame(cache, runtime, indices[0])
        with torch.no_grad():
            observed = parallel.models[0].observation_batch([warm['sam']], [warm['rgb']],
                                                            [warm['pixels']], [warm['hw']])[0]
            c.selfverify.refine(parallel.models[0], warm, observed, SETTINGS)
        del warm, observed
        torch.cuda.empty_cache()
        out.put(('ready', os.getpid(), 0, 0.0, 0.0))
        go.wait()
        begun = time.monotonic()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=4) if prefetch else None
        pending = {}

        def submit(i):
            if pool is not None and i < len(indices):
                pending[i] = pool.submit(_cached_frame, cache, runtime, indices[i])

        for i in range(min(8, len(indices))):
            submit(i)
        solved = 0
        for start in range(0, len(indices), workers):
            group = list(range(start, min(start + workers, len(indices))))
            if pool is not None:
                batch = [pending.pop(i).result() for i in group]
                for i in range(start + workers, start + workers + 2 * workers):
                    submit(i)
            else:
                batch = [_cached_frame(cache, runtime, indices[i]) for i in group]
            with torch.no_grad():
                parallel.run(batch)
            solved += len(batch)
            del batch
        if pool is not None:
            pool.shutdown(wait=True)
        wall = time.monotonic() - begun
        out.put(('done', os.getpid(), solved, wall, round(torch.cuda.max_memory_allocated() / 2. ** 20, 1)))
    except Exception:
        out.put(('error', os.getpid(), traceback.format_exc(), 0.0, 0.0))


def run_shards(chunks, workers, prefetch, label):
    ctx = multiprocessing.get_context('spawn')
    out, go = ctx.Queue(), ctx.Event()
    procs = [ctx.Process(target=_shard_worker, args=(chunk, workers, prefetch, go, out)) for chunk in chunks]
    for p in procs:
        p.start()
    for _ in procs:
        message = out.get(timeout=900)
        if message[0] != 'ready':
            print('[%s] worker error:\n%s' % (label, message[2]))
            return None
    tick = time.monotonic()
    go.set()
    done = []
    for _ in procs:
        message = out.get(timeout=900)
        if message[0] == 'done':
            done.append(message)
        else:
            print('[%s] worker error:\n%s' % (label, message[2]))
    wall = time.monotonic() - tick
    for p in procs:
        p.join(timeout=60)
    frames = sum(m[2] for m in done)
    packet = dict(label=label, shards=len(done), workers=workers, prefetch=bool(prefetch),
                  frames=frames, wall=round(wall, 3), fps=round(frames / wall, 2) if wall else None,
                  ms_per_frame=round(1000 * wall / frames, 1) if frames else None,
                  gpu_MiB=round(sum(m[4] for m in done), 1))
    print('[%s] %s' % (label, json.dumps(packet)))
    sys.stdout.flush()
    return packet


def main():
    begun = time.monotonic()
    results = []

    c, config, runtime, cache = _runtime(0.35)
    _leader(runtime)
    indices = _pick_indices(runtime)
    raw, s1 = [], []
    with torch.no_grad():
        for index in indices:
            frame, seconds = _raw_frame(runtime, index)
            raw.append(frame)
            s1.append(seconds)
            if cache.read(runtime.records[index]) is None:
                cache.write(frame)
    dates = sorted({runtime.records[i][0] for i in indices})
    print('[P0] frames=%d dates=%s raw_stage1_steady=%.1f ms (first %.0f ms)'
          % (len(indices), dates, 1000 * sum(s1[1:]) / max(1, len(s1[1:])), 1000 * s1[0]))
    sys.stdout.flush()
    del raw
    torch.cuda.empty_cache()

    state = torch.load(CKPT, map_location='cpu')
    frames = [_cached_frame(cache, runtime, i) for i in indices]

    def build(workers):
        return engine.ParallelFrames(lambda: vr.model(c, config, state['model'], False),
                                     c.selfverify.refine, SETTINGS, workers)

    for workers, subset, label in ((2, frames[:8], 'P1-w2'), (4, frames, 'P1-w4')):
        torch.cuda.reset_peak_memory_stats()
        shard = build(workers)
        with torch.no_grad():
            warm = shard.models[0].observation_batch([subset[0]['sam']], [subset[0]['rgb']],
                                                     [subset[0]['pixels']], [subset[0]['hw']])[0]
            c.selfverify.refine(shard.models[0], subset[0], warm, SETTINGS)
            del warm
            tick = time.monotonic()
            for start in range(0, len(subset), workers):
                shard.run(subset[start:start + workers])
            torch.cuda.synchronize()
            wall = time.monotonic() - tick
            peak = torch.cuda.max_memory_allocated() / 2. ** 20
            latency = []
            for k in range(2):
                single = time.monotonic()
                shard.run([subset[k]])
                torch.cuda.synchronize()
                latency.append(1000 * (time.monotonic() - single))
        packet = dict(label=label, workers=workers, frames=len(subset), wall=round(wall, 3),
                      fps=round(len(subset) / wall, 2),
                      ms_per_frame=round(1000 * wall / len(subset), 1),
                      single_frame_latency_ms=round(sum(latency) / len(latency), 1),
                      gpu_MiB=round(peak, 1))
        results.append(packet)
        print('[%s] %s' % (label, json.dumps(packet)))
        sys.stdout.flush()
        del shard
        torch.cuda.empty_cache()
    del frames, state
    torch.cuda.empty_cache()

    half = len(indices) // 2
    for prefetch, label in ((False, 'P2-2shard-w4'), (True, 'P3-2shard-w4-prefetch')):
        packet = run_shards([indices[:half], indices[half:]], 4, prefetch, label)
        if packet:
            results.append(packet)

    print('[summary] ' + json.dumps(dict(complete=True, results=results,
                                         total_seconds=round(time.monotonic() - begun, 1))))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print('[summary] {"complete": false}')
        sys.exit(1)
