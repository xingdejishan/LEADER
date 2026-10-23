#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Second sweep: cached operating point (E) + clean parallel stage 1 (C) + batched stage 2 (B).

E  the deployed "cached" stage 1: InputCache.load + cached-frame build (jpg decode + SAM read
   + sha re-verification of .bin/.jpg)  <- what the 19.35 frames/s deployment actually did
B  observation_batch(8) + selfverify.refine_batch(8)
C1 1 worker process x 3 frames, raw stage 1 (SC2 + visible pool)
C4 4 worker processes x 2 frames, raw stage 1

Writes only inside /home/zhangcy/.dsh_scratch (a throwaway 8-entry input cache).
"""
import hashlib
import json
import multiprocessing
import os
import sys
import time
import traceback

SCRATCH = '/home/zhangcy/.dsh_scratch'
CACHE_DIR = SCRATCH + '/tiny-input-cache'
EVAL = '/home/zhangcy/projects/leader-v5-eval-throughput-20261003'
V7FULL = '/home/zhangcy/projects/v7-full-20261006'
V7PLAN = '/data2/zhangcy/v7-full-20261006/plan.json'
CKPT = '/data2/zhangcy/v7-scratch-continue-20261009-v1/checkpoints/full-epoch7.pt'
INIT = '/home/zhangcy/projects/e2e-moe-20261020-v1/inputs/initial.npz'

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.chdir(SCRATCH)

import numpy as np                       # noqa: E402
import torch                             # noqa: E402

assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'

sys.path.insert(0, EVAL)
from runtime import Runtime                               # noqa: E402
from cache_runtime_v2 import InputCache                   # noqa: E402
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


class _NoCache(object):
    def __init__(self, *a, **k):
        self.directory = None

    def load(self, *a, **k):
        raise AssertionError('InputCache disabled')

    def read(self, *a, **k):
        return None

    def write(self, *a, **k):
        raise AssertionError('InputCache disabled')


def _runtime(c, config):
    test = vr.TestLoader(c, config)
    runtime = Runtime.__new__(Runtime)
    runtime.config = config
    runtime.plan = test.plan
    runtime.datasets = test.datasets
    runtime.records = test.records
    runtime.denied = []
    return runtime


def _leader(runtime):
    from tools.magic_nclt_runtime import load_pretrained_model
    runtime.leader, saved = load_pretrained_model(runtime.config['base_checkpoint'], False, torch.device('cuda:0'))
    runtime.leader.eval().requires_grad_(False)
    runtime.center = torch.as_tensor(saved['center_t'], device='cuda', dtype=torch.float32).reshape(3)


def _stage1(runtime, rec):
    torch.cuda.synchronize()
    tick = time.monotonic()
    loaded = runtime.load(rec)
    initial = runtime.initial(loaded)
    geometry = runtime.geometry(loaded, initial[1].detach().cpu())
    frame = runtime.frame(loaded, *initial, geometry)
    torch.cuda.synchronize()
    return frame, time.monotonic() - tick


def _worker(indices, go, out):
    try:
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(0.1)
        c, parent, config, adaptive, Loader = vr.imports()
        import cache_runtime_v2
        cache_runtime_v2.InputCache = _NoCache
        denied = c.deny_test(config)
        runtime = _runtime(c, config)
        runtime.denied = denied
        _leader(runtime)
        out.put(('ready', os.getpid(), 0))
        go.wait()
        rows = []
        with torch.no_grad():
            for index in indices:
                frame, seconds = _stage1(runtime, runtime.records[index])
                rows.append((index, frame['key'], round(seconds, 4)))
                del frame
        out.put(('done', os.getpid(), rows, round(torch.cuda.max_memory_allocated() / 2. ** 20, 1)))
    except Exception:
        out.put(('error', os.getpid(), traceback.format_exc()))


def run_workers(chunks):
    ctx = multiprocessing.get_context('spawn')
    out, go = ctx.Queue(), ctx.Event()
    workers = [ctx.Process(target=_worker, args=(chunk, go, out)) for chunk in chunks]
    for w in workers:
        w.start()
    ready = 0
    while ready < len(workers):
        message = out.get(timeout=420)
        if message[0] == 'error':
            print('[C] worker error:\n' + message[2])
        ready += 1
    tick = time.monotonic()
    go.set()
    got = []
    for _ in workers:
        message = out.get(timeout=420)
        if message[0] == 'done':
            got.append(message)
        else:
            print('[C] worker error:\n' + message[2])
    wall = time.monotonic() - tick
    for w in workers:
        w.join(timeout=60)
    return wall, got


def main():
    begun = time.monotonic()
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.2)
    c, parent, config, adaptive, Loader = vr.imports()
    cache_runtime_v2 = __import__('cache_runtime_v2')
    real_cache = cache_runtime_v2.InputCache
    cache_runtime_v2.InputCache = _NoCache
    denied = c.deny_test(config)
    runtime = _runtime(c, config)
    runtime.denied = denied
    _leader(runtime)
    cache_runtime_v2.InputCache = real_cache
    cache = real_cache(runtime, CACHE_DIR, {'reason': 'dsh cost sweep, scratch only', 'GT_used': False})

    state = torch.load(CKPT, map_location='cpu')
    m = vr.model(c, config, state['model'], False)
    del state
    settings = json.load(open(V7PLAN))['inference_method']

    with np.load(INIT, allow_pickle=False) as archive:
        keys = [str(k) for k in archive['keys']][2:10]
    indices = [next(i for i, rec in enumerate(runtime.records) if rec[2] == k) for k in keys]
    print('[boot] seconds=%.1f indices=%s' % (time.monotonic() - begun, indices))
    sys.stdout.flush()

    # raw stage 1 once, and populate the throwaway cache
    frames, s1 = [], []
    with torch.no_grad():
        for index in indices:
            frame, seconds = _stage1(runtime, runtime.records[index])
            frames.append(frame)
            s1.append(seconds)
            if cache.read(runtime.records[index]) is None:
                cache.write(frame)
    print('[A] raw stage1 s/frame steady=%.4f  first=%.4f  all=%s'
          % (sum(s1[1:]) / len(s1[1:]), s1[0], [round(x, 3) for x in s1]))

    # E: deployed cached stage 1
    e = []
    for index in indices:
        rec = runtime.records[index]
        torch.cuda.synchronize()
        tick = time.monotonic()
        loaded, cached = cache.load(rec)
        arrays, metadata = cached
        frame = {name: torch.from_numpy(value).cuda() for name, value in arrays.items()}
        frame.update(rgb=loaded['rgb'].cuda(),
                     sam=loaded['sample']['sam_features'].to(device='cuda', dtype=torch.float32),
                     hw=loaded['sample']['sam_metadata']['resized_size_hw'])
        torch.cuda.synchronize()
        e.append(time.monotonic() - tick)
        assert frame['points'].shape == frames[indices.index(index)]['points'].shape
        del loaded, cached, arrays, frame
    print('[E] cached stage1 (jpg+SAM+sha re-verify+npz) %.1f ms/frame  all=%s'
          % (1000 * sum(e) / len(e), [round(1000 * x) for x in e]))
    sys.stdout.flush()

    # B: batched stage 2
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    tick = time.monotonic()
    observations = m.observation_batch([f['sam'] for f in frames], [f['rgb'] for f in frames],
                                       [f['pixels'] for f in frames], [f['hw'] for f in frames])
    torch.cuda.synchronize()
    t_obs = time.monotonic() - tick
    tick = time.monotonic()
    engine.refine_batch(m, frames, observations, settings)
    torch.cuda.synchronize()
    t_ref = time.monotonic() - tick
    peak = torch.cuda.max_memory_allocated() / 2. ** 20
    print('[B] n=8 obs=%.4f s refine_batch=%.4f s -> %.1f ms/frame  peak_MiB=%.1f'
          % (t_obs, t_ref, 1000 * (t_obs + t_ref) / 8, peak))
    del observations
    del frames
    torch.cuda.empty_cache()
    sys.stdout.flush()

    # C: parallel raw stage 1
    wall1, got1 = run_workers([indices[:3]])
    rows1 = [row for _, rows, _ in got1 for row in rows]
    print('[C1] 1 worker x %d frames wall=%.3f s -> %.1f ms/frame  all=%s'
          % (len(rows1), wall1, 1000 * wall1 / max(1, len(rows1)), [round(row[2], 3) for row in rows1]))
    sys.stdout.flush()
    wall4, got4 = run_workers([indices[0:2], indices[2:4], indices[4:6], indices[6:8]])
    rows4 = [row for _, rows, _ in got4 for row in rows]
    print('[C4] 4 workers x 2 frames wall=%.3f s -> %.1f ms/frame amortized  all=%s'
          % (wall4, 1000 * wall4 / max(1, len(rows4)), sorted(round(row[2], 3) for row in rows4)))
    print('[C4] worker peaks MiB=%s' % [p for _, _, p in got4])

    e_ms = 1000 * sum(e) / len(e)
    s2_ms = 1000 * (t_obs + t_ref) / 8
    print('[D] serial raw 1x1        = %.0f ms/frame' % (1000 * (sum(s1[1:]) / len(s1[1:]) + 0.198)))
    print('[D] cached stage1 + batch8 = %.0f ms/frame (max of %.0f, %.0f)' % (max(e_ms, s2_ms), e_ms, s2_ms))
    print('[D] 4 procs raw stage1     = %.0f ms/frame' % (1000 * wall4 / max(1, len(rows4))))
    print('[D] 52 ms reachable with 8 frames: %s' % ('YES' if max(e_ms, s2_ms) <= 52 else 'NO'))
    print('[summary] ' + json.dumps(dict(complete=True, a_s1_steady=round(sum(s1[1:]) / len(s1[1:]), 4),
                                         e_ms=round(e_ms, 1), s2_ms=round(s2_ms, 1),
                                         c1_ms=round(1000 * wall1 / max(1, len(rows1)), 1),
                                         c4_ms=round(1000 * wall4 / max(1, len(rows4)), 1),
                                         total_seconds=round(time.monotonic() - begun, 1))))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print('[summary] {"complete": false}')
        sys.exit(1)
