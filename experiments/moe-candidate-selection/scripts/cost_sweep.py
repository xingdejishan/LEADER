#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Operating-point sweep for the v7 upstream (stage 1 LEADER -> stage 2 field refinement).

A  serial single process / single thread, 8 frames
B  stage-2 batched: observation_batch(8) + selfverify.refine_batch(8) from infer_batch_engine_v2
C  stage 1 with 4 worker processes (spawn), same 8 frames
D  derived pipeline numbers + verdict vs 52 ms/frame

Read-only: no InputCache, no writes, no ground truth.
"""
import hashlib
import json
import multiprocessing
import os
import sys
import time
import traceback

SCRATCH = '/home/zhangcy/.dsh_scratch'
EVAL = '/home/zhangcy/projects/leader-v5-eval-throughput-20261003'
V7FULL = '/home/zhangcy/projects/v7-full-20261006'
V7PLAN = '/data2/zhangcy/v7-full-20261006/plan.json'
CKPT = '/data2/zhangcy/v7-scratch-continue-20261009-v1/checkpoints/full-epoch7.pt'
CKPT_SHA = '4df0c1c98292edec43acde531bd57d0dfbea9db2c0c4530bba9c5aff78a939d8'
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
sys.path.insert(0, V7FULL)
import v7_runtime as vr                                   # noqa: E402
import cache_runtime_v2                                   # noqa: E402
import infer_batch_engine_v2 as engine                    # noqa: E402

# symlinked velodyne_sync shim (see smoke_stage12.py); script-local, no project edit
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


def _base(fraction):
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(fraction)
    c, parent, config, adaptive, Loader = vr.imports()
    cache_runtime_v2.InputCache = _NoCache
    denied = c.deny_test(config)
    test = vr.TestLoader(c, config)
    runtime = Runtime.__new__(Runtime)
    runtime.config = config
    runtime.plan = test.plan
    runtime.datasets = test.datasets
    runtime.records = test.records
    runtime.denied = denied
    from tools.magic_nclt_runtime import load_pretrained_model
    runtime.leader, saved = load_pretrained_model(config['base_checkpoint'], False, torch.device('cuda:0'))
    runtime.leader.eval().requires_grad_(False)
    runtime.center = torch.as_tensor(saved['center_t'], device='cuda', dtype=torch.float32).reshape(3)
    del saved
    return c, config, runtime


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
    """Stage 1 in a fresh process: leader only, no FAR model."""
    try:
        c, config, runtime = _base(0.1)
        out.put(('ready', os.getpid(), time.monotonic()))
        go.wait()
        rows = []
        with torch.no_grad():
            for index in indices:
                frame, seconds = _stage1(runtime, runtime.records[index])
                rows.append((index, frame['key'], round(seconds, 4), frame['input_sha256'][:8]))
                del frame
                torch.cuda.empty_cache()
        out.put(('done', os.getpid(), rows, round(torch.cuda.max_memory_allocated() / 2. ** 20, 1)))
    except Exception:
        out.put(('error', os.getpid(), traceback.format_exc()))


def main():
    begun = time.monotonic()
    c, config, runtime = _base(0.2)

    state = torch.load(CKPT, map_location='cpu')
    assert hashlib.sha256(open(CKPT, 'rb').read()).hexdigest() == CKPT_SHA
    m = vr.model(c, config, state['model'], False)
    del state
    settings = json.load(open(V7PLAN))['inference_method']

    with np.load(INIT, allow_pickle=False) as archive:
        ref_keys = [str(k) for k in archive['keys']]
        ref_poses = np.asarray(archive['poses'], float)
    keys = ref_keys[2:10]
    indices = [next(i for i, rec in enumerate(runtime.records) if rec[2] == k) for k in keys]
    print('[boot] seconds=%.1f  frames=%d  indices=%s' % (time.monotonic() - begun, len(keys), indices))
    sys.stdout.flush()

    # ---------------- A: serial single process, single thread ----------------
    frames = []
    s1 = []
    with torch.no_grad():
        for index in indices:
            frame, seconds = _stage1(runtime, runtime.records[index])
            frames.append(frame)
            s1.append(seconds)
    s2 = []
    poses_serial = []
    for frame in frames:
        torch.cuda.synchronize()
        tick = time.monotonic()
        observed = m.observation_batch([frame['sam']], [frame['rgb']], [frame['pixels']], [frame['hw']])[0]
        pose, _, _, info = c.selfverify.refine(m, frame, observed, settings)
        torch.cuda.synchronize()
        s2.append(time.monotonic() - tick)
        poses_serial.append(pose.detach().cpu().double().numpy().copy())
        del observed, pose
    a_stage1 = sum(s1) / len(s1)
    a_stage2 = sum(s2) / len(s2)
    print('[A] stage1=%.4f s/frame  stage2=%.4f s/frame  total=%.4f s/frame  (%.1f ms)  per_frame_s1=%s'
          % (a_stage1, a_stage2, a_stage1 + a_stage2, 1000 * (a_stage1 + a_stage2),
             [round(x, 3) for x in s1]))
    print('[A] per_frame_s2=%s  peak_MiB=%.1f' % ([round(x, 3) for x in s2],
                                                  torch.cuda.max_memory_allocated() / 2. ** 20))
    sys.stdout.flush()

    # ---------------- B: batched stage 2 ----------------
    batch = {}
    for n in (8, 4):
        try:
            picked = frames[:n]
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            tick = time.monotonic()
            observations = m.observation_batch([f['sam'] for f in picked], [f['rgb'] for f in picked],
                                               [f['pixels'] for f in picked], [f['hw'] for f in picked])
            torch.cuda.synchronize()
            t_obs = time.monotonic() - tick
            tick = time.monotonic()
            results = engine.refine_batch(m, picked, observations, settings)
            torch.cuda.synchronize()
            t_ref = time.monotonic() - tick
            peak = torch.cuda.max_memory_allocated() / 2. ** 20
            batched = [r[0].detach().cpu().double().numpy().copy() for r in results]
            delta = max(float(np.abs(b - a).max()) for b, a in zip(batched, poses_serial[:n]))
            batch[n] = dict(obs=t_obs, refine=t_ref, total=t_obs + t_ref, peak=peak, delta=delta)
            print('[B] n=%d obs=%.4f s  refine_batch=%.4f s  stage2=%.4f s  -> %.1f ms/frame  peak_MiB=%.1f  max|pose-serial|=%.2e'
                  % (n, t_obs, t_ref, t_obs + t_ref, 1000 * (t_obs + t_ref) / n, peak, delta))
            del observations, results, batched
        except Exception:
            traceback.print_exc()
            print('[B] n=%d FAILED' % n)
        torch.cuda.empty_cache()
    sys.stdout.flush()

    # free the parent's GPU frames before the workers start
    del frames
    torch.cuda.empty_cache()

    # ---------------- C: 4 stage-1 worker processes ----------------
    ctx = multiprocessing.get_context('spawn')
    out = ctx.Queue()
    go = ctx.Event()
    chunks = [indices[i::4] for i in range(4)]
    workers = [ctx.Process(target=_worker, args=(chunk, go, out)) for chunk in chunks]
    for w in workers:
        w.start()
    ready = 0
    errors = []
    while ready < len(workers):
        try:
            message = out.get(timeout=420)
        except Exception:
            errors.append('worker did not report ready within 420 s')
            break
        if message[0] == 'ready':
            ready += 1
        elif message[0] == 'error':
            errors.append(message[2])
            ready += 1
    if errors:
        print('[C] worker error:\n' + errors[0])
    tick = time.monotonic()
    go.set()
    per_worker = []
    for _ in workers:
        try:
            message = out.get(timeout=420)
        except Exception:
            break
        if message[0] == 'done':
            per_worker.append((message[1], message[2], message[3]))
        else:
            errors.append(message[2])
    wall = time.monotonic() - tick
    for w in workers:
        w.join(timeout=60)
    flat = [row for _, rows, _ in per_worker for row in rows]
    worker_peak = sum(p for _, _, p in per_worker)
    c_stage1 = (sum(row[2] for row in flat) / len(flat)) if flat else float('nan')
    c_wall = (wall / float(len(indices))) if flat else float('nan')
    print('[C] workers=%d frames=%d wall=%.3f s -> %.4f s/frame wall-amortized; mean per-frame s1=%.4f s; sum_peak_MiB=%.0f'
          % (len(per_worker), len(flat), wall, c_wall, c_stage1, worker_peak))
    print('[C] per_frame=%s' % sorted(round(row[2], 3) for row in flat))
    sys.stdout.flush()

    # ---------------- D: derived ----------------
    b8 = batch.get(8)
    b4 = batch.get(4)
    stage2_batch = (b8['total'] / 8.) if b8 else a_stage2
    pipeline = max(c_wall, stage2_batch)
    verdict = pipeline
    print('[D] serial 1x1 = %.1f ms/frame' % (1000 * (a_stage1 + a_stage2)))
    print('[D] 4 loader procs + refine_batch8 pipelined = %.1f ms/frame (max of %.1f and %.1f)'
          % (1000 * pipeline, 1000 * c_wall, 1000 * stage2_batch))
    print('[D] 52 ms/frame reachable at 8 frames: %s' % ('YES' if verdict <= 0.052 else 'NO'))
    print('[D] concurrency implied by 52 ms: stage1 needs %.1fx (%.0f procs), stage2 needs %.1fx (%.0f in flight)'
          % (a_stage1 / 0.052, a_stage1 / 0.052, a_stage2 / 0.052, a_stage2 / 0.052))
    print('[summary] ' + json.dumps(dict(complete=True, a_stage1=round(a_stage1, 4), a_stage2=round(a_stage2, 4),
                                         b8=b8, b4=b4, c_wall_per_frame=round(c_wall, 4),
                                         c_mean_s1=round(c_stage1, 4), pipeline_ms=round(1000 * pipeline, 1),
                                         total_seconds=round(time.monotonic() - begun, 1))))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print('[summary] {"complete": false}')
        sys.exit(1)
