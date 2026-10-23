#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shard scaling on one card: N shards x workers=4 over the warm input cache.

Equal work per shard (3 frames each) for both N=4 and N=6 so the comparison is not
biased by per-shard replica warm-up.  Total GPU memory is read from nvidia-smi
(includes per-process CUDA context, which torch's own counters miss).

Writes: nothing.  GPU 2 only, per-process memory fraction 0.15.
"""
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
import traceback

SCRATCH = '/home/zhangcy/.dsh_scratch'
CACHE_DIR = SCRATCH + '/accel-input-cache'
EVAL = '/home/zhangcy/projects/leader-v5-eval-throughput-20261003'
V7FULL = '/home/zhangcy/projects/v7-full-20261006'
V7PLAN = '/data2/zhangcy/v7-full-20261006/plan.json'
CKPT = '/data2/zhangcy/v7-scratch-continue-20261009-v1/checkpoints/full-epoch7.pt'
PER_SHARD = 3
FRACTION = 0.15

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.chdir(SCRATCH)

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

QUERY = ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits', '-i', '2']


def _safe_child_allow_symlink(root, date, relative_path):
    value = str(relative_path)
    relative = PurePosixPath(value)
    if ('\\' in value or ':' in value or '\x00' in value or relative.is_absolute()
            or not relative.parts or any(part in ('', '.', '..') for part in relative.parts)):
        raise ValueError('manifest paths must be clean relative paths')
    return Path(root) / date / Path(*relative.parts)


_lnm._safe_child = _safe_child_allow_symlink
SETTINGS = json.load(open(V7PLAN))['inference_method']
REAL_CACHE = crv.InputCache          # captured before any _NoCache patch


class _NoCache(object):
    def __init__(self, *a, **k):
        self.directory = None


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


def _worker(indices, go, out):
    try:
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(FRACTION)
        c, parent, config, adaptive, Loader = vr.imports()
        crv.InputCache = _NoCache
        denied = c.deny_test(config)
        test = vr.TestLoader(c, config)
        runtime = Runtime.__new__(Runtime)
        runtime.config = config
        runtime.plan = test.plan
        runtime.datasets = test.datasets
        runtime.records = test.records
        runtime.denied = denied
        cache = REAL_CACHE(runtime, CACHE_DIR, {'reason': 'dsh accel scaling', 'GT_used': False})
        state = torch.load(CKPT, map_location='cpu')
        parallel = engine.ParallelFrames(lambda: vr.model(c, config, state['model'], False),
                                         c.selfverify.refine, SETTINGS, 4)
        del state
        warm = _cached_frame(cache, runtime, indices[0])
        with torch.no_grad():
            observed = parallel.models[0].observation_batch([warm['sam']], [warm['rgb']],
                                                            [warm['pixels']], [warm['hw']])[0]
            c.selfverify.refine(parallel.models[0], warm, observed, SETTINGS)
        del warm, observed
        torch.cuda.empty_cache()
        out.put(('ready', os.getpid()))
        go.wait()
        begun = time.monotonic()
        solved = 0
        for start in range(0, len(indices), 4):
            batch = [_cached_frame(cache, runtime, i) for i in indices[start:start + 4]]
            with torch.no_grad():
                parallel.run(batch)
            solved += len(batch)
            del batch
        wall = time.monotonic() - begun
        out.put(('done', os.getpid(), solved, wall))
    except Exception:
        out.put(('error', os.getpid(), traceback.format_exc(), 0.0))


def sample(stop, series):
    while not stop.is_set():
        try:
            out = subprocess.run(QUERY, capture_output=True, text=True, timeout=5).stdout.strip()
            if out:
                used, util = [float(x.strip()) for x in out.split(',')[:2]]
                series.append((used, util))
        except Exception:
            pass
        time.sleep(0.2)


def run(shards, indices):
    chunks = [[indices[(i * PER_SHARD + j) % len(indices)] for j in range(PER_SHARD)] for i in range(shards)]
    ctx = multiprocessing.get_context('spawn')
    out, go = ctx.Queue(), ctx.Event()
    series = []
    stop = threading.Event()
    watcher = threading.Thread(target=sample, args=(stop, series))
    watcher.start()
    time.sleep(0.6)
    baseline = series[-1][0] if series else float('nan')
    procs = [ctx.Process(target=_worker, args=(chunk, go, out)) for chunk in chunks]
    for p in procs:
        p.start()
    for _ in procs:
        message = out.get(timeout=900)
        if message[0] != 'ready':
            print('[N=%d] worker error:\n%s' % (shards, message[2]))
            stop.set()
            return None
    tick = time.monotonic()
    go.set()
    done = []
    for _ in procs:
        message = out.get(timeout=900)
        if message[0] == 'done':
            done.append(message)
        else:
            print('[N=%d] worker error:\n%s' % (shards, message[2]))
    wall = time.monotonic() - tick
    stop.set()
    watcher.join()
    for p in procs:
        p.join(timeout=60)
    frames = sum(m[2] for m in done)
    peak = max(s[0] for s in series)
    util = [s[1] for s in series]
    packet = dict(shards=shards, frames=frames, wall=round(wall, 3), fps=round(frames / wall, 2),
                  ms_per_frame=round(1000 * wall / frames, 1),
                  gpu_total_MiB=round(peak, 0), gpu_baseline_MiB=round(baseline, 0),
                  gpu_delta_GB=round((peak - baseline) / 1024., 2),
                  per_shard_GB=round((peak - baseline) / 1024. / shards, 3),
                  util_mean=round(sum(util) / len(util), 1), util_max=max(util))
    print('[N=%d] %s' % (shards, json.dumps(packet)))
    sys.stdout.flush()
    return packet


def main():
    indices = sorted(int(p.stem) for p in Path(CACHE_DIR).glob('*.npz'))
    print('[pool] cached frames=%d' % len(indices))
    results = []
    for shards in (4, 6):
        packet = run(shards, indices)
        if packet:
            results.append(packet)
        time.sleep(2.0)
    print('[summary] ' + json.dumps(dict(complete=True, results=results)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
