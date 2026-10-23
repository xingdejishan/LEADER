#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage-1 scaling only: 1 worker x 3 frames vs 4 workers x 2 frames.

No parent bootstrap: the parent does not touch CUDA at all, so nothing is
inherited and spawn children are clean.  Read-only apart from stdout.
"""
import json
import multiprocessing
import os
import sys
import time
import traceback

SCRATCH = '/home/zhangcy/.dsh_scratch'
EVAL = '/home/zhangcy/projects/leader-v5-eval-throughput-20261003'
V7FULL = '/home/zhangcy/projects/v7-full-20261006'
INDICES = [1063, 1072, 2269, 2271, 3868, 3869, 3871, 3872]

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.chdir(SCRATCH)

import torch                             # noqa: E402

assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'

sys.path.insert(0, EVAL)
from runtime import Runtime                               # noqa: E402
sys.path.insert(0, V7FULL)
import v7_runtime as vr                                   # noqa: E402
import cache_runtime_v2                                   # noqa: E402

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
        del saved
        out.put(('ready', os.getpid(), 0, 0))
        go.wait()
        rows = []
        with torch.no_grad():
            for index in indices:
                frame, seconds = _stage1(runtime, runtime.records[index])
                rows.append((index, frame['key'], round(seconds, 4)))
                del frame
        out.put(('done', os.getpid(), rows, round(torch.cuda.max_memory_allocated() / 2. ** 20, 1)))
    except Exception:
        out.put(('error', os.getpid(), traceback.format_exc(), 0))


def run_workers(chunks, label):
    ctx = multiprocessing.get_context('spawn')
    out, go = ctx.Queue(), ctx.Event()
    workers = [ctx.Process(target=_worker, args=(chunk, go, out)) for chunk in chunks]
    for w in workers:
        w.start()
    ready = 0
    while ready < len(workers):
        message = out.get(timeout=600)
        if message[0] == 'error':
            print('[%s] worker error:\n%s' % (label, message[2]))
            return float('nan'), []
        ready += 1
    tick = time.monotonic()
    go.set()
    done = []
    for _ in workers:
        message = out.get(timeout=600)
        if message[0] == 'done':
            done.append(message)
        else:
            print('[%s] worker error:\n%s' % (label, message[2]))
    wall = time.monotonic() - tick
    for w in workers:
        w.join(timeout=60)
    rows = [row for _, _, rs, _ in done for row in rs]
    print('[%s] workers=%d frames=%d wall=%.3f s -> %.1f ms/frame amortized (%.2f x serial)'
          % (label, len(done), len(rows), wall, 1000 * wall / max(1, len(rows)),
             (sum(r[2] for r in rows) / max(1, len(rows))) / (wall / max(1, len(rows)))))
    print('[%s] per_frame=%s  peaks_MiB=%s' % (label, [round(r[2], 3) for r in rows],
                                               [p for _, _, _, p in done]))
    sys.stdout.flush()
    return wall, rows


def main():
    begun = time.monotonic()
    wall1, rows1 = run_workers([INDICES[:3]], 'C1')
    wall4, rows4 = run_workers([INDICES[0:2], INDICES[2:4], INDICES[4:6], INDICES[6:8]], 'C4')
    print('[summary] ' + json.dumps(dict(complete=True,
                                         c1_ms=round(1000 * wall1 / max(1, len(rows1)), 1),
                                         c4_ms=round(1000 * wall4 / max(1, len(rows4)), 1),
                                         c4_frames=len(rows4),
                                         total_seconds=round(time.monotonic() - begun, 1))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
