#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""8-frame replicate of the 2-shard x workers=4 shape with live GPU/CPU sampling.

Reuses the input cache built by accel_sweep.py (no new stage-1 frames).
Answers: is the aggregate plateau GPU-bound or CPU/tenant-contention-bound?
"""
import json
import multiprocessing
import subprocess
import sys
import threading
import time
from pathlib import Path

import accel_sweep as A                                 # reuse the tested shard worker

QUERY = ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,clocks.sm,temperature.gpu,power.draw',
         '--format=csv,noheader,nounits', '-i', '2']


def sample(stop, series):
    while not stop.is_set():
        try:
            out = subprocess.run(QUERY, capture_output=True, text=True, timeout=5).stdout.strip()
            if out:
                series.append(out)
        except Exception:
            pass
        time.sleep(0.25)


def main():
    indices = sorted(int(p.stem) for p in Path(A.CACHE_DIR).glob('*.npz'))
    print('[probe] cached frames=%d -> %s' % (len(indices), indices))
    picked = indices[:8]
    chunks = [picked[:4], picked[4:8]]

    ctx = multiprocessing.get_context('spawn')
    out, go = ctx.Queue(), ctx.Event()
    procs = [ctx.Process(target=A._shard_worker, args=(chunk, 4, True, go, out)) for chunk in chunks]
    for p in procs:
        p.start()
    for _ in procs:
        message = out.get(timeout=900)
        if message[0] != 'ready':
            print('[probe] worker error:\n%s' % message[2])
            return 1

    stop, series = threading.Event(), []
    watcher = threading.Thread(target=sample, args=(stop, series))
    watcher.start()
    tick = time.monotonic()
    go.set()
    done = []
    for _ in procs:
        message = out.get(timeout=900)
        if message[0] == 'done':
            done.append(message)
        else:
            print('[probe] worker error:\n%s' % message[2])
    wall = time.monotonic() - tick
    stop.set()
    watcher.join()
    for p in procs:
        p.join(timeout=60)

    frames = sum(m[2] for m in done)
    gpu = sum(m[4] for m in done)
    print('[probe] frames=%d wall=%.3f s fps=%.2f ms/frame=%.1f gpu_MiB=%.0f'
          % (frames, wall, frames / wall, 1000 * wall / frames, gpu))
    stats = []
    for line in series:
        parts = [x.strip() for x in line.split(',')]
        if len(parts) >= 5:
            try:
                stats.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
            except ValueError:
                pass
    if stats:
        util = [s[0] for s in stats]
        print('[probe] nvidia-smi samples=%d util%% mean=%.1f p95=%.0f max=%.0f | clocks_MHz mean=%.0f | '
              'power_W mean=%.0f | mem_MiB max=%.0f'
              % (len(stats), sum(util) / len(util), sorted(util)[int(.95 * (len(util) - 1))], max(util),
                 sum(s[2] for s in stats) / len(stats), sum(s[4] for s in stats) / len(stats),
                 max(s[1] for s in stats)))
    print('[probe] loadavg=%s nproc=%d'
          % (Path('/proc/loadavg').read_text().split()[:3], len(__import__('os').sched_getaffinity(0))))
    print('[summary] ' + json.dumps(dict(complete=True, frames=frames, wall=round(wall, 3),
                                         fps=round(frames / wall, 2),
                                         ms_per_frame=round(1000 * wall / frames, 1),
                                         gpu_MiB=round(gpu, 1),
                                         util_mean=round(sum(s[0] for s in stats) / len(stats), 1) if stats else None,
                                         util_max=max(s[0] for s in stats) if stats else None,
                                         clocks_mean=round(sum(s[2] for s in stats) / len(stats)) if stats else None)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
