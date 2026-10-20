"""R4 shard runner: one process handles a contiguous block of frames, one JSON per frame.

    python r4_run.py --arm main  --worker 3 --of 24
    python r4_run.py --arm truth --worker 0 --of 4 --indices 0,15,30
    python r4_run.py --arm main  --limit 20 --of 1 --worker 0      # smoke

Resume: a frame whose output JSON already exists is skipped, so the exact same
command can simply be relaunched after an interruption.
"""
import argparse
import json
import os
import resource
import socket
import sys
import time
from pathlib import Path

# One BLAS thread per worker: the shared box must see (processes x threads) bounded.
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4_lib as L  # noqa: E402

OUT = Path('/data2/zhangcy/r4-full-20261019-v1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='main')          # main | truth
    ap.add_argument('--of', type=int, default=1)
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--indices', default='')
    ap.add_argument('--no-refine', action='store_true')
    ap.add_argument('--tag', default='')
    A = ap.parse_args()

    tag = A.tag or A.arm
    res = OUT / 'results' / tag
    res.mkdir(parents=True, exist_ok=True)
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    logf = OUT / 'logs' / ('%s.%03d.log' % (tag, A.worker))

    w = L.World()
    n = len(w.keys)
    order = sorted(range(n), key=w.sort_index)
    if A.indices:
        pool = [int(x) for x in A.indices.split(',') if x.strip()]
        block = np.array_split(np.array(pool), A.of)[A.worker].tolist()
    else:
        block = np.array_split(np.array(order), A.of)[A.worker].tolist()
    todo = [i for i in block if not (res / ('%s__%s.json' % tuple(w.keys[i].split('/')))).exists()]
    if A.limit:
        todo = todo[:A.limit]

    began = time.time()
    log = open(logf, 'a')

    def say(msg):
        line = '[%s w%03d %s] %s' % (tag, A.worker, time.strftime('%H:%M:%S'), msg)
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()

    say('start block=%d todo=%d (of %d) refine=%s host=%s pid=%d'
        % (len(block), len(todo), n, not A.no_refine, socket.gethostname(), os.getpid()))
    done = 0
    for i in todo:
        key = w.keys[i]
        path = res / ('%s__%s.json' % tuple(key.split('/')))
        try:
            rec = L.process_frame(w, i, i, u_source=A.arm,
                                  do_refine=not A.no_refine,
                                  do_truth_probe=(A.arm == 'main'))
            rec['worker'] = A.worker
            rec['pid'] = os.getpid()
            L.dump(rec, path)
            done += 1
            if done % 10 == 0 or done == len(todo):
                el = time.time() - began
                say('%d/%d  %.1fs  %.2f s/frame  rss=%.0fMB  last=%s s=%+.1f d=%+.2f '
                    'cost=%.3f dep=%.3f mp=%d abst=%s'
                    % (done, len(todo), el, el / done,
                       resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                       key, rec['s'], rec['d'],
                       rec['cost_selected'] if rec['cost_selected'] is not None else -1,
                       rec['cost_deploy'] if rec['cost_deploy'] is not None else -1,
                       rec['map_points'], rec['abstain']))
        except Exception as exc:
            import traceback
            say('FAIL %s %s: %s' % (key, type(exc).__name__, exc))
            log.write(traceback.format_exc() + '\n')
            log.flush()
    el = time.time() - began
    say('done %d frames in %.1fs (%.2f s/frame) rss_peak=%.0fMB'
        % (done, el, el / max(done, 1),
           resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0))
    log.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
