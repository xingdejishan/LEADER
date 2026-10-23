"""Measure end-to-end online throughput with N shard processes on one GPU."""
import json
import os
import subprocess
import sys
import time

PROJ = '/home/zhangcy/projects/e2e-moe-20261020-v1'
PY = '/home/zhangcy/.conda/envs/leader/bin/python'
DATES = '2012-02-12,2012-02-19'
LIMIT = 10


def run(shards):
    logs = open('/home/zhangcy/.dsh_scratch/bench_%d.log' % shards, 'w')
    procs = []
    t0 = time.monotonic()
    for i in range(shards):
        cmd = [PY, 'e2e_online.py', '--shard', '%d/%d' % (i, shards),
               '--limit', str(LIMIT), '--dates', DATES,
               '--out-prefix', 'results/acc%d' % shards]
        procs.append(subprocess.Popen(cmd, cwd=PROJ, stdout=subprocess.DEVNULL,
                                      stderr=logs, env=dict(os.environ)))
    for p in procs:
        p.wait()
    wall = time.monotonic() - t0
    logs.close()
    frames, medians = 0, []
    for i in range(shards):
        path = os.path.join(PROJ, 'results/acc%d-shard%d.jsonl' % (shards, i))
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path)]
        frames += len(rows)
        for r in rows:
            medians.append(float(r['total']))
    medians.sort()
    med = medians[len(medians) // 2] if medians else float('nan')
    print('shards=%d  frames=%d  wall=%.1fs  ms/frame=%.0f  frames/s=%.2f  '
          'per-frame median=%.3fs'
          % (shards, frames, wall, 1000 * wall / max(frames, 1),
             frames / max(wall, 1e-9), med))
    return frames, wall


if __name__ == '__main__':
    for n in (1, 2, 4):
        run(n)
