#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark harness for e2e_online.py: N concurrent pipelined processes.

Steady state is measured inside each process: wall clock between the first and
the last completed *steady* frame (frame 0 of each process is dropped, it carries
model load and pipeline fill).  Aggregate throughput is the sum of the
per-process steady rates, since the processes run concurrently.

  python bench_pipeline.py p1:1:100 p2:2:100 p4:4:100 p1r4:1:100:--refine-replicas=4
  BENCH_REUSE=1 python bench_pipeline.py p1:1:100      # re-aggregate only
"""
import json
import os
import subprocess
import sys
import time

PROJ = '/home/zhangcy/projects/e2e-moe-20261020-v1'
PY = '/home/zhangcy/.conda/envs/leader/bin/python'
DRIVER = PROJ + '/e2e_online.py'
RES = PROJ + '/results'
SCRATCH = '/home/zhangcy/.dsh_scratch'
GPU = 2
STAGES = ('decode', 'leader', 'refine', 'geometric_rot')


def gpu_used():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used',
                          '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        index, used = [x.strip() for x in line.split(',')]
        if int(index) == GPU:
            return float(used)
    return 0.0


def shard_path(prefix, index):
    return os.path.join(RES, '%s-shard%d.jsonl' % (prefix, index))


def aggregate(name, prefix, procs, frames, extra, wall=None, gpu_gb=None):
    rates, per_stage, rss, torch_peak = [], [], [], []
    rows_total = scored = skipped = 0
    for i in range(procs):
        rows = [json.loads(line) for line in open(shard_path(prefix, i))]
        rows_total += len(rows)
        steady = rows[1:]
        span = steady[-1]['t_end'] - steady[0]['t_end']
        rate = (len(steady) - 1) / span
        rates.append(rate)
        for stage in STAGES:
            per_stage.append(sum(row['seconds'][stage] for row in steady)
                             / len(steady) / (1.0 / rate))
        rss.append((steady[0]['rss_MiB'], steady[-1]['rss_MiB']))
        with open(os.path.join(SCRATCH, '%s-%d.log' % (prefix, i))) as stream:
            for line in stream:
                if line.startswith('[report] '):
                    payload = json.loads(line[9:])
                    torch_peak.append(payload['peak_gpu_reserved_MiB'])
                    scored = payload['scored_against_initial']
                    skipped = payload['skipped_not_in_initial']
    result = dict(name=name, procs=procs, frames_per_proc=frames,
                  frames_total=rows_total,
                  ms_per_frame_aggregate=round(1000.0 / sum(rates), 1),
                  frames_per_second=round(sum(rates), 4),
                  wall_seconds=wall,
                  stage_utilization={stage: round(sum(per_stage[k::len(STAGES)]) / procs, 3)
                                     for k, stage in enumerate(STAGES)},
                  gpu_delta_GB=gpu_gb,
                  torch_reserved_GB=round(sum(torch_peak) / 1024.0, 2),
                  rss_first_last=[rss[0][0], rss[-1][1]],
                  scored=scored, skipped=skipped, extra=list(extra))
    print('[bench] ' + json.dumps(result, sort_keys=True), flush=True)
    return result


def run(name, procs, frames, dates, extra=()):
    prefix = 'bench_' + name
    if os.environ.get('BENCH_REUSE') and all(
            os.path.exists(shard_path(prefix, i))
            and sum(1 for _ in open(shard_path(prefix, i))) >= frames
            for i in range(procs)):
        print('[bench] reusing existing output for %s' % name, flush=True)
        return aggregate(name, prefix, procs, frames, extra)

    for entry in os.listdir(RES):
        if entry.startswith(prefix + '-'):
            os.remove(os.path.join(RES, entry))
    base = gpu_used()
    children, logs = [], []
    begun = time.time()
    for i in range(procs):
        cmd = [PY, DRIVER, '--limit', str(frames), '--shard', '%d/%d' % (i, procs),
               '--dates', dates, '--out-prefix', 'results/' + prefix,
               '--gpu', str(GPU)] + list(extra)
        handle = open(os.path.join(SCRATCH, '%s-%d.log' % (prefix, i)), 'w')
        logs.append(handle)
        children.append(subprocess.Popen(cmd, cwd=PROJ, stdout=handle,
                                         stderr=subprocess.STDOUT))
    peak = 0.0
    while any(child.poll() is None for child in children):
        peak = max(peak, gpu_used())
        time.sleep(3)
    codes = [child.returncode for child in children]
    for handle in logs:
        handle.close()
    wall = round(time.time() - begun, 1)
    if any(codes):
        for i in range(procs):
            with open(os.path.join(SCRATCH, '%s-%d.log' % (prefix, i))) as stream:
                print('---- %s shard %d tail ----' % (name, i))
                print(''.join(stream.readlines()[-25:]))
        raise SystemExit('%s failed, exit codes %r' % (name, codes))
    return aggregate(name, prefix, procs, frames, extra, wall,
                     round((peak - base) / 1024.0, 2))


def main():
    specs = []
    for token in sys.argv[1:]:
        parts = token.split(':')
        extra = parts[3].split(',') if len(parts) > 3 and parts[3] else []
        specs.append((parts[0], int(parts[1]), int(parts[2]), extra))
    dates = os.environ.get('BENCH_DATES', '2012-02-12')
    table = []
    for name, procs, frames, extra in specs:
        print('[bench] %s: %d process(es), %d frames each, dates=%s %s'
              % (name, procs, frames, dates, ' '.join(extra)), flush=True)
        table.append(run(name, procs, frames, dates, extra))
    print()
    print('%-6s %4s %7s %11s %10s %8s %7s %7s' % ('cfg', 'proc', 'fr/proc',
                                                  'ms/frame*', 'frame/s', 'GPU GB',
                                                  'wall s', 'reused'))
    for row in table:
        print('%-6s %4d %7d %11.1f %10.2f %8s %7s %7d'
              % (row['name'], row['procs'], row['frames_per_proc'],
                 row['ms_per_frame_aggregate'], row['frames_per_second'],
                 row['gpu_delta_GB'], row['wall_seconds'], row['scored']))
    print('ms/frame* = 1000 / aggregate steady rate; utilisations: see [bench] json')


if __name__ == '__main__':
    main()
