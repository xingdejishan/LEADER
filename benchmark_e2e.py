import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from benchmark_acceleration import available_ram, query_gpus, stop_owned


ROOT = Path(__file__).resolve().parent
PYTHON = '/home/zhangcy/.conda/envs/leader/bin/python'


def stats(values):
    values = sorted(values)
    if not values:
        return {'n': 0}

    def percentile(p):
        pos = (len(values)-1)*p
        lo = int(pos)
        hi = min(lo+1, len(values)-1)
        return 1000*(values[lo]+(values[hi]-values[lo])*(pos-lo))

    return dict(n=len(values), mean_ms=1000*statistics.mean(values), p50_ms=percentile(.5),
                p95_ms=percentile(.95), max_ms=1000*max(values))


def phase(name, workers, shards, args, directory, keys_file):
    time.sleep(3)
    devices = query_gpus()
    assignments = [args.gpus[i % len(args.gpus)] for i in range(shards)]
    for gpu in set(assignments):
        need = assignments.count(gpu)*(.15*devices[gpu]['total']+512)
        if devices[gpu]['util'] > 10 or need > devices[gpu]['free']*.5:
            raise RuntimeError('GPU %d insufficient idle/free headroom; no workers launched' % gpu)
    if available_ram() < max(64, shards*24)*2**30:
        raise MemoryError('insufficient host RAM headroom')
    processes, logs = [], []
    started = time.monotonic()
    try:
        for shard, gpu in enumerate(assignments):
            log = (directory / ('%s-%d.log' % (name, shard))).open('x')
            logs.append(log)
            command = [PYTHON, '-B', str(ROOT/'e2e_profile_worker.py'),
                       '--keys-file', str(keys_file), '--shard', '%d/%d' % (shard, shards),
                       '--gpu', str(gpu), '--selection-mode', 'gpu_sparse',
                       '--frontend-workers', str(workers), '--gpu-fraction', '.15',
                       '--out', str(directory/('%s-%d.jsonl' % (name, shard)))]
            processes.append(subprocess.Popen(command, cwd=str(ROOT), stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True,
                                               env=dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='1',
                                                        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')))
        while any(p.poll() is None for p in processes):
            if time.monotonic()-started > args.timeout:
                raise TimeoutError(name+' phase timeout')
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(name+' worker failed; inspect logs')
            if available_ram() < 32*2**30:
                raise MemoryError('host RAM below safety floor')
            current = query_gpus()
            if any(current[g]['free'] < 1024 for g in set(assignments)):
                raise MemoryError('GPU free memory below safety floor')
            time.sleep(2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError(name+' worker failed')
    finally:
        stop_owned(processes)
        for log in logs:
            log.close()
    wall = time.monotonic()-started
    rows, steady, initialization = [], [], []
    for shard in range(shards):
        data = [json.loads(s) for s in (directory/('%s-%d.jsonl' % (name, shard))).read_text().splitlines()]
        if len(data) <= 4 or any(r['status'] == 'error' for r in data):
            raise RuntimeError('insufficient or failed profiling frames')
        initialization.append(data[0]['e2e_profile']['initialization_seconds'])
        rows.extend(data)
        steady.extend(data[4:])
    if len(rows) != args.frames or len({r['key'] for r in rows}) != args.frames:
        raise RuntimeError('frame coverage mismatch')
    span = max(r['e2e_profile']['final_T_ready_at'] for r in rows)-min(
        r['e2e_profile']['prepare_started_at'] for r in rows)
    stable_span = max(r['e2e_profile']['final_T_ready_at'] for r in steady)-min(
        r['e2e_profile']['compute_started_at'] for r in steady)
    report = dict(frames=len(rows), steady_frames=len(steady), frontend_workers=workers,
                  shards=shards, gpu_assignments=assignments,
                  wall_seconds_including_startup=wall,
                  model_initialization=stats(initialization),
                  processing_span_seconds=span, aggregate_ms_per_output=1000*span/len(rows),
                  steady_processing_span_seconds=stable_span,
                  steady_aggregate_ms_per_output=1000*stable_span/len(steady),
                  steady_fps=len(steady)/stable_span,
                  steady_input_read_to_batch_output_latency=stats([
                      r['e2e_profile']['input_read_to_batch_T_seconds'] for r in steady]),
                  steady_single_frame_service=stats([r['e2e_profile']['single_frame_service_seconds']
                      for r in steady if r['e2e_profile']['single_frame_service_seconds'] is not None]),
                  steady_selection=stats([r['selection_seconds'] for r in steady]),
                  stage_seconds_mean={k:statistics.mean(r['seconds'][k] for r in steady)
                                      for k in steady[0]['seconds']},
                  status=dict(Counter(r['status'] for r in rows)),
                  peak_torch_allocated_bytes=max(r['e2e_profile']['peak_torch_allocated_bytes'] for r in rows),
                  ground_truth_read=False)
    (directory/(name+'-summary.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps({name:report}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keys-file', required=True)
    parser.add_argument('--gpus', default='3,4,5')
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    args.gpus = [int(g) for g in args.gpus.split(',')]
    keys = json.loads(Path(args.keys_file).read_text())
    args.frames = len(keys)
    if (os.path.realpath(sys.prefix) != os.path.realpath(Path(PYTHON).parent.parent)
            or not 48 <= len(keys) <= 96 or len(set(keys)) != len(keys)
            or len(args.gpus) != 3 or len(set(args.gpus)) != 3):
        parser.error('expected existing environment, 48..96 unique keys, three distinct GPUs')
    directory = ROOT/'results'/('e2e-bench-20261024-'+str(time.time_ns()))
    directory.mkdir(mode=0o700)
    keys_file = directory/'keys.json'
    keys_file.write_text(json.dumps(keys))
    report = dict(scope='input scan + synchronized RGB + cached SAM -> final T; live models',
                  warmup='exclude first four frames per shard; short finite sample',
                  latency='batch T deliverable after CUDA synchronization; includes prefetch queue',
                  single_service='read/preparation + compute, excluding prefetch queue, workers=1 only',
                  wrapper_sha256=hashlib.sha256((ROOT/'e2e_profile_worker.py').read_bytes()).hexdigest())
    (directory/'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(directory=str(directory))), flush=True)
    for name, workers, shards in [('single', 1, 1), ('batch4', 4, 1), ('six_shards', 4, 6)]:
        report[name] = phase(name, workers, shards, args, directory, keys_file)
        (directory/'summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
