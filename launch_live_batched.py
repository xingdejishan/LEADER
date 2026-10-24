import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmark_acceleration import available_ram, query_gpus, stop_owned


ROOT = Path(__file__).resolve().parent
PYTHON = '/home/zhangcy/.conda/envs/leader/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keys-file', required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--frames', type=int, choices=(96, 384), default=96)
    parser.add_argument('--shards', type=int, choices=(3, 4), default=3)
    args = parser.parse_args()
    if os.path.realpath(sys.prefix) != os.path.realpath(Path(PYTHON).parent.parent):
        raise RuntimeError('existing environment required')
    gpu = query_gpus()[args.gpu]
    estimated_memory = args.shards * (.15 * gpu['total'] + 768)
    if (gpu['util'] > 5 or gpu['free'] - estimated_memory < 4096
            or available_ram() < 192 * 2**30):
        raise RuntimeError('insufficient idle resources for bounded shard trial')
    existing = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if existing:
        raise RuntimeError('selected GPU already has compute processes')
    parent = Path('/data2/zhangcy').resolve(strict=True)
    if parent.stat().st_uid != os.getuid():
        raise PermissionError('scratch parent must be owned')
    stamp = str(time.time_ns())
    directory = parent / ('e2e-live-batch-launch-' + stamp)
    directory.mkdir(mode=0o700)
    (directory / 'results').mkdir(mode=0o700)
    gate = directory / 'results' / 'go.json'
    children = [parent / ('e2e-live-batch-shard-%s-%d' % (stamp, i)) for i in range(args.shards)]
    processes, logs = [], []
    started = time.monotonic()
    print(str(directory), flush=True)
    try:
        for index, child in enumerate(children):
            log = (directory / 'results' / ('shard-%d.log' % index)).open('x')
            logs.append(log)
            processes.append(subprocess.Popen([PYTHON, '-B', str(ROOT / 'benchmark_live_batched.py'),
                '--keys-file', args.keys_file, '--scratch-root', str(child), '--gpu', str(args.gpu),
                '--frames', str(args.frames), '--mode', 'batched', '--shards', str(args.shards), '--shard', str(index),
                '--gate-file', str(gate)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')))
        while not all((child / 'results' / 'ready.json').exists() for child in children):
            monitor(processes, args.gpu, started, directory)
            time.sleep(1)
        gate.write_text(json.dumps(dict(start=time.monotonic())))
        while any(process.poll() is None for process in processes):
            monitor(processes, args.gpu, started, directory)
            time.sleep(1)
        if any(process.returncode != 0 for process in processes):
            raise RuntimeError('bounded worker failed; inspect logs')
    finally:
        stop_owned(processes)
        for log in logs:
            log.close()
    reports = [json.loads((child / 'results' / 'report.json').read_text()) for child in children]
    phases = [report['reports']['batched'] for report in reports]
    rows = [row for report in reports for row in report['rows']['batched']]
    if len(rows) != args.frames or len({row['key'] for row in rows}) != args.frames:
        raise RuntimeError('incomplete or duplicate output coverage')
    span = max(phase['finished_at'] for phase in phases) - min(phase['started_at'] for phase in phases)
    from benchmark_live_batched import statistics

    summary = dict(gpu=args.gpu, shards=args.shards, batch_size=4, frames=args.frames,
        scope=reports[0]['scope'], processing_span_seconds=span,
        aggregate_ms_per_frame=span * 1000 / len(rows),
        input_to_T=statistics([row['latency_seconds'] for row in rows]),
        GT_attempts=sum(report['GT_attempts'] for report in reports),
        peak_torch_bytes_per_worker=max(report['peak_torch_bytes'] for report in reports),
        initialization_seconds=max(report['initialization_seconds'] for report in reports),
        children=[str(child) for child in children],
        accuracy_status='batching changes FP32 results; not promoted; no GT-based accuracy evaluation')
    (directory / 'results' / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def monitor(processes, gpu, started, directory):
    if time.monotonic() - started > 600:
        raise TimeoutError('finite benchmark timeout')
    if any(process.poll() not in (None, 0) for process in processes):
        raise RuntimeError('worker failed; inspect isolated logs: ' + str(directory))
    if query_gpus()[gpu]['free'] < 4096 or available_ram() < 32 * 2**30:
        raise MemoryError('benchmark global resource safety floor reached')
    stat = os.statvfs(str(directory))
    if stat.f_bavail * stat.f_frsize < 2**30:
        raise OSError('scratch filesystem safety floor reached')


if __name__ == '__main__':
    main()
