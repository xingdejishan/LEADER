import argparse
import datetime
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/zhangcy/.conda/envs/leader/bin/python'


def query_gpus():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    return {int(row[0]): dict(total=int(row[1]), free=int(row[2]), util=int(row[3]))
            for row in (line.split(',') for line in text.strip().splitlines())}


def available_ram():
    return int(next(l for l in Path('/proc/meminfo').read_text().splitlines()
                    if l.startswith('MemAvailable:')).split()[1]) * 1024


def stop_owned(processes):
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    for process in processes:
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_phase(name, args, directory, keys_file):
    gpus = query_gpus()
    assignments = [args.gpus[i % len(args.gpus)] for i in range(args.shards)]
    for gpu in set(assignments):
        count = assignments.count(gpu)
        need = count * (0.15 * gpus[gpu]['total'] + 512)
        if need > gpus[gpu]['free'] * 0.5 or gpus[gpu]['util'] > 10:
            raise RuntimeError('GPU %s is not sufficiently idle/free for bounded shards' % gpu)
    if available_ram() < max(64, args.shards * 24) * 2 ** 30:
        raise RuntimeError('insufficient host RAM safety headroom')
    processes, logs = [], []
    started = time.monotonic()
    try:
        for shard, gpu in enumerate(assignments):
            output = directory / ('%s-s%d.jsonl' % (name, shard))
            log = (directory / ('%s-s%d.log' % (name, shard))).open('x')
            logs.append(log)
            command = [PYTHON, '-B', str(ROOT / 'e2e_online.py'), '--keys-file', str(keys_file),
                       '--shard', '%s/%s' % (shard, args.shards), '--gpu', str(gpu),
                       '--out', str(output), '--frontend-workers', '4' if name == 'accelerated' else '1',
                       '--fast-pnp', '1' if name == 'accelerated' else '0',
                       '--verify-cache', '1' if name == 'accelerated' else '0']
            processes.append(subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True,
                                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                                                        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                                                        MKL_NUM_THREADS='1')))
        while any(p.poll() is None for p in processes):
            if time.monotonic() - started > args.timeout:
                raise TimeoutError(name + ' benchmark timeout')
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(name + ' shard failed; inspect phase logs')
            if available_ram() < 32 * 2 ** 30:
                raise MemoryError('host available RAM below safety floor')
            current_gpus = query_gpus()
            if any(current_gpus[g]['free'] < 1024 for g in set(assignments)):
                raise MemoryError('GPU free memory below 1 GiB safety floor')
            time.sleep(2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError(name + ' shard failed')
    finally:
        stop_owned(processes)
        for log in logs:
            log.close()
    wall = time.monotonic() - started
    rows = []
    for shard in range(args.shards):
        values = [json.loads(line) for line in
                  (directory / ('%s-s%d.jsonl' % (name, shard))).read_text().splitlines()]
        if any(r['status'] == 'error' for r in values):
            raise RuntimeError('error rows in benchmark')
        rows.extend(values)
    if len(rows) != args.frames or len({r['key'] for r in rows}) != len(rows):
        raise RuntimeError('benchmark frame coverage mismatch')
    stage = {key: statistics.mean(r['seconds'][key] for r in rows) for key in rows[0]['seconds']}
    span = max(r['batch_finished_at'] for r in rows) - min(r['batch_started_at'] for r in rows)
    detail = [r['visual']['detail'] for r in rows if r.get('visual') and r['visual'].get('detail')]
    report = dict(frames=len(rows), shards=args.shards, gpu_assignments=assignments,
                  wall_seconds_including_startup=wall,
                  processing_span_seconds=span, processing_ms_per_output=1000 * span / len(rows),
                  frames_per_second=len(rows) / span, stage_seconds_mean=stage,
                  frame_work_seconds_mean=statistics.mean(sum(r['seconds'].values()) for r in rows),
                  visual_stage_seconds_mean={key: statistics.mean(d['seconds'][key] for d in detail)
                                             for key in detail[0]['seconds']} if detail else {},
                  batched_matching_frames=sum(bool(d['batched']) for d in detail),
                  visual_selected=sum(r['arbitration']['chosen'] == 'visual' for r in rows))
    print(json.dumps({name: report}), flush=True)
    return report, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=48)
    parser.add_argument('--shards', type=int, default=6)
    parser.add_argument('--gpus', default='0,1,3')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--only', choices=('both', 'baseline', 'accelerated'), default='both')
    args = parser.parse_args()
    args.gpus = [int(g) for g in args.gpus.split(',')]
    if (os.path.realpath(sys.prefix) != os.path.realpath(Path(PYTHON).parent.parent)
            or not 1 <= args.shards <= 6 or not 4 * args.shards <= args.frames <= 96
            or not args.gpus or len(set(args.gpus)) != len(args.gpus)):
        parser.error('invalid environment or finite benchmark bounds')
    directory = ROOT / 'results' / ('acceleration-bench-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    directory = directory.resolve()
    if ROOT / 'results' not in directory.parents:
        raise RuntimeError('benchmark output escapes project')
    directory.mkdir(mode=0o700)
    keys = [r['key'] for r in json.loads((ROOT / 'inputs/frames.json').read_text())[:args.frames]]
    keys_file = directory / 'keys.json'
    keys_file.write_text(json.dumps(keys))
    print(json.dumps(dict(directory=str(directory), frames=args.frames)), flush=True)
    summary, outputs = {}, {}
    phases = ('baseline', 'accelerated') if args.only == 'both' else (args.only,)
    for phase in phases:
        summary[phase], outputs[phase] = run_phase(phase, args, directory, keys_file)
    if len(phases) == 2:
        import numpy as np

        before = {r['key']: r for r in outputs['baseline']}
        distances, angles = [], []
        for row in outputs['accelerated']:
            old = before[row['key']]
            a, b = np.asarray(old['T']), np.asarray(row['T'])
            distances.append(float(np.linalg.norm(a[:3, 3] - b[:3, 3])))
            cos = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)
            angles.append(float(np.degrees(np.arccos(cos))))
        summary['comparison'] = dict(speedup=summary['baseline']['processing_span_seconds'] /
                                    summary['accelerated']['processing_span_seconds'],
                                    max_pose_shift_m=max(distances), mean_pose_shift_m=statistics.mean(distances),
                                    max_rotation_difference_deg=max(angles),
                                    changed_choice=sum(before[r['key']]['arbitration']['chosen'] !=
                                                       r['arbitration']['chosen'] for r in outputs['accelerated']),
                                    accuracy_evaluated=False)
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(summary=str(directory / 'summary.json'), results=summary)), flush=True)


if __name__ == '__main__':
    main()
