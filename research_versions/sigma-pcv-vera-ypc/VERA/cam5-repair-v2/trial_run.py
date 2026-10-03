import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
from trial_common import PLAN, ROOT, STORE, atomic, guard, output_bytes, sha, verify_source

verify_source()
os.nice(15)
lock = (ROOT / 'storage/execution.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
launch = json.loads((STORE / 'launch.json').read_text())
assert launch['pid'] == os.getpid()


def choose_gpu():
    candidates = []
    for row in subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines():
        gpu, free, utilization = [float(value.strip()) for value in row.split(',')]
        if int(gpu) in PLAN['allowed_gpus'] and free >= PLAN['gpu_initial_free_floor_gib'] * 1024 and utilization <= PLAN['gpu_initial_utilization_ceiling']:
            candidates.append((utilization, -free, int(gpu)))
    return min(candidates)[2] if candidates else None


def gpu_guard(gpu, pid):
    rows = {int(row.split(',')[0]): float(row.split(',')[1]) for row in subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()}
    assert gpu in PLAN['allowed_gpus'] and rows[gpu] >= PLAN['gpu_free_floor_gib'] * 1024, 'shared GPU free-memory floor'
    for row in subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True).splitlines():
        values = row.split(',')
        if int(values[0]) == pid:
            assert float(values[1]) <= PLAN['gpu_process_ceiling_gib'] * 1024, 'private GPU memory ceiling'


stages = [('probe', ['trial_probe.py'], True)]
for epoch in PLAN['evaluate_epochs']:
    stages.extend([('train-%02d' % epoch, ['trial_train.py', str(epoch)], True), ('predict-%02d' % epoch, ['trial_predict.py', str(epoch)], True), ('score-%02d' % epoch, ['trial_score.py', str(epoch)], False)])
history = json.loads((STORE / 'stages.json').read_text()) if (STORE / 'stages.json').exists() else []
for stage, arguments, is_gpu in stages:
    if any(row['stage'] == stage and row['exit'] == 0 and not row.get('guard_stop') for row in history):
        continue
    attempt = 0
    while True:
        try:
            guard()
            assert output_bytes() < PLAN['output_limit_gib'] * 1024 ** 3, 'private output ceiling'
            safe = True
        except AssertionError:
            safe = False
        gpu = choose_gpu() if is_gpu and safe else None
        if not safe or (is_gpu and gpu is None):
            atomic(STORE / 'progress.json', {'state': 'waiting_safe_resources', 'stage': stage, 'time': time.time()})
            time.sleep(30)
            continue
        attempt += 1
        log = STORE / (stage + '-attempt-%d.log' % time.time_ns())
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu) if is_gpu else '-1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with log.open('x') as stream:
            child = subprocess.Popen([sys.executable, '-B', '-u', str(ROOT / arguments[0]), *arguments[1:]], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            process = psutil.Process(child.pid)
            start = process.create_time()
            record = {'stage': stage, 'pid': child.pid, 'process_create_time': start, 'uid': os.getuid(), 'gpu': gpu, 'log': str(log), 'time': time.time(), 'attempt': attempt}
            atomic(STORE / 'current-stage.json', record)
            reason, last_gpu, last_disk = None, 0., 0.
            while child.poll() is None:
                try:
                    guard()
                    rss = sum(member.memory_info().rss for member in [process] + process.children(recursive=True) if member.is_running())
                    assert rss < PLAN['rss_limit_gib'] * 1024 ** 3, 'worker RSS ceiling'
                    if is_gpu and time.time() - last_gpu > 5:
                        gpu_guard(gpu, child.pid)
                        last_gpu = time.time()
                    if time.time() - last_disk > 20:
                        assert output_bytes() < PLAN['output_limit_gib'] * 1024 ** 3, 'private output ceiling'
                        last_disk = time.time()
                except (psutil.NoSuchProcess, FileNotFoundError):
                    if child.poll() is not None:
                        break
                except (AssertionError, RuntimeError) as error:
                    reason = str(error)
                    if process.is_running() and process.create_time() == start and process.uids().real == 1003:
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                    break
                time.sleep(1)
            code = child.wait()
        record.update(exit=code, end=time.time(), guard_stop=reason)
        history.append(record)
        atomic(STORE / 'stages.json', history)
        if code == 0 and reason is None:
            break
        tail = log.read_text()[-4000:]
        recoverable = bool(reason) or 'out of memory' in tail.lower()
        if not recoverable or attempt >= 3:
            atomic(STORE / 'needs-attention.json', {'stage': stage, 'exit': code, 'guard_stop': reason, 'log': str(log), 'time': time.time(), 'checkpoint_preserved': True})
            atomic(STORE / 'pipeline-exit.json', {'exit': 1, 'stage': stage, 'time': time.time()})
            sys.exit(1)
        atomic(STORE / 'progress.json', {'state': 'resource_retry_from_checkpoint', 'stage': stage, 'attempt': attempt, 'time': time.time()})
        time.sleep(60)
reports = [json.loads((STORE / 'eval' / ('e%02d' % epoch) / 'report.json').read_text()) for epoch in PLAN['evaluate_epochs']]
best = min(reports, key=lambda row: (row['results']['vera']['failures'], row['results']['vera']['mpe_m'] / .17949788079580942 + row['results']['vera']['moe_deg'] / 1.6242310898570118, row['epoch']))
atomic(STORE / 'comparison.json', {'selected_epoch': best['epoch'], 'selection': 'reused development set, not independent test', 'reports': reports, 'requires_research_review': True, 'remaining': 'Review actual visual choice, candidate quality calibration and final pose evidence; no automatic further training'})
atomic(STORE / 'pipeline-exit.json', {'exit': 0, 'comparison_sha256': sha(STORE / 'comparison.json'), 'time': time.time()})
