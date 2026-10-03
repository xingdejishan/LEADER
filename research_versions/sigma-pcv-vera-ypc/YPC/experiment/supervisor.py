import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

assert sys.prefix == '/home/zhangcy/.conda/envs/leader'
root = Path(__file__).resolve().parent
config = json.loads((root/'config.json').read_text())
store = Path(config['store'])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name, value):
    path = store/name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def identify(pid):
    proc = Path('/proc')/str(pid)
    fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(fields[19]), uid=proc.stat().st_uid)


def users_on_gpu():
    result = []
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    for line in output.splitlines():
        uuid, pid = [v.strip() for v in line.split(',')]
        if uuid == config['gpu_uuid']:
            proc = Path('/proc')/pid
            try:
                result.append(dict(pid=int(pid), uid=proc.stat().st_uid))
            except FileNotFoundError:
                pass
    return result


write('launch.json', dict(identify(os.getpid()), time=time.time(), config_sha256=sha(root/'config.json')))
try:
    for phase in ['checks', 'export', 'train']:
        cpu_phase = phase == 'score'
        if not cpu_phase:
            while True:
                apps = users_on_gpu()
                if not apps:
                    time.sleep(15)
                    if not users_on_gpu():
                        break
                write('controller.json', dict(phase=phase, status='waiting_for_gpu_without_other_users', gpu=config['gpu'], time=time.time()))
                time.sleep(45)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1' if cpu_phase else str(config['gpu']), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            CONDA_PREFIX='/home/zhangcy/.conda/envs/leader', PATH='/home/zhangcy/.conda/envs/leader/bin:/usr/bin:/bin', PYTHONDONTWRITEBYTECODE='1')
        with (store/(phase+'.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', str(root/'entry.py'), '--phase', phase], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = identify(child.pid)
            write('worker.json', dict(identity, phase=phase, time=time.time(), gpu=None if cpu_phase else config['gpu']))
            peak = 0
            started = time.monotonic()
            while child.poll() is None:
                proc = Path('/proc')/str(child.pid)
                try:
                    fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
                    if fields[0] == 'Z':
                        break
                    assert fields[19] == str(identity['start_ticks']) and proc.stat().st_uid == os.getuid()
                    assert (proc/'exe').resolve(strict=True) == Path(sys.executable).resolve()
                    rss = next((int(line.split()[1])*1024 for line in (proc/'status').read_text().splitlines() if line.startswith('VmRSS:')), 0)
                    peak = max(peak, rss)
                    available = int(next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')).split()[1])*1024
                    reason = None
                    if rss > config['limits']['rss_gib']*1024**3 or available < config['limits']['system_free_gib']*1024**3 or shutil.disk_usage(store).free < config['limits']['disk_free_gib']*1024**3:
                        reason = 'resource_floor'
                    if not cpu_phase and any(v['pid'] != child.pid for v in users_on_gpu()):
                        reason = 'another_user_entered_gpu'
                    if time.monotonic()-started > config['limits']['wall_seconds']:
                        reason = 'phase_wall_budget'
                    if reason:
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait(timeout=20)
                        raise RuntimeError(reason)
                except FileNotFoundError:
                    pass
                write('controller.json', dict(phase=phase, status='running', rss_peak_bytes=peak, gpu=None if cpu_phase else config['gpu'], time=time.time()))
                time.sleep(2)
            code = child.wait()
        write(phase+'-exit.json', dict(code=code, rss_peak_bytes=peak, time=time.time()))
        if code:
            raise RuntimeError(phase+' exited '+str(code))
    write('pipeline-complete.json', dict(time=time.time(), phase_gate_sha256=sha(store/'phase-gate.json')))
except Exception as error:
    write('failed.json', dict(error=repr(error), phase=phase, time=time.time()))
    raise

