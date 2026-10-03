import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def write(path,value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def own_group_stop(child):
    if child.poll() is None:
        assert Path('/proc') .joinpath(str(child.pid)).stat().st_uid == os.getuid()
        assert os.getpgid(child.pid) == child.pid
        os.killpg(child.pid,signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL)
            child.wait()


def check_gpu():
    gpu = subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert int(gpu[1]) >= 14336
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    for line in apps.splitlines():
        uuid,pid = [x.strip() for x in line.split(',')]
        if uuid==gpu[0].strip():
            p = Path('/proc')/pid
            if p.exists():
                assert p.stat().st_uid==os.getuid(), 'another user owns a GPU1 process'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--smoke-only',action='store_true')
    args = parser.parse_args()
    assert sys.prefix=='/home/zhangcy/.conda/envs/leader'
    config = json.loads(Path(args.config).read_text())
    store = Path(config['store'])
    phases = ['smoke'] if args.smoke_only else ['stage2','predict','score']
    if not args.smoke_only:
        assert json.loads((store/'smoke_passed.json').read_text())['passed']
    for phase in phases:
        if phase!='score':
            check_gpu()
        command = [sys.executable,'-B','-u',str(Path(__file__).parent/'run_repair_v3.py'),'--config',args.config]
        if phase!='score':
            command += ['--phase',phase]
        with (store/(phase+'.log')).open('x') as log:
            child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1' if phase=='score' else '1'))
            write(store/'worker.json',dict(phase=phase,pid=child.pid,start_ticks=Path('/proc/{}/stat'.format(child.pid)).read_text().rsplit(')',1)[1].split()[19]))
            worker_start=Path('/proc/{}/stat'.format(child.pid)).read_text().rsplit(')',1)[1].split()[19]
            peak_rss = 0
            try:
                while child.poll() is None:
                    proc = Path('/proc')/str(child.pid)
                    if proc.exists():
                        assert proc.stat().st_uid==os.getuid()
                        assert (proc/'stat').read_text().rsplit(')',1)[1].split()[19]==worker_start
                    try:
                        rss = int(next(line for line in (proc/'status').read_text().splitlines() if line.startswith('VmRSS:')).split()[1])*1024
                        peak_rss = max(peak_rss,rss)
                    except (FileNotFoundError,StopIteration):
                        rss = 0
                    available = int(next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')).split()[1])*1024
                    if rss>10*1024**3 or available<32*1024**3 or shutil.disk_usage(store).free<40*1024**3:
                        raise RuntimeError('resource floor reached')
                    time.sleep(1)
            except BaseException:
                own_group_stop(child)
                raise
            result = dict(phase=phase,exit_code=child.returncode,peak_rss_bytes=peak_rss,time=time.time())
            write(store/(phase+'-exit.json'),result)
            if child.returncode != 0:
                write(store/'failed.json',result)
                raise RuntimeError('phase failed: '+phase)
    write(store/('smoke-pipeline-complete.json' if args.smoke_only else 'completion.json'),dict(phases=phases,time=time.time()))


if __name__=='__main__':
    main()
