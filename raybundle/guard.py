import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time


def snapshot(gpu):
    raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
    uuid,memory,utilization=[v.strip() for v in raw.strip().split(',')]
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
    pids=[int(line.rsplit(',',1)[1]) for line in raw.splitlines() if line.split(',')[0].strip()==uuid]
    return dict(time=time.time(),uuid=uuid,memory_mib=int(memory),utilization=int(utilization),pids=pids)


def rss(pid):
    path=Path('/proc')/str(pid)/'status'
    if not path.exists():
        return 0
    values=dict(line.split(':',1) for line in path.read_text().splitlines() if ':' in line)
    return int(values.get('VmRSS','0 kB').split()[0])*1024


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    root=Path('/home/zhangcy/projects/leader-ray-bundle-20260928').resolve()
    run,source=Path(args.run).resolve(),Path(args.source).resolve()
    if root not in run.parents or root not in source.parents or root.stat().st_uid!=os.getuid() or sys.prefix!='/home/zhangcy/.conda/envs/leader':
        raise ValueError('invalid environment or private paths')
    if run.exists():
        raise FileExistsError(run)
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or command[0]!=sys.executable:
        raise ValueError('child must use the existing environment')
    checks=[]
    for index in range(2):
        state=snapshot(args.gpu)
        if state['memory_mib'] or state['utilization'] or state['pids']:
            raise RuntimeError('GPU is not fully idle: '+json.dumps(state))
        checks.append(state)
        if index==0:
            time.sleep(30)
    if shutil.disk_usage(root).free < 7*2**30:
        raise RuntimeError('insufficient startup disk')
    run.mkdir(mode=0o700)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (run/'console.log').open('x') as log:
        child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch=dict(child_pid=child.pid,guard_pid=os.getpid(),gpu=args.gpu,checks=checks,command=command,source=str(source))
    (run/'launch.json').write_text(json.dumps(launch,indent=2))
    failure=None
    try:
        while child.poll() is None:
            state=snapshot(args.gpu)
            foreign=[]
            for pid in state['pids']:
                try:
                    if os.getpgid(pid)!=child.pid:
                        foreign.append(pid)
                except ProcessLookupError:
                    pass
            if foreign:
                raise RuntimeError('foreign GPU process appeared: '+str(foreign))
            memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            if rss(child.pid)>12*2**30 or int(memory['MemAvailable'].split()[0])*1024<32*2**30 or shutil.disk_usage(root).free<3*2**30:
                raise RuntimeError('memory or disk guard triggered')
            state.update(rss_bytes=rss(child.pid),free_disk_bytes=shutil.disk_usage(root).free)
            temporary=run/'guard.tmp'
            temporary.write_text(json.dumps(state))
            os.replace(temporary,run/'guard.json')
            time.sleep(5)
    except BaseException as error:
        failure=str(error)
        if child.poll() is None:
            if Path('/proc',str(child.pid)).stat().st_uid!=os.getuid() or os.getpgid(child.pid)!=child.pid:
                raise RuntimeError('child identity changed; refusing signal')
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait(timeout=10)
    (run/'exit.json').write_text(json.dumps(dict(code=child.returncode,failure=failure,time=time.time()),indent=2))


if __name__=='__main__':
    main()
