import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--registry',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    output.mkdir()
    tasks=json.loads(Path(args.registry).read_text())['tasks']
    disabled=set()
    previous=None
    worker=Path(__file__).with_name('validate_job.py')
    launcher=Path(__file__).with_name('validation_subprocess.py')
    save(output/'ready.json',dict(tasks=[t['name'] for t in tasks],time=time.time(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    while True:
        choices=[]
        completed=[]
        for task in tasks:
            run=Path(task['output'])
            if task['name'] in disabled:
                completed.append(task['name'])
                continue
            pending=[p for p in sorted((run/'queue').glob('*.json')) if not (run/'results'/p.name).exists()]
            if pending:
                choices.append((task,pending[0]))
            elif (run/'training_done.json').exists():
                if not (run/'completion.json').exists():
                    save(run/'completion.json',dict(status='training-and-validation-completed',best=json.loads((run/'selection.json').read_text())))
                completed.append(task['name'])
            elif (run/'training_worker.json').exists():
                descriptor=json.loads((run/'training_worker.json').read_text())
                end=Path(descriptor['guard_run'])/'exit.json'
                if end.exists():
                    status=json.loads(end.read_text())
                    if status['code']!=0 or status['failure']:
                        save(run/'validation_idle_after_training_failure.json',status)
                        disabled.add(task['name'])
                        completed.append(task['name'])
        if len(completed)==len(tasks):
            save(output/'completion.json',dict(tasks=completed,time=time.time()))
            return
        if not choices:
            save(output/'progress.json',dict(phase='waiting-for-checkpoints',time=time.time()))
            time.sleep(5)
            continue
        choices.sort(key=lambda pair:(pair[0]['name']==previous,pair[1].stat().st_mtime))
        task,job=choices[0]
        task_file=output/(task['name']+'.json')
        save(task_file,task)
        save(output/'progress.json',dict(phase='validating',variant=task['name'],job=str(job),time=time.time()))
        command=[sys.executable,'-B','-u',str(launcher),'--source',task['source'],'--worker',str(worker),'--task',str(task_file),'--job',str(job)]
        result=subprocess.run(command,cwd=task['source'])
        previous=task['name']
        if result.returncode:
            save(Path(task['output'])/'validation_failed.json',dict(code=result.returncode,job=str(job),time=time.time()))
            disabled.add(task['name'])


if __name__=='__main__':
    main()
