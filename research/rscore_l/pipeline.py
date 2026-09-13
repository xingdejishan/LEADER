import json
import subprocess
import sys
import time
from pathlib import Path

from .prepare import save_json


def run_all(args):
    root = args.root
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    run = Path(__file__).with_name('run.py')
    stages = [('prepare', ['prepare']), ('features', ['features']), ('geometry', ['geometry']), ('overlap', ['overlap']),
        ('node2vec-pose', ['node2vec', '--graph', 'pose']), ('node2vec-lidar', ['node2vec', '--graph', 'lidar'])]
    for split in ('train', 'val', 'test'):
        stages.append(('retrieval-' + split, ['retrieval', '--split', split]))
    stages.append(('freeze', ['freeze']))
    for variant in ('depth', 'geometry', 'lidar', 'scrfacto'):
        stages.append(('train-' + variant, ['train', '--variant', variant]))
        for split in ('val', 'test'):
            stages.append(('export-' + variant + '-' + split, ['export', '--variant', variant, '--split', split]))
            stages.append(('evaluate-' + variant + '-' + split, ['evaluate', '--variant', variant, '--split', split]))
        stages.append(('benchmark-' + variant, ['benchmark', '--variant', variant]))
    for name, command in stages:
        stage(root, logs, run, args, name, command)
    rows = json.loads((root / 'data/manifest.json').read_text())['train']
    for session in sorted(set(row['session_id'] for row in rows)):
        stage(root, logs, run, args, 'fold-' + session, ['train', '--variant', 'lidar-fold-' + session, '--exclude-session', session])
    stage(root, logs, run, args, 'reliability', ['reliability'])
    for split in ('val', 'test'):
        stage(root, logs, run, args, 'export-reliable-' + split, ['export', '--variant', 'reliable', '--split', split])
        stage(root, logs, run, args, 'evaluate-reliable-' + split, ['evaluate', '--variant', 'reliable', '--split', split])
        stage(root, logs, run, args, 'evaluate-glace-' + split, ['evaluate', '--variant', 'glace', '--split', split])
    stage(root, logs, run, args, 'benchmark-reliable', ['benchmark', '--variant', 'reliable'])
    reports = {variant: {split: json.loads((root / 'evaluation' / variant / split / 'summary.json').read_text()) for split in ('val', 'test')}
        for variant in ('glace', 'scrfacto', 'depth', 'geometry', 'lidar', 'reliable')}
    save_json(root / 'comparison.json', reports)
    save_json(root / 'state.json', dict(status='COMPLETE', finished=time.time(), comparison=str(root / 'comparison.json')))


def stage(root, logs, run, args, name, command):
    marker = logs / (name + '.done.json')
    full = [sys.executable, str(run), *command, '--root', str(root), '--bundle', str(args.bundle), '--iterations', str(args.iterations)]
    if marker.exists():
        if json.loads(marker.read_text())['command'] != full:
            raise RuntimeError('Saved stage command differs: ' + name)
        return
    state = dict(status='RUNNING', stage=name, command=full, started=time.time())
    save_json(root / 'state.json', state)
    with (logs / (name + '.log')).open('w') as log:
        process = subprocess.run(full, stdout=log, stderr=subprocess.STDOUT)
    state.update(returncode=process.returncode, finished=time.time(), status='DONE' if process.returncode == 0 else 'FAILED')
    save_json(root / 'state.json', state)
    if process.returncode:
        raise RuntimeError('Stage failed: ' + name)
    save_json(marker, state)
