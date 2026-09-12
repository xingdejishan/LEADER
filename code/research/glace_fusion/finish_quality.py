import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--quality-target', choices=['reprojection', 'joint3d'], default='reprojection')
    args = parser.parse_args()
    root = args.root
    deadline = time.time() + 1800
    while time.time() < deadline:
        state = json.loads((root / 'state.json').read_text())
        if state['stage'] == 'complete':
            break
        if state['stage'] == 'failed':
            raise RuntimeError(state)
        time.sleep(10)
    else:
        raise TimeoutError('Training and evaluation did not complete')
    selected = json.loads((root / 'selection.json').read_text())['arm']
    commands = [
        ('confidence', ['correspondence_confidence', '--validation', str(root / selected / 'validation'),
            '--test', str(root / 'test'), '--out', str(root / 'confidence'), '--quality-target', args.quality_target]),
        ('raw_audit', ['correspondence_audit', '--coordinates', str(root / 'test/coordinates'),
            '--out', str(root / 'raw_audit.json')]),
        ('filtered_audit', ['correspondence_audit', '--coordinates', str(root / 'confidence/coordinates'),
            '--out', str(root / 'filtered_audit.json')]),
        ('raw_joint', ['cached_joint_eval', '--coordinates', str(root / 'test/coordinates'),
            '--out', str(root / 'raw_joint')]),
        ('filtered_joint', ['cached_joint_eval', '--coordinates', str(root / 'confidence/coordinates'),
            '--out', str(root / 'filtered_joint')]),
    ]
    try:
        for stage, arguments in commands:
            (root / 'finish_state.json').write_text(json.dumps(dict(stage=stage, time=time.time())))
            with (root / (stage + '.log')).open('w') as output:
                subprocess.run([sys.executable, '-m', 'research.glace_fusion.' + arguments[0]] + arguments[1:],
                    env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='2'),
                    stdout=output, stderr=subprocess.STDOUT, check=True)
        (root / 'finish_state.json').write_text(json.dumps(dict(stage='complete', time=time.time())))
    except Exception:
        import traceback
        (root / 'finish_state.json').write_text(json.dumps(dict(stage='failed', error=traceback.format_exc())))
        raise


if __name__ == '__main__':
    main()
