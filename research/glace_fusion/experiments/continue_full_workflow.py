import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path('/root/rivermind-data/glace_nclt_corrected_20260912')
repo = Path('/root/rivermind-data/LEADER-v1-glace-independent')
camera = Path('/root/nclt_full_test_camera')
dates = ['2012-02-12', '2012-02-19', '2012-03-31', '2012-05-26']
deit = '/root/rivermind-data/LEADER-v1-visual-glace/research/visual_glace/CVPR23_DeitS_Rerank.pth'


def state(stage, **fields):
    record = {'stage': stage, 'updated_at': time.time(), **fields}
    (root / 'workflow.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


def execute(stage, arguments):
    state(stage)
    with (root / (stage + '.log')).open('w') as log:
        subprocess.run([sys.executable] + arguments, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)


try:
    while not (root / 'training_finished.json').exists():
        state('waiting_for_training')
        time.sleep(60)
    if json.loads((root / 'training_finished.json').read_text())['returncode'] != 0:
        raise RuntimeError('Training failed; evaluation is blocked')
    quality = root / 'quality_report.json'
    execute('quality_check', ['-m', 'research.glace_fusion.validate_training_head', '--run-root', str(root),
                              '--head', str(root / 'glace_head.pt'), '--vendor', str(root / 'vendor_corrected'),
                              '--deit-checkpoint', deit, '--output', str(quality)])
    if not json.loads(quality.read_text())['ready_for_test']:
        raise RuntimeError('Training localization quality gate failed; evaluation is blocked')
    while not all((camera / date / '.complete').exists() for date in dates):
        failures = [date for date in dates if (camera / date / 'failure.json').exists()]
        if failures:
            raise RuntimeError('Camera download failed: ' + ', '.join(failures))
        state('waiting_for_test_images', complete_dates=[date for date in dates if (camera / date / '.complete').exists()])
        time.sleep(60)
    rows = []
    for date in dates:
        with (camera / date / 'image_index.csv').open() as handle:
            rows.extend(csv.DictReader(handle))
    with (camera / 'all_images.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (camera / 'calibration').mkdir(exist_ok=True)
    shutil.copyfile('/root/rivermind-data/datasets/NCLT_camera_v1/calibration/processing_metadata.json',
                    camera / 'calibration/processing_metadata.json')
    execute('full_test', ['-m', 'research.glace_fusion.full_comparison', '--head', str(root / 'glace_head.pt'),
                          '--quality-report', str(quality), '--vendor', str(root / 'vendor_corrected'),
                          '--deit-checkpoint', deit, '--camera-root', str(camera), '--out', str(root / 'full_test')])
    report = json.loads((root / 'full_test/report.json').read_text())
    if not report['complete']:
        raise RuntimeError('Evaluation did not cover all requested scans')
    state('complete', report=str(root / 'full_test/report.json'), frames=report['frames'])
except Exception as exc:
    state('needs_attention', error=repr(exc))
    raise
