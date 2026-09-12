import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .retrain_rgb_baseline import digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--stage1', type=Path, required=True)
    parser.add_argument('--test-scene', type=Path, required=True)
    parser.add_argument('--dataset-folder', type=Path, required=True)
    parser.add_argument('--leader-checkpoint', type=Path, required=True)
    parser.add_argument('--deit-checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    try:
        exp = json.loads((args.experiment / 'experiment.json').read_text())
        local_meta = json.loads((args.experiment / 'scene/scene_meta.json').read_text())
        test_meta = json.loads((args.test_scene / 'scene_meta.json').read_text())
        old_stems = {r['image'] for r in local_meta['splits']['test']['pairs']}
        anchor = np.loadtxt(args.stage1 / 'scene/train/poses' / (exp['anchor_image'] + '.txt'))
        rows = []
        for row in sorted(test_meta['splits']['test']['pairs'], key=lambda r: (r['sequence'], r['image_timestamp_us'])):
            gt = np.loadtxt(args.test_scene / 'test/poses' / (row['image'] + '.txt'))
            distance = float(np.linalg.norm(gt[:3, 3] - anchor[:3, 3]))
            angle = float(np.degrees(np.arccos(np.clip((np.sum(gt[:3, :3] * anchor[:3, :3]) - 1) / 2, -1, 1))))
            if distance <= exp['radius_m']:
                rows.append(dict(row, distance_from_anchor_m=distance, angle_from_anchor_deg=angle,
                    in_orientation_support=angle < 30, previous_probe=row['image'] in old_stems))
        if not rows or not old_stems <= {r['image'] for r in rows}:
            raise ValueError('The expanded frame set does not cover the earlier probe')
        write_json(args.out / 'rows.json', rows)
        write_json(args.out / 'stems.json', [r['image'] for r in rows])
        heads = dict(stage1=args.stage1 / 'stage1_80k/head.pt', improved=args.experiment / 'lidar_auxiliary/head.pt')
        vendors = dict(stage1=args.stage1 / 'vendor', improved=args.experiment / 'lidar_auxiliary/vendor')
        plan = dict(created=time.time(), scope='all available cached test images within the frozen local training region',
            full_nclt=False, available_dates=sorted({r['sequence'] for r in test_meta['splits']['test']['pairs']}),
            declared_split_dates=test_meta['splits']['test']['dates'],
            missing_dates=test_meta.get('missing_sequences', []),
            spatial_frames=len(rows), orientation_supported=sum(r['in_orientation_support'] for r in rows),
            new_supported=sum(r['in_orientation_support'] and not r['previous_probe'] for r in rows),
            head_hashes={label: digest(p) for label, p in heads.items()},
            anchor_image=exp['anchor_image'], radius_m=exp['radius_m'], orientation_limit_deg=30,
            region_selection='GT used solely for fixed evaluation coverage; ranking never uses GT',
            models_frozen=True, training_performed=False,
            real_candidate_protocol='LEADER seedwise + LEADER final + v1-two-stage; rank by fixed 10px camera score; no P3P/refinement/tuning',
            primary_new_frame_scope='newly evaluated frames within training orientation support, reported separately')
        write_json(args.out / 'plan.json', plan)
        print(json.dumps(plan), flush=True)
        for label in heads:
            write_json(args.out / 'state.json', dict(stage='camera_evaluation', model=label, time=time.time()))
            command = [sys.executable, '-m', 'research.glace_fusion.evaluate_scene', '--scene', str(args.test_scene),
                '--head', str(heads[label]), '--vendor', str(vendors[label]), '--deit-checkpoint', str(args.deit_checkpoint),
                '--out', str(args.out / label), '--image-stems', str(args.out / 'stems.json'),
                '--valid-mask', str(args.experiment / 'valid_mask.npy')]
            with (args.out / (label + '.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(args.out / 'state.json', dict(stage='real_leader_candidates', time=time.time()))
        command = [sys.executable, '-m', 'research.glace_fusion.real_candidate_eval',
            '--rows', str(args.out / 'rows.json'), '--stage1-coordinates', str(args.out / 'stage1'),
            '--improved-coordinates', str(args.out / 'improved'), '--scene', str(args.test_scene),
            '--checkpoint', str(args.leader_checkpoint), '--dataset-folder', str(args.dataset_folder),
            '--out', str(args.out / 'real_candidates')]
        with (args.out / 'real_candidates.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        for label, path in heads.items():
            if digest(path) != plan['head_hashes'][label]:
                raise ValueError('A frozen head changed during evaluation')
        write_json(args.out / 'state.json', dict(stage='complete', time=time.time(), full_nclt=False))
        print('FULL_REGION_COMPLETE ' + str(args.out), flush=True)
    except Exception:
        import traceback
        write_json(args.out / 'state.json', dict(stage='failed', time=time.time(), error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
