import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class Local905EvaluatorTests(unittest.TestCase):
    def test_evaluator_checks_hash_and_keeps_failures_in_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / 'data' / 'train_scene'
            poses = scene / 'train' / 'poses'
            poses.mkdir(parents=True)
            body_from_camera = np.eye(4)
            body_from_camera[0, 3] = 2
            (scene / 'scene_meta.json').write_text(json.dumps({
                'T_BC_camera_to_body': body_from_camera.tolist()}), encoding='utf-8')
            keys = ['scans/date/velodyne_sync/1.bin', 'scans/date/velodyne_sync/2.bin']
            truth = np.eye(4)
            truth[1, 3] = 3
            camera_pose = truth @ body_from_camera
            for stem in ('1', '2'):
                np.savetxt(poses / (stem + '.txt'), camera_pose)
            split = root / 'split.json'
            split.write_text(json.dumps({'splits': {'test': keys}}), encoding='utf-8')
            split_hash = hashlib.sha256(split.read_bytes()).hexdigest()
            predictions = root / 'predictions.json'
            predictions.write_text(json.dumps({
                'protocol': 'local905_gt_isolated_online_v1',
                'subset': 'test', 'split_sha256': split_hash,
                'elapsed_seconds': 1,
                'predictions': [
                    {'scan': keys[0], 'status': 'ok', 'T_world_body': truth.tolist()},
                    {'scan': keys[1], 'status': 'failed'},
                ],
            }), encoding='utf-8')
            checksum = hashlib.sha256(predictions.read_bytes()).hexdigest()
            predictions.with_suffix('.sha256').write_text(checksum + '\n', encoding='ascii')
            output = root / 'report.json'
            command = [sys.executable, '-m', 'tools.eval_local905_gt',
                       '--data_root', str(root / 'data'), '--split', str(split),
                       '--predictions', str(predictions), '--out', str(output)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual((report['frames'], report['successful_frames'],
                              report['failed_frames']), (2, 1, 1))
            self.assertIsNone(report['all_frame_mpe_mean_m'])
            self.assertAlmostEqual(report['success_only_mpe_mean_m'], 0)
            predictions.write_text(predictions.read_text(encoding='utf-8') + ' ', encoding='utf-8')
            failed = subprocess.run(command[:-1] + [str(root / 'second.json')],
                                    capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('Prediction file changed', failed.stderr)


if __name__ == '__main__':
    unittest.main()
