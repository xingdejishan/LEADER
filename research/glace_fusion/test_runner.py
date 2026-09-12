import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from . import make_glace_scene as scene
from . import run_fusion_eval as runner
from .joint_solver import JointSolverConfig, solve
from .nclt_camera import (NCLTTrajectory, camera_rows, preprocess_image,
                          stored_intrinsics, validate_dates)


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.camera = self.root / 'camera'
        calibration = self.camera / 'calibration' / 'cam_params'
        calibration.mkdir(parents=True)
        self.K = np.array([[1000., 0, 808], [0, 900, 616], [0, 0, 1]])
        np.savetxt(calibration / 'K_cam5.csv', self.K, delimiter=',')
        np.savetxt(calibration / 'x_lb3_c5.csv', np.zeros(6), delimiter=',')
        self.rows = []
        for i, date in enumerate(['2012-01-22', '2012-02-12']):
            ts = 1_000_000 + i * 10_000_000
            path = self.camera / f'{date}.jpg'
            Image.new('RGB', (808, 616), (120, 80, 10)).save(path)
            self.rows.append(dict(sequence=date, camera='Cam5',
                                  group_target_timestamp=ts + 300_000,
                                  original_image_timestamp=ts + 500_000,
                                  saved_path=path.name, original_width=1616, original_height=1232))
            gt_dir = self.root / 'NCLT' / date
            gt_dir.mkdir(parents=True)
            np.savetxt(gt_dir / f'groundtruth_{date}.csv',
                       [[ts, 0, 0, 0, 0, 0, 0], [ts + 1_000_000, 2, 0, 0, 0, 0, 0]], delimiter=',')
        with (self.camera / 'all_images.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

    def test_intrinsics_raw_and_presized(self):
        for size, scale in [((1616, 1232), 1.), ((808, 616), .5)]:
            path = self.root / f'{scale}.png'
            Image.new('RGB', size).save(path)
            K, _ = stored_intrinsics(self.K, self.rows[0], path)
            np.testing.assert_allclose(K[:2], self.K[:2] * scale)
            image, final_K = preprocess_image(path, K, 616)
            self.assertEqual(image.shape, (616, 808))
            np.testing.assert_allclose(final_K[:2], self.K[:2] * .5)

    def test_interpolation_translation_slerp_no_extrapolation(self):
        path = self.root / 'gt.csv'
        np.savetxt(path, [[1000000, 0, 0, 0, 0, 0, np.deg2rad(170)],
                          [2000000, 2, 0, 0, 0, 0, np.deg2rad(-170)]], delimiter=',')
        trajectory = NCLTTrajectory(path)
        mid = trajectory.at([1500000])[0]
        self.assertAlmostEqual(mid[0, 3], 1.)
        np.testing.assert_allclose(mid[:3, :3], Rotation.from_euler('z', 180, degrees=True).as_matrix(), atol=1e-12)
        with self.assertRaises(ValueError):
            trajectory.at([999999])

    def test_scene_uses_exposure_gt_and_stored_K(self):
        output = self.root / 'scene'
        argv = ['scene', '--dataset_folder', str(self.root), '--camera_root', str(self.camera),
                '--out', str(output), '--train_dates', '2012-01-22',
                '--test_dates', '2012-02-12', '--body_to_lb3_ssc_deg', '0,0,0,0,0,0']
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            scene.main()
        pose = np.loadtxt(output / 'train/poses/1500000.txt')
        self.assertAlmostEqual(pose[0, 3], 1.)
        K = np.loadtxt(output / 'train/calibration/1500000.txt')
        np.testing.assert_allclose(K[:2], self.K[:2] * .5)
        self.assertEqual(camera_rows(self.camera, 5)[0]['timestamp_us'], 1500000)
        with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
            scene.main()

    def test_split_leakage_and_missing_dates(self):
        with self.assertRaises(ValueError):
            validate_dates(['2012-02-12'], ['2012-03-31'])
        argv = ['scene', '--dataset_folder', str(self.root), '--camera_root', str(self.camera),
                '--out', str(self.root / 'missing')]
        with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
            scene.main()

    def test_calibration_metadata_original_path(self):
        original = self.root / 'original'
        original.mkdir()
        (self.camera / 'calibration/cam_params').rename(original / 'cam_params')
        (self.camera / 'calibration/processing_metadata.json').write_text(
            json.dumps({'original_calibration_dir': str(original)}))
        K, _ = scene.calibration_chain(self.camera, 5, '0,0,0,0,0,0', None)
        np.testing.assert_allclose(K, self.K)

    def test_camera_to_body_extrinsic_direction(self):
        camera_pose = np.array([.04, -.002, 0., 160., 89., 161.])
        np.savetxt(self.camera / 'calibration/cam_params/x_lb3_c5.csv', camera_pose, delimiter=',')
        body_pose = np.array([.035, .002, -1.23, -179.93, -.23, .50])
        def official_ssc(values):
            roll, pitch, yaw = np.deg2rad(values[3:])
            sr, cr, sp, cp, sy, cy = np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch), np.sin(yaw), np.cos(yaw)
            T = np.eye(4)
            T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                         [-sp, cp * sr, cp * cr]]
            T[:3, 3] = values[:3]
            return T
        official_body_to_camera = np.linalg.inv(official_ssc(camera_pose)) @ np.linalg.inv(official_ssc(body_pose))
        _, E = scene.calibration_chain(self.camera, 5, ','.join(map(str, body_pose)), None)
        np.testing.assert_allclose(np.linalg.inv(E), official_body_to_camera, atol=1e-12)

    def test_module_and_script_entrypoints(self):
        root = Path(__file__).resolve().parents[2]
        for entry in [['-m', 'research.glace_fusion.run_fusion_eval'],
                      ['research/glace_fusion/run_fusion_eval.py']]:
            result = subprocess.run([sys.executable] + entry + ['--help'], cwd=root,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_full_runner_reports_all_backends(self):
        rng = np.random.default_rng(4)
        points = rng.uniform([-2, -2, 4], [2, 2, 8], size=(80, 3))
        T = np.eye(4)
        T[0, 3] = 1
        world = points + T[:3, 3]
        K = self.K.copy()
        K[:2] *= .5
        projected = points @ K.T
        uv = projected[:, :2] / projected[:, 2:]
        pool = self.root / 'pool'
        pool.mkdir()
        for index, ts in enumerate([11_490_000, 11_900_000]):
            np.savez(pool / f'{index}.npz', scan_timestamp_us=ts,
                     c_local_all=points, c_pred_all=world, u_pred_all=np.ones(len(points)),
                     T_corr=np.eye(4), center_t=np.zeros(3), T_WB=T, T_WB_gt=T,
                     seedwise_T_WB=np.array([T]))
        output = SimpleNamespace(T_WB=T, uv=uv, xyz_world=world, inlier_count=80,
                                 inlier_mask=np.ones(80, dtype=bool))
        for backend in ['joint', 'compare', 'fallback']:
            dest = self.root / backend
            argv = ['runner', '--pool_dir', str(pool), '--vendor_dir', 'unused',
                    '--glace_head', 'unused', '--camera_root', str(self.camera),
                    '--dataset_folder', str(self.root), '--allow_partial',
                    '--body_to_lb3_ssc_deg', '0,0,0,0,0,0',
                    '--backend', backend, '--out_dir', str(dest)]
            with patch.object(sys, 'argv', argv), patch.object(runner, 'GLACEAdapter') as adapter, contextlib.redirect_stdout(io.StringIO()):
                adapter.return_value.infer.return_value = output
                runner.main()
            report = json.loads((dest / 'report.json').read_text())
            self.assertEqual(report['n_frames'], 1)
            self.assertEqual(report['n_input_lidar_frames'], 2)
            self.assertEqual(report['n_skipped_sync'], 1)
            self.assertEqual(report['synchronized_fraction'], .5)
            self.assertEqual(report['glace_baseline']['n_accepted'], 1)
            self.assertAlmostEqual(report['glace_baseline']['errors']['mean_t'], 0.)
            self.assertAlmostEqual(report['leader_baseline']['errors']['mean_t'], .02)
            if backend == 'compare':
                for mode in ['select', 'joint', 'joint_refine']:
                    self.assertEqual(report[mode]['n_frames'], 1)
            self.assertEqual(len(json.loads((dest / 'records.json').read_text())), 1)


class TestSingleModalOrder(unittest.TestCase):
    def test_modality_order_and_margin(self):
        from . import joint_solver
        for modality in ['LIDAR', 'CAMERA']:
            for separation, expected in [(.2, 'SINGLE_MODAL'), (.02, 'REJECTED')]:
                A, B = np.eye(4), np.eye(4)
                B[0, 3] = 2
                own = modality.lower()
                other = 'camera' if own == 'lidar' else 'lidar'
                problem = SimpleNamespace(cfg=JointSolverConfig(), uv=np.zeros((0, 2)),
                                          xyz_world=np.zeros((0, 3)), K=np.eye(3), T_BC=np.eye(4))
                def score(T):
                    best = T[0, 3] > 1
                    return {'score': .6 if best else .5,
                            own + '_score': .1 if best else .1 + separation,
                            other + '_score': 1. if best else .8}
                problem.score = score
                problem.support = lambda T: {'n_' + own: 20, own + '_ratio': .8,
                                              'n_' + other: 0, other + '_ratio': 0.}
                problem.observability = lambda T: (True, {})
                with patch.object(joint_solver, 'p3p_candidates', return_value=([], {})):
                    result = solve(problem, A, B, mode='joint')
                self.assertEqual(result.status, expected)
                if expected == 'SINGLE_MODAL':
                    np.testing.assert_allclose(result.pose, B)
                    self.assertEqual(result.modality, modality)


if __name__ == '__main__':
    unittest.main()
