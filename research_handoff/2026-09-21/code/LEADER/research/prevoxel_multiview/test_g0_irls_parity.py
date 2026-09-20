import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from research.prevoxel_multiview.g0_irls_parity import evaluate_parity, pose_difference, reference_poses


def pose(translation=(0., 0., 0.), rotation_deg=0.):
    output = np.eye(4)
    output[:3, :3] = Rotation.from_euler("z", rotation_deg, degrees=True).as_matrix()
    output[:3, 3] = translation
    return output


def test_pose_difference_reports_translation_and_rotation():
    translation, rotation = pose_difference(pose((.3, .4, 0.), 5.), pose())
    assert translation == pytest.approx(.5)
    assert rotation == pytest.approx(5.)


def test_reference_poses_uses_initial_pose_for_b0_artifacts():
    output = reference_poses({"records": [{"frame_id": "one", "initial_pose": pose().tolist(),
                                             "final_pose": pose((1., 0., 0.)).tolist()}]})
    np.testing.assert_allclose(output["one"], pose())


def test_evaluate_parity_uses_selected_row_order_for_seed_and_thresholds():
    rows = [{"frame_id": "first"}, {"frame_id": "second"}]
    baseline = {"first": pose(), "second": pose()}
    received_seeds = []

    def runner(row, seed):
        received_seeds.append(seed)
        return pose((5e-7, 0., 0.))

    records = evaluate_parity(rows, baseline, runner, 1e-6, 1e-5, 2089)
    assert received_seeds == [2089, 2090]
    assert all(record["within_tolerance"] for record in records)
