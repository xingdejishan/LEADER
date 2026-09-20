import math

import numpy as np
import torch

from visual_pose_candidates import CandidatePose, SIDE, final_pose_loss, hypotheses, rotation_matrix
from local_visual_refinement_roma import apply_local_delta
from oracle_pose_refinement import pose_error


def sample():
    torch.manual_seed(7)
    n = 7
    h = hypotheses()
    return {'visual': torch.randn(n, SIDE * SIDE, 36),
            'geometry': torch.randn(n, SIDE * SIDE, 7),
            'valid': torch.ones(n, SIDE * SIDE, dtype=torch.bool),
            'projection_grid': torch.randn(n, len(h), 2) * .3,
            'hypotheses': torch.tensor(h, dtype=torch.float32),
            'cameras': torch.tensor([0, 0, 1, 1, 2, 2, 2]),
            'lidar_hessian': torch.eye(6)}


def test_pose_gradient_reaches_visual_scorer():
    model = CandidatePose()
    data = sample()
    delta, probability = model(data)
    truth = torch.eye(4)
    truth[0, 3] = .05
    truth[:3, :3] = rotation_matrix(torch.tensor([0., .01, 0.]))
    loss = final_pose_loss(delta, torch.eye(4), truth)
    loss.backward()
    gradient = model.scorer[0].weight.grad
    assert torch.isfinite(gradient).all()
    assert gradient[:, :36].abs().sum() > 0
    assert torch.allclose(probability.sum(), torch.tensor(1.))


def test_zero_visual_ignores_all_visual_values():
    model = CandidatePose()
    data = sample()
    first = model(data, True)[0]
    data['visual'] = torch.randn_like(data['visual']) * 1000
    second = model(data, True)[0]
    assert torch.equal(first, second)


def test_observation_permutation_preserves_shared_pose():
    model = CandidatePose()
    data = sample()
    original = model(data)[0]
    order = torch.randperm(len(data['visual']))
    for key in ('visual', 'geometry', 'valid', 'projection_grid', 'cameras'):
        data[key] = data[key][order]
    assert torch.allclose(original, model(data)[0], atol=1e-7)


def test_rotation_and_translation_loss_units():
    delta = torch.tensor([.1, 0., 0., 0., 0., math.radians(1)])
    assert torch.allclose(final_pose_loss(delta, torch.eye(4), torch.eye(4)), torch.tensor(2.), atol=1e-5)
    r = rotation_matrix(delta[3:])
    assert torch.allclose(r @ r.T, torch.eye(3), atol=1e-6)


def test_hypotheses_symmetric_and_bounded():
    h = hypotheses()
    np.testing.assert_allclose(h.mean(0), 0, atol=1e-15)
    assert np.linalg.norm(h[:, :3], axis=1).max() <= .15 + 1e-10
    assert np.linalg.norm(h[:, 3:], axis=1).max() <= math.radians(1.5) + 1e-10


def test_no_valid_evidence_keeps_leader_pose():
    model = CandidatePose()
    data = sample()
    data['valid'][:] = False
    delta, posterior = model(data)
    assert torch.equal(delta, torch.zeros(6))
    assert posterior[0] == 1


def test_loss_agrees_with_existing_pose_metrics_in_world_frame():
    initial = np.eye(4)
    initial[:3, :3] = rotation_matrix(torch.tensor([.2, -.3, 1.], dtype=torch.float64)).numpy()
    initial[:3, 3] = [125., -305., 6.]
    truth = apply_local_delta(initial, np.array([.04, -.02, .01, .003, -.005, .008]))
    delta = np.array([.01, .02, -.01, -.002, .006, -.003])
    translation, rotation = pose_error(apply_local_delta(initial, delta), truth)
    loss = final_pose_loss(torch.tensor(delta), torch.tensor(initial), torch.tensor(truth))
    np.testing.assert_allclose(float(loss), translation / .1 + rotation, rtol=1e-9)
