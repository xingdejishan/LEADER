import numpy as np

from .joint_solver import validate_pose


def solver_pose(pose):
    pose = validate_pose(pose, tol=1e-4)
    u, _, vt = np.linalg.svd(pose[:3, :3])
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    pose[:3, :3] = u @ correction @ vt
    return validate_pose(pose)
