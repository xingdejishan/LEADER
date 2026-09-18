"""NCLT raw-point -> six-camera projection (Track A pre-voxel fusion).

Reuses the official NCLT calibration chain exactly as
`research/projection_audit_assets/project_vel_to_cam.py`:
  lb3-sensor frame --ssc(x_body_lb3)--> body --ssc(x_lb3_c)^-1--> camera j --K_j--> pixels.

The velodyne_sync bins are the native NCLT packed format: 8 bytes/point,
x/y/z as little-endian uint16 scaled by 0.005 with offset -100.0 (see
LEADER/data/robotcar_sdk/python/velodyne.py `data2xyzi`). Points stay in the
lb3 sensor frame; no pose is used (GT isolation).

Coordinate convention (NCLT official): T_AB maps B->A. `camera_to_body` maps
camera->body; the point transform body->camera is its inverse.
"""
import json
import os

import numpy as np

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config(path=DEFAULT_CONFIG):
    with open(path) as handle:
        return json.load(handle)


def load_vel_lb3(filename):
    """Parse native NCLT velodyne_sync bin -> (N,3) float64 xyz in lb3 sensor frame."""
    raw = np.fromfile(filename, dtype=np.uint8)
    n = raw.size // 8
    a = raw.reshape(n, 8)
    xyz = a[:, :6].reshape(n, 3, 2)
    vals = xyz[:, :, 0].astype(np.uint16) + (xyz[:, :, 1].astype(np.uint16) << 8)
    return vals.astype(np.float64) * 0.005 - 100.0


def ssc_to_homo(ssc):
    """Official NCLT 6-DOF ssc -> 4x4 homogeneous transform (sensor->parent)."""
    sr, cr = np.sin(np.pi / 180.0 * ssc[3]), np.cos(np.pi / 180.0 * ssc[3])
    sp, cp = np.sin(np.pi / 180.0 * ssc[4]), np.cos(np.pi / 180.0 * ssc[4])
    sh, ch = np.sin(np.pi / 180.0 * ssc[5]), np.cos(np.pi / 180.0 * ssc[5])
    H = np.zeros((4, 4))
    H[0, 0] = ch * cp
    H[0, 1] = -sh * cr + ch * sp * sr
    H[0, 2] = sh * sr + ch * sp * cr
    H[1, 0] = sh * cp
    H[1, 1] = ch * cr + sh * sp * sr
    H[1, 2] = -ch * sr + sh * sp * cr
    H[2, 0] = -sp
    H[2, 1] = cp * sr
    H[2, 2] = cp * cr
    H[0, 3], H[1, 3], H[2, 3], H[3, 3] = ssc[0], ssc[1], ssc[2], 1.0
    return H


class SixCameraProjector:
    """Project raw lb3-frame points into all six cameras. No pose input."""

    def __init__(self, config=None):
        self.cfg = config or load_config()
        cam_cfg = self.cfg["camera"]
        audit = self.cfg["paths"]["audit_assets"]
        self.n_cams = cam_cfg["n_cameras"]
        self.width = self.cfg["image"]["width"]
        self.height = self.cfg["image"]["height"]
        self.min_depth = self.cfg["visibility"]["min_depth"]
        self.max_depth = self.cfg["visibility"]["max_depth"]

        T_body_lb3 = ssc_to_homo(cam_cfg["body_to_lb3_ssc_deg"])
        self.T_body_lb3 = T_body_lb3
        self.T_c_body = np.zeros((self.n_cams, 4, 4))
        self.K = np.zeros((self.n_cams, 3, 3))
        for cam in range(self.n_cams):
            x_lb3_c = np.loadtxt(os.path.join(audit, "x_lb3_c%d.csv" % cam), delimiter=",")
            self.T_c_body[cam] = np.linalg.inv(ssc_to_homo(x_lb3_c)) @ np.linalg.inv(T_body_lb3)
            self.K[cam] = np.loadtxt(self._K_path(cam))

    def _K_path(self, cam):
        root = self.cfg["paths"]["wsl_root"]
        if cam == 5:
            # Cam5 uses the per-frame calibration file of the original pipeline;
            # K is constant across frames, resolve via the first frame dir listing.
            cal_dir = self.cfg["paths"]["cam5_calibration"]
            first = sorted(os.listdir(cal_dir))[0]
            return os.path.join(cal_dir, first)
        return os.path.join(root, "Cam%d" % cam, "K.txt")

    def resolve_image_path(self, frame_id, cam):
        root = self.cfg["paths"]["wsl_root"]
        win = self.cfg["paths"]["win_root"]
        if cam == 5:
            p = os.path.join(win, "glace-local", "data", "validation_scene", "train", "rgb", frame_id + ".jpg")
            if os.path.exists(p):
                return p
            return os.path.join(root, "Cam5", frame_id + ".jpg")
        return os.path.join(root, "Cam%d" % cam, frame_id + ".jpg")

    def resolve_calibration_path(self, frame_id, cam):
        if cam == 5:
            return os.path.join(self.cfg["paths"]["cam5_calibration"], frame_id + ".txt")
        return os.path.join(self.cfg["paths"]["wsl_root"], "Cam%d" % cam, "K.txt")

    def project(self, points_lb3):
        """points_lb3: (N,3) lb3 sensor frame. Returns dict of (N,6) arrays."""
        n = points_lb3.shape[0]
        homo = np.hstack([points_lb3, np.ones((n, 1))])
        uv = np.zeros((n, self.n_cams, 2))
        depth = np.full((n, self.n_cams), np.inf)
        valid = np.zeros((n, self.n_cams), dtype=bool)
        cam_xyz = np.zeros((n, self.n_cams, 3))
        for cam in range(self.n_cams):
            pc = (self.T_c_body[cam] @ homo.T).T[:, :3]
            cam_xyz[:, cam, :] = pc
            z = pc[:, 2]
            front = (z > self.min_depth) & (z < self.max_depth)
            proj = (self.K[cam] @ pc.T).T
            safe = proj[:, 2:3] > 1e-9
            u = np.where(safe[:, 0], proj[:, 0] / np.maximum(proj[:, 2], 1e-9), -1.0)
            v = np.where(safe[:, 0], proj[:, 1] / np.maximum(proj[:, 2], 1e-9), -1.0)
            inb = front & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
            uv[:, cam, 0] = u
            uv[:, cam, 1] = v
            depth[:, cam] = z
            valid[:, cam] = inb
        return {"uv": uv, "depth": depth, "valid": valid, "cam_xyz": cam_xyz}


