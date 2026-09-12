"""Fusion packet construction.

Turns the two modality outputs into the fixed fusion interface:

    LEADER side   C_L = {(T_corr^-1 c_local_i,  c_pred_i + center_t)}   (3D-3D pool)
    Camera side   C_C = {(u_j, P_j^W)}                                  (2D-3D pool)

Confidences are support rates of the final pose over each modality's FULL
correspondence pool (never raw u_pred / inlier_count), optionally mapped to
P(success | q) by isotonic calibrators fitted on a validation split:

    q_L = #{ ||T_L p_i - P_i||  < s_L } / N_L
    q_C = #{ ||pi((T_C E)^-1 P_j) - u_j|| < s_C } / N_C
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from .lidar_camera_fusion import (CameraEvidence, EvidenceStamp, FusionEvidence,
                                      LidarEvidence)
except ImportError:  # running with the package directory itself on sys.path
    from lidar_camera_fusion import (CameraEvidence, EvidenceStamp, FusionEvidence,
                                     LidarEvidence)


def lidar_pool_from_export(export: dict) -> dict:
    """LEADER export (pre-top-50% pool) -> body/world correspondence pool.

    export keys: c_local_all [N,3] (leveled frame, i.e. T_corr applied),
    c_pred_all [N,3] (world - center_t), T_corr [4,4] (raw body -> leveled),
    center_t [3]. The leveling correction is inverted so the packet's source
    points live in the same raw body frame B that T_WB maps to the world."""
    c_local = np.asarray(export["c_local_all"], dtype=np.float64)
    c_pred = np.asarray(export["c_pred_all"], dtype=np.float64)
    T_corr = np.asarray(export["T_corr"], dtype=np.float64)
    center_t = np.asarray(export["center_t"], dtype=np.float64).reshape(3)
    p_body = (np.linalg.inv(T_corr) @ np.column_stack(
        [c_local, np.ones(len(c_local))]).T).T[:, :3]
    p_world = c_pred + center_t[None, :]
    u = np.asarray(export.get("u_pred_all", np.zeros(len(c_pred))), dtype=np.float64)
    return {"p_body": p_body, "p_world": p_world, "u": u}


def lidar_support_rate(T_WB, p_body, p_world, threshold_m) -> float:
    T = np.asarray(T_WB, dtype=float)
    r = np.linalg.norm(p_body @ T[:3, :3].T + T[:3, 3] - p_world, axis=1)
    return float(np.mean(r < threshold_m)) if len(r) else 0.0


def camera_support_rate(T_WB, T_BC, K, uv, xyz_world, threshold_px) -> float:
    """Support of T_WB over the full camera correspondence set. Non-positive or
    non-finite depths count as outliers but stay in the denominator."""
    T_WC = np.asarray(T_WB, dtype=float) @ np.asarray(T_BC, dtype=float)
    q = (xyz_world - T_WC[:3, 3]) @ T_WC[:3, :3]  # world -> camera (R^T via row vectors)
    valid = np.isfinite(q).all(axis=1) & (q[:, 2] > 1e-6)
    err = np.full(len(q), np.inf)
    proj = q[valid] @ np.asarray(K, dtype=float).T
    with np.errstate(invalid="ignore", divide="ignore"):
        uv_proj = proj[:, :2] / proj[:, 2:]
    err[valid] = np.linalg.norm(uv_proj - np.asarray(uv, dtype=float)[valid], axis=1)
    return float(np.mean(err < threshold_px)) if len(err) else 0.0


class IsotonicCalibrator:
    """Monotone PAVA fit of P(success | support rate). Identity until fitted."""

    def __init__(self):
        self._x = None
        self._y = None

    @property
    def calibrated(self) -> bool:
        return self._x is not None

    def fit(self, q, success):
        q = np.asarray(q, dtype=float)
        y = np.asarray(success, dtype=float)
        if q.shape != y.shape or q.ndim != 1 or len(q) < 2:
            raise ValueError("Calibration needs matched 1-D q/success arrays")
        order = np.argsort(q, kind="stable")
        q, y = q[order], y[order]
        # PAVA with unit weights; track block sizes as integers for x boundaries
        values = y.astype(float).tolist()
        lengths = [1] * len(values)
        i = 0
        while i < len(values) - 1:
            if values[i] > values[i + 1] + 1e-15:
                n = lengths[i] + lengths[i + 1]
                v = (values[i] * lengths[i] + values[i + 1] * lengths[i + 1]) / n
                values[i:i + 2] = [v]
                lengths[i:i + 2] = [n]
                i = max(i - 1, 0)
            else:
                i += 1
        xs, vs = [], []
        cum = 0
        for v, n in zip(values, lengths):
            cum += n
            xs.append(q[cum - 1])
            vs.append(v)
        self._x = np.asarray(xs, dtype=float)
        self._y = np.asarray(vs, dtype=float)
        return self

    def __call__(self, q):
        q = np.asarray(q, dtype=float)
        if self._x is None:
            return np.clip(q, 0.0, 1.0)
        return np.clip(np.interp(q, self._x, self._y, left=self._y[0], right=self._y[-1]), 0.0, 1.0)

    def save(self, path):
        payload = {"kind": "isotonic_support_rate",
                   "x": None if self._x is None else self._x.tolist(),
                   "y": None if self._y is None else self._y.tolist()}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cal = cls()
        if payload.get("x") is not None:
            cal._x = np.asarray(payload["x"], dtype=float)
            cal._y = np.asarray(payload["y"], dtype=float)
        return cal


def make_evidence_stamps(frame_id, lidar_ts_s, camera_ts_s, map_id="leader-nclt-v1",
                         calibration_id="nclt-cam-default", world_frame="world", body_frame="body"):
    lidar = EvidenceStamp(evidence_id=f"{frame_id}-lidar", timestamp_s=float(lidar_ts_s),
                          world_frame=world_frame, body_frame=body_frame,
                          map_id=map_id, calibration_id=calibration_id)
    camera = EvidenceStamp(evidence_id=f"{frame_id}-camera", timestamp_s=float(camera_ts_s),
                           world_frame=world_frame, body_frame=body_frame,
                           map_id=map_id, calibration_id=calibration_id)
    return lidar, camera


def make_fusion_evidence(p_body, p_world, uv, xyz_world, K, T_BC,
                         lidar_stamp, camera_stamp,
                         lidar_inlier_mask=None, camera_inlier_mask=None, camera_reliability=None):
    return FusionEvidence(
        lidar=LidarEvidence(points_body=p_body, points_world=p_world,
                            stamp=lidar_stamp, original_inlier_mask=lidar_inlier_mask),
        camera=CameraEvidence(pixel_xy=uv, scene_xyz_world=xyz_world,
                              stamp=camera_stamp, original_inlier_mask=camera_inlier_mask,
                              reliability=camera_reliability),
        K=K, T_BC=T_BC)


def packet_summary(T_L, c_L, T_C, c_C, pool_l, pool_c, K, E):
    """The per-frame packet in the design-document shape (all plain arrays)."""
    return {
        "T_L": T_L, "c_L": c_L,
        "T_C": T_C, "c_C": c_C,
        "C_L": {"p_B": pool_l["p_body"], "P_W": pool_l["p_world"], "u": pool_l["u"]},
        "C_C": {"u": pool_c["uv"], "P_W": pool_c["xyz_world"]},
        "K": K, "E": E,
    }


def diverse_poses(poses, k, dt_m=0.5, dR_rad=np.deg2rad(3.0)):
    """Greedy farthest-point subset in (translation, rotation) pose distance.
    Keeps at most k poses that are mutually dissimilar; used to pick fallback
    hypotheses from the SC2-PCR seedwise pool."""
    poses = [np.asarray(p, dtype=float) for p in np.asarray(poses)]
    if len(poses) <= k:
        return list(poses)

    def dist(a, b):
        t = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
        c = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)
        return t + 0.5 * float(np.arccos(c))

    chosen = [poses[0]]
    while len(chosen) < k:
        best, best_d = None, -1.0
        for p in poses:
            d = min(dist(p, c) for c in chosen)
            if d > best_d:
                best, best_d = p, d
        if best_d <= 1e-9:
            break
        chosen.append(best)
    return chosen
