from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Optional, Union

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class FusionConfig:
    conf_use: float = 0.8
    conf_gap: float = 0.15
    pose_dt_m: float = 0.5
    pose_dR_rad: float = np.deg2rad(3.0)
    lidar_scale_m: float = 0.3
    camera_scale_px: float = 4.0
    sample_budget: int = 256
    retain_candidates: int = 8
    seed: int = 20
    cluster_dt_m: float = 0.5
    cluster_dR_rad: float = np.deg2rad(3.0)
    min_lidar_inliers: int = 6
    min_camera_inliers: int = 6
    min_lidar_ratio: float = 0.2
    min_camera_ratio: float = 0.2
    max_score: float = 0.7
    min_score_separation: float = 0.03
    translation_scale_m: float = 1.0
    rotation_scale_rad: float = 1.0
    min_eigenvalue: float = 1e-6
    max_condition: float = 1e10
    jacobian_step: float = 1e-5
    max_refine_evaluations: int = 150
    refine_tolerance: float = 1e-7
    min_depth_m: float = 1e-6
    invalid_depth_residual: float = 1000.0
    sample_rank_ratio: float = 1e-4
    sample_min_spread: float = 1e-6
    max_sync_delta_s: float = 0.02
    pose_validation_tolerance: float = 1e-6

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid configuration: {name}")
        for name in ("sample_budget", "retain_candidates", "seed", "min_lidar_inliers",
                     "min_camera_inliers", "max_refine_evaluations"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        for name in ("conf_use", "conf_gap", "min_lidar_ratio", "min_camera_ratio",
                     "max_score", "min_score_separation"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        for name in ("lidar_scale_m", "camera_scale_px", "translation_scale_m",
                     "rotation_scale_rad", "min_eigenvalue", "jacobian_step",
                     "refine_tolerance", "min_depth_m", "sample_rank_ratio",
                     "sample_min_spread", "pose_validation_tolerance"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if (self.retain_candidates < 2 or self.min_lidar_inliers < 1
                or self.min_camera_inliers < 1 or self.max_refine_evaluations < 1
                or self.max_condition < 1 or self.invalid_depth_residual <= 1
                or self.refine_tolerance <= np.finfo(float).eps):
            raise ValueError("Invalid verification or refinement settings")

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class EvidenceStamp:
    evidence_id: str
    timestamp_s: float
    world_frame: str
    body_frame: str
    map_id: str
    calibration_id: str
    length_unit: str = "m"


@dataclass
class LidarEvidence:
    points_body: np.ndarray
    points_world: np.ndarray
    stamp: EvidenceStamp
    original_inlier_mask: Optional[np.ndarray] = None


@dataclass
class CameraEvidence:
    pixel_xy: np.ndarray
    scene_xyz_world: np.ndarray
    stamp: EvidenceStamp
    original_inlier_mask: Optional[np.ndarray] = None
    reliability: Optional[np.ndarray] = None


@dataclass
class FusionEvidence:
    lidar: LidarEvidence
    camera: CameraEvidence
    K: np.ndarray
    T_BC: np.ndarray


@dataclass
class FusionResult:
    pose: Optional[np.ndarray]
    status: str
    reason: Optional[str] = None
    source: Optional[str] = None
    request_new_evidence: bool = False
    diagnostics: dict = field(default_factory=dict)


def _pose(T, cfg):
    T = np.asarray(T, dtype=float)
    tol = cfg.pose_validation_tolerance
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=tol, rtol=0)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=tol, rtol=0)
            or abs(np.linalg.det(T[:3, :3]) - 1) > tol):
        raise ValueError("Expected finite SE(3) T_WB (or extrinsic T_BC)")
    return T.copy()


def pose_distance(A, B):
    dt = float(np.linalg.norm(A[:3, 3] - B[:3, 3]))
    angle = np.arccos(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1))
    return dt, float(angle)


def _near(A, B, dt, dr):
    t, r = pose_distance(A, B)
    return t <= dt and r <= dr


def _array(value, width):
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != width or not np.isfinite(arr).all():
        raise ValueError(f"Expected finite Nx{width} candidate array")
    return arr.copy()


def _validate_evidence(e, cfg):
    if not isinstance(e, FusionEvidence):
        raise ValueError("Evidence loader must return FusionEvidence")
    L, C = e.lidar, e.camera
    a, b = L.stamp, C.stamp
    for stamp in (a, b):
        if not isinstance(stamp, EvidenceStamp):
            raise ValueError("Missing evidence metadata")
        if not all(isinstance(getattr(stamp, k), str) and getattr(stamp, k)
                   for k in ("evidence_id", "world_frame", "body_frame", "map_id", "calibration_id")):
            raise ValueError("Empty evidence metadata")
        if stamp.length_unit != "m" or not np.isfinite(stamp.timestamp_s):
            raise ValueError("Evidence must use metres and finite timestamps")
    for key in ("world_frame", "body_frame", "map_id", "calibration_id", "length_unit"):
        if getattr(a, key) != getattr(b, key):
            raise ValueError(f"Evidence metadata mismatch: {key}")
    if abs(a.timestamp_s - b.timestamp_s) > cfg.max_sync_delta_s:
        raise ValueError("Unsynchronized evidence")
    p, P = _array(L.points_body, 3), _array(L.points_world, 3)
    xy, X = _array(C.pixel_xy, 2), _array(C.scene_xyz_world, 3)
    if len(p) != len(P) or len(xy) != len(X):
        raise ValueError("Correspondence lengths differ")
    for mask, n in ((L.original_inlier_mask, len(p)), (C.original_inlier_mask, len(xy))):
        if mask is not None:
            mask = np.asarray(mask)
            if mask.dtype != bool or mask.shape != (n,):
                raise ValueError("Original inlier mask must index all candidates")
    if C.reliability is not None:
        u = np.asarray(C.reliability)
        if (u.shape != (len(xy),) or not np.isfinite(u).all()
                or np.any((u < 0) | (u > 1))):
            raise ValueError("Invalid correspondence reliability")
    K = np.asarray(e.K, dtype=float)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1], rtol=0, atol=1e-12)
            or abs(K[0, 1]) > 1e-12 or abs(K[1, 0]) > 1e-12):
        raise ValueError("Use undistorted pixels and standard pinhole K with zero skew")
    return p, P, xy, X, K.copy(), _pose(e.T_BC, cfg)


def _residuals(T, data, cfg):
    p, P, xy, X, K, E = data
    rL = (p @ T[:3, :3].T + T[:3, 3] - P) / cfg.lidar_scale_m
    WC = T @ E
    q = (X - WC[:3, 3]) @ WC[:3, :3]
    valid = q[:, 2] > cfg.min_depth_m
    rC = np.zeros((len(X), 2))
    rC[:, 0] = cfg.invalid_depth_residual
    projected = q[valid] @ K.T
    rC[valid] = (projected[:, :2] / projected[:, 2:] - xy[valid]) / cfg.camera_scale_px
    return rL, rC, valid


def _score(T, data, cfg):
    rL, rC, valid = _residuals(T, data, cfg)
    sL, sC = np.sum(rL * rL, axis=1), np.sum(rC * rC, axis=1)
    mL, mC = sL <= 1, (sC <= 1) & valid
    SL, SC = float(np.minimum(sL, 1).mean()), float(np.minimum(sC, 1).mean())
    return dict(score=0.5 * (SL + SC), lidar_score=SL, camera_score=SC,
                n_lidar_inliers=int(mL.sum()), n_camera_inliers=int(mC.sum()),
                lidar_ratio=float(mL.mean()), camera_ratio=float(mC.mean()),
                lidar_inlier_mask=mL, camera_inlier_mask=mC)


def _support(s, cfg):
    return (s["n_lidar_inliers"] >= cfg.min_lidar_inliers
            and s["n_camera_inliers"] >= cfg.min_camera_inliers
            and s["lidar_ratio"] >= cfg.min_lidar_ratio
            and s["camera_ratio"] >= cfg.min_camera_ratio
            and s["score"] <= cfg.max_score)


def _nondegenerate(points, cfg):
    s = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return len(s) >= 2 and s[0] > cfg.sample_min_spread and s[1] / s[0] >= cfg.sample_rank_ratio


def _rigid(p, P):
    u, _, vt = np.linalg.svd((p - p.mean(0)).T @ (P - P.mean(0)))
    D = np.eye(3)
    D[2, 2] = np.linalg.det(vt.T @ u.T)
    T = np.eye(4)
    T[:3, :3] = vt.T @ D @ u.T
    T[:3, 3] = P.mean(0) - T[:3, :3] @ p.mean(0)
    return T


def _generate(data, cfg, stats):
    import cv2

    p, P, xy, X, K, E = data
    rng = np.random.default_rng(cfg.seed)
    inverse_E = np.linalg.inv(E)
    for trial in range(cfg.sample_budget):
        if trial % 2 == 0:
            stats["lidar_sample_attempts"] += 1
            if len(p) < 3:
                continue
            idx = rng.choice(len(p), 3, replace=False)
            if not (_nondegenerate(p[idx], cfg) and _nondegenerate(P[idx], cfg)):
                stats["degenerate_samples"] += 1
                continue
            yield _rigid(p[idx], P[idx]), "LIDAR_SVD"
        else:
            stats["camera_sample_attempts"] += 1
            if len(xy) < 4:
                continue
            idx = rng.choice(len(xy), 4, replace=False)
            if not (_nondegenerate(xy[idx], cfg) and _nondegenerate(X[idx], cfg)):
                stats["degenerate_samples"] += 1
                continue
            try:
                out = cv2.solvePnPGeneric(np.ascontiguousarray(X[idx]),
                                          np.ascontiguousarray(xy[idx]), K, None,
                                          flags=cv2.SOLVEPNP_AP3P)
                for rv, tv in zip(out[1], out[2]):
                    CW = np.eye(4)
                    CW[:3, :3] = cv2.Rodrigues(rv)[0]
                    CW[:3, 3] = tv.ravel()
                    yield np.linalg.inv(CW) @ inverse_E, "CAMERA_AP3P"
            except (cv2.error, np.linalg.LinAlgError):
                stats["pnp_failures"] += 1


def _clusters(scored, cfg):
    kept = []
    for item in sorted(scored, key=lambda h: h[2]["score"]):
        if not any(_near(item[0], other[0], cfg.cluster_dt_m, cfg.cluster_dR_rad) for other in kept):
            kept.append(item)
    return kept


def _increment(T, x, cfg):
    inc = np.eye(4)
    inc[:3, :3] = Rotation.from_rotvec(x[3:] * cfg.rotation_scale_rad).as_matrix()
    inc[:3, 3] = x[:3] * cfg.translation_scale_m
    return T @ inc


def _robust_vector(r):
    s = np.sum(r * r, axis=1)
    factor = np.ones_like(s)
    np.divide(np.log1p(s), s, out=factor, where=s > 0)
    return (r * np.sqrt(factor[:, None]) / np.sqrt(len(r))).ravel()


def _refine(T, data, cfg):
    def fun(x):
        rL, rC, _ = _residuals(_increment(T, x, cfg), data, cfg)
        # Transform entire correspondence blocks so LM minimizes the specified Cauchy objective.
        return np.concatenate((_robust_vector(rL), _robust_vector(rC)))

    fit = least_squares(fun, np.zeros(6), method="lm", x_scale=1.0,
                        max_nfev=cfg.max_refine_evaluations, ftol=cfg.refine_tolerance,
                        xtol=cfg.refine_tolerance, gtol=cfg.refine_tolerance)
    return _increment(T, fit.x, cfg), dict(success=bool(fit.success), nfev=int(fit.nfev))


def _observability(T, data, score, cfg):
    rL, rC, _ = _residuals(T, data, cfg)
    mL, mC = score["lidar_inlier_mask"], score["camera_inlier_mask"]
    def residual(x):
        L, C, _ = _residuals(_increment(T, x, cfg), data, cfg)
        return np.concatenate((L[mL].ravel(), C[mC].ravel()))

    weights = np.concatenate((np.repeat(0.5 / len(rL) / (1 + (rL[mL] ** 2).sum(1)), 3),
                              np.repeat(0.5 / len(rC) / (1 + (rC[mC] ** 2).sum(1)), 2)))
    columns = []
    for k in range(6):
        dx = np.zeros(6)
        dx[k] = cfg.jacobian_step
        columns.append((residual(dx) - residual(-dx)) / (2 * cfg.jacobian_step))
    J = np.column_stack(columns)
    H = J.T @ (weights[:, None] * J)
    values = np.linalg.eigvalsh(H)
    low, high = max(0.0, float(values[0])), max(0.0, float(values[-1]))
    condition = high / low if low > 0 else None
    ok = low >= cfg.min_eigenvalue and condition is not None and condition <= cfg.max_condition
    return ok, dict(min_eigenvalue=low, condition=condition, eigenvalues=values.tolist())


def localize(T_L, c_L, T_C, c_C, *,
             evidence: Optional[Union[FusionEvidence, Callable[[], FusionEvidence]]] = None,
             config: Optional[FusionConfig] = None,
             previous_evidence_ids: Optional[tuple] = None,
             extra_hypotheses: Optional[Iterable] = None) -> FusionResult:
    started = perf_counter()
    cfg = config if config is not None else FusionConfig()
    diagnostics = {"pose_convention": "T_WB"}

    def finish(pose, status, reason=None, source=None, retry=False):
        diagnostics["elapsed_s"] = perf_counter() - started
        return FusionResult(None if pose is None else pose.copy(), status, reason, source,
                            retry, diagnostics)

    try:
        L = None if T_L is None else _pose(T_L, cfg)
        C = None if T_C is None else _pose(T_C, cfg)
        for T, confidence in ((L, c_L), (C, c_C)):
            if confidence is not None and (not np.isscalar(confidence)
                    or not np.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError("Pose confidence must be None or in [0,1]")
            if T is None and confidence is not None:
                raise ValueError("A missing pose cannot have confidence")
    except (TypeError, ValueError) as exc:
        diagnostics["error"] = str(exc)
        return finish(None, "REJECTED", "INVALID_INPUT")

    if L is not None and C is not None:
        dt, dr = pose_distance(L, C)
        diagnostics.update(disagreement_translation_m=dt, disagreement_rotation_rad=dr)
        if c_L is not None and c_C is not None:
            if max(c_L, c_C) >= cfg.conf_use and abs(c_L - c_C) >= cfg.conf_gap:
                return finish(L if c_L > c_C else C, "SELECTED",
                              source="LIDAR" if c_L > c_C else "CAMERA")
            if min(c_L, c_C) >= cfg.conf_use and dt <= cfg.pose_dt_m and dr <= cfg.pose_dR_rad:
                alpha = c_C / (c_L + c_C)
                T = np.eye(4)
                T[:3, 3] = (1 - alpha) * L[:3, 3] + alpha * C[:3, 3]
                rotations = Rotation.from_matrix(np.stack((L[:3, :3], C[:3, :3])))
                T[:3, :3] = Slerp([0, 1], rotations)(alpha).as_matrix()
                diagnostics["alpha_camera"] = float(alpha)
                return finish(T, "FUSED", source="LERP_SLERP")

    diagnostics["fallback_triggered"] = True
    if evidence is None:
        return finish(None, "REJECTED", "MISSING_EVIDENCE", retry=True)
    try:
        e = evidence() if callable(evidence) else evidence
        data = _validate_evidence(e, cfg)
    except (TypeError, ValueError, AttributeError, OSError) as exc:
        diagnostics["error"] = str(exc)
        return finish(None, "REJECTED", "INVALID_EVIDENCE", retry=True)
    diagnostics["evidence_ids"] = [e.lidar.stamp.evidence_id, e.camera.stamp.evidence_id]
    if previous_evidence_ids is not None and tuple(diagnostics["evidence_ids"]) == tuple(previous_evidence_ids):
        return finish(None, "REJECTED", "STALE_EVIDENCE", retry=True)
    diagnostics["n_lidar_candidates"], diagnostics["n_camera_candidates"] = len(data[0]), len(data[2])
    if len(data[0]) < cfg.min_lidar_inliers or len(data[2]) < cfg.min_camera_inliers:
        return finish(None, "REJECTED", "INSUFFICIENT_SUPPORT", retry=True)

    stats = dict(lidar_sample_attempts=0, camera_sample_attempts=0, degenerate_samples=0,
                 pnp_failures=0, invalid_hypotheses=0)
    diagnostics["sampling"] = stats
    hypotheses = []

    def add(T, source):
        try:
            T = _pose(T, cfg)
            score = _score(T, data, cfg)
            if not np.isfinite(score["score"]):
                raise ValueError("Nonfinite score")
            hypotheses.append((T, source, score))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            stats["invalid_hypotheses"] += 1

    for T, source in ((L, "LIDAR_ORIGINAL"), (C, "CAMERA_ORIGINAL")):
        if T is not None:
            add(T, source)
    n_extra = 0
    if extra_hypotheses is not None:
        for item in extra_hypotheses:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], str):
                T_extra, extra_label = item
            else:
                T_extra, extra_label = item, "EXTRA"
            add(T_extra, extra_label)
            n_extra += 1
    diagnostics["n_extra_hypotheses"] = n_extra
    try:
        for T, source in _generate(data, cfg, stats):
            add(T, source)
    except ImportError as exc:
        diagnostics["error"] = str(exc)
        return finish(None, "REJECTED", "DEPENDENCY_UNAVAILABLE")
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        diagnostics["error"] = str(exc)
        return finish(None, "REJECTED", "NUMERICAL_FAILURE", retry=True)
    diagnostics["n_generated_and_original"] = len(hypotheses)
    seeds = _clusters(hypotheses, cfg)[:cfg.retain_candidates]
    diagnostics["refinements"] = []
    for T, source, _ in seeds:
        try:
            refined, info = _refine(T, data, cfg)
            info["source"] = source
            diagnostics["refinements"].append(info)
            add(refined, source + "_REFINED")
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            diagnostics["refinements"].append(dict(source=source, success=False, error=str(exc)))

    clusters = _clusters(hypotheses, cfg)
    diagnostics["n_pose_clusters"] = len(clusters)
    supported = [h for h in clusters if _support(h[2], cfg)]
    diagnostics["n_supported_clusters"] = len(supported)
    if not supported:
        return finish(None, "REJECTED", "INSUFFICIENT_SUPPORT", retry=True)
    T, source, score = supported[0]
    diagnostics["best_evidence"] = score
    diagnostics["second_score_margin"] = (supported[1][2]["score"] - score["score"]
                                           if len(supported) > 1 else None)
    if len(supported) > 1 and diagnostics["second_score_margin"] < cfg.min_score_separation:
        return finish(None, "REJECTED", "AMBIGUOUS", retry=True)
    try:
        ok, spectrum = _observability(T, data, score, cfg)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return finish(None, "REJECTED", "NUMERICAL_FAILURE", retry=True)
    diagnostics["observability"] = spectrum
    if not ok:
        return finish(None, "REJECTED", "DEGENERATE", retry=True)
    return finish(T, "EVIDENCE_VERIFIED", source=source)


def evaluate_results(results, gt_T_WB, *, eps_t_m, eps_R_rad):
    if (not np.isfinite(eps_t_m) or not np.isfinite(eps_R_rad)
            or eps_t_m <= 0 or eps_R_rad <= 0):
        raise ValueError("Success thresholds must be finite and positive")
    results, truth = list(results), list(gt_T_WB)
    if len(results) != len(truth) or not results:
        raise ValueError("Evaluation requires equal, nonempty result and GT sequences")
    errors, failures, fallbacks, elapsed = [], 0, 0, []
    cfg = FusionConfig()
    for result, gt in zip(results, truth):
        gt = _pose(gt, cfg)
        fallbacks += int(result.diagnostics.get("fallback_triggered", False))
        elapsed.append(float(result.diagnostics.get("elapsed_s", 0)))
        if result.pose is not None:
            dt, dr = pose_distance(_pose(result.pose, cfg), gt)
            failures += int(not (dt < eps_t_m and dr < eps_R_rad))
            errors.append((dt, dr))
    accepted, total = len(errors), len(results)
    return dict(n_queries=total, n_accepted=accepted, n_wrongly_accepted=failures,
                coverage=accepted / total, failure_rate_accepted=failures / accepted if accepted else None,
                localization_success_rate=(accepted - failures) / total,
                median_translation_error_m=float(np.median([x[0] for x in errors])) if errors else None,
                median_rotation_error_rad=float(np.median([x[1] for x in errors])) if errors else None,
                fallback_rate=fallbacks / total, mean_latency_s=float(np.mean(elapsed)),
                eps_t_m=eps_t_m, eps_R_rad=eps_R_rad)


def select_configuration(validation_runs, gt_T_WB, *, eps_t_m, eps_R_rad,
                         target_failure_rate, validation_split_id):
    if not validation_split_id or not 0 <= target_failure_rate <= 1:
        raise ValueError("Provide a validation split ID and risk target in [0,1]")
    truth = list(gt_T_WB)
    trials = []
    for cfg, results in validation_runs:
        if not isinstance(cfg, FusionConfig):
            raise ValueError("Each validation run must provide a FusionConfig")
        metrics = evaluate_results(results, truth, eps_t_m=eps_t_m, eps_R_rad=eps_R_rad)
        trials.append(dict(config=asdict(cfg), metrics=metrics))
    feasible = [t for t in trials if t["metrics"]["n_accepted"] > 0
                and t["metrics"]["failure_rate_accepted"] <= target_failure_rate]
    best = min(feasible, key=lambda t: (-t["metrics"]["coverage"],
               t["metrics"]["failure_rate_accepted"], t["metrics"]["mean_latency_s"])) if feasible else None
    return dict(selected_config=None if best is None else best["config"],
                validation_split_id=validation_split_id, target_failure_rate=target_failure_rate,
                eps_t_m=eps_t_m, eps_R_rad=eps_R_rad, trials=trials)
