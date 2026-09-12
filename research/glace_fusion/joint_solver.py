"""Shared pose solver: the every-frame joint backend (v2 architecture).

Dual-SCR front-ends, shared geometric backend. LEADER and GLACE are unchanged;
their correspondences and pose candidates compete on the same multimodal
evidence, and a single rigid body variable T_WB is jointly fitted to the
support of both modalities. This backend replaces the confidence-gated
SELECT / LERP+SLERP quick path: every frame goes through

    candidates -> joint scoring -> joint refinement -> acceptance.

Candidate pool per frame:
    H = {T_L, T_C} + all valid SC2-PCR seedwise poses (no fitness pre-filter)
        + P3P hypotheses sampled from the full camera correspondence set.

Scoring (weights fixed per frame, modality-balanced):
    w_i^L = a_i / sum_k a_k,  a_i = exp(ln(10)/pi * atan(clip(s_i^L, +-10 pi)))
        (TRR-style bounded transform of LEADER's per-point reliability)
    w_j^C = 1 / N_C (uniform prior; outlier influence comes from the residuals)
    S(T) = 1/2 sum_i w_i^L min(||r_i^L(T)||^2, 1)
         + 1/2 sum_j w_j^C min(||r_j^C(T)||^2, 1)
    r_i^L = (R p_i + t - P_i) / s_L                (metres / s_L)
    r_j^C = (pi_K(E^-1 T^-1 P_j) - u_j) / s_C      (pixels / s_C)
    non-positive / non-finite projection depths count as full outliers and
    stay in the denominator.

Refinement: on each kept candidate's support set, minimise
    1/2 sum w_i rho(||r_i||^2),  rho(z) = log(1+z)
over one rigid T_WB (SciPy LM, the Ceres recipe - quaternion manifold /
ScaledLoss semantics - implemented in block form so that the LM objective is
exactly the weighted Cauchy loss; no prior toward T_L/T_C). Refined candidates
are re-scored on the FULL correspondence sets and the pre-refinement candidate
is retained.

Acceptance (every frame):
    JOINT          two-modality support, unique best cluster, observable
    SINGLE_MODAL   no jointly supported candidate, but exactly one modality
                   has a unique candidate passing its own stricter acceptance
                   while the other modality supports nothing (degraded,
                   explicitly labelled - never reported as joint)
    AMBIGUOUS      two supported clusters with close scores
    REJECTED       insufficient support
    plus DEGENERATE when J^T W J is rank deficient at the winner.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from time import perf_counter
from typing import Optional

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class JointSolverConfig:
    lidar_scale_m: float = 0.3
    camera_scale_px: float = 4.0
    min_depth_m: float = 1e-6
    invalid_depth_px: float = 1e3
    trr_weight_scale: float = 10.0
    camera_sample_budget: int = 256
    camera_grid: int = 4
    retain_candidates: int = 4
    refine_max_nfev: int = 20
    refine_tolerance: float = 1e-8
    max_refine_evaluations: int = 150
    cluster_dt_m: float = 0.5
    cluster_dR_rad: float = np.deg2rad(3.0)
    min_score_separation: float = 0.03
    min_lidar_inliers: int = 6
    min_camera_inliers: int = 6
    min_lidar_ratio: float = 0.2
    min_camera_ratio: float = 0.2
    max_score: float = 0.7
    single_modal_min_inliers: int = 12
    single_modal_min_ratio: float = 0.35
    single_modal_max_score: float = 0.5
    single_modal_min_separation: float = 0.06
    min_eigenvalue: float = 1e-6
    max_condition: float = 1e10
    jacobian_step: float = 1e-5
    translation_scale_m: float = 1.0
    rotation_scale_rad: float = 1.0
    seed: int = 20
    pose_validation_tolerance: float = 1e-6

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid configuration: {name}")
        for name in ("camera_sample_budget", "camera_grid", "retain_candidates",
                     "refine_max_nfev", "max_refine_evaluations", "seed",
                     "min_lidar_inliers", "min_camera_inliers",
                     "single_modal_min_inliers"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        for name in ("min_lidar_ratio", "min_camera_ratio", "max_score",
                     "min_score_separation", "single_modal_min_ratio",
                     "single_modal_max_score", "single_modal_min_separation"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        for name in ("lidar_scale_m", "camera_scale_px", "min_depth_m",
                     "invalid_depth_px", "trr_weight_scale", "camera_grid",
                     "translation_scale_m", "rotation_scale_rad",
                     "min_eigenvalue", "jacobian_step"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class JointResult:
    pose: Optional[np.ndarray]
    status: str          # JOINT / SINGLE_MODAL / AMBIGUOUS / DEGENERATE / REJECTED
    source: Optional[str] = None
    modality: Optional[str] = None   # set for SINGLE_MODAL
    reason: Optional[str] = None
    diagnostics: dict = field(default_factory=dict)


def validate_pose(T, tol=1e-6):
    T = np.asarray(T, dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=tol, rtol=0)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=tol, rtol=0)
            or abs(np.linalg.det(T[:3, :3]) - 1) > tol):
        raise ValueError("Expected finite SE(3) T_WB")
    return T.copy()


def pose_distance(A, B):
    dt = float(np.linalg.norm(A[:3, 3] - B[:3, 3]))
    angle = np.arccos(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1))
    return dt, float(angle)


def lidar_reliability_weights(u_raw, trr_scale=10.0):
    """a_i = exp(ln(scale)/pi * atan(clip(s_i^L, +-10pi))), normalised to sum 1."""
    u = np.clip(np.asarray(u_raw, dtype=float).ravel(), -10 * np.pi, 10 * np.pi)
    a = np.exp(np.log(trr_scale) / np.pi * np.arctan(u))
    total = a.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Degenerate reliability weights")
    return a / total


def _increment(T, x, cfg):
    from scipy.spatial.transform import Rotation
    inc = np.eye(4)
    inc[:3, :3] = Rotation.from_rotvec(x[3:] * cfg.rotation_scale_rad).as_matrix()
    inc[:3, 3] = x[:3] * cfg.translation_scale_m
    return T @ inc


class JointProblem:
    """Bundle of one frame's multimodal evidence with fixed per-frame weights."""

    def __init__(self, p_body, p_world, u_raw, uv, xyz_world, K, T_BC,
                 config: JointSolverConfig):
        self.cfg = config
        self.p_body = np.asarray(p_body, dtype=float)
        self.p_world = np.asarray(p_world, dtype=float)
        self.uv = np.asarray(uv, dtype=float)
        self.xyz_world = np.asarray(xyz_world, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.T_BC = validate_pose(T_BC, config.pose_validation_tolerance)
        if self.p_body.shape != self.p_world.shape or self.p_body.ndim != 2 or self.p_body.shape[1] != 3:
            raise ValueError("LiDAR pool shapes mismatch")
        if self.uv.shape != (len(self.xyz_world), 2):
            raise ValueError("Camera pool shapes mismatch")
        if len(self.p_body) != len(np.asarray(u_raw, dtype=float).ravel()):
            raise ValueError("LiDAR reliability length mismatch")
        self.u_raw = np.asarray(u_raw, dtype=float).ravel().copy()
        self.w_L = lidar_reliability_weights(u_raw, config.trr_weight_scale)
        self.w_C = np.full(len(self.xyz_world), 1.0 / max(len(self.xyz_world), 1))
        self._depth_cache = {}

    # ---- residuals ------------------------------------------------------
    def residuals(self, T):
        cfg = self.cfg
        T = np.asarray(T, dtype=float)
        rL = (self.p_body @ T[:3, :3].T + T[:3, 3] - self.p_world) / cfg.lidar_scale_m
        T_WC = T @ self.T_BC
        q = (self.xyz_world - T_WC[:3, 3]) @ T_WC[:3, :3]
        valid = np.isfinite(q).all(axis=1) & (q[:, 2] > cfg.min_depth_m)
        rC = np.full((len(q), 2), cfg.invalid_depth_px / cfg.camera_scale_px)
        if valid.any():
            proj = q[valid] @ self.K.T
            rC[valid] = (proj[:, :2] / proj[:, 2:] - self.uv[valid]) / cfg.camera_scale_px
        return rL, rC, valid

    def score(self, T):
        rL, rC, _ = self.residuals(T)
        sL = np.einsum('ij,ij->i', rL, rL)
        sC = np.einsum('ij,ij->i', rC, rC)
        SL = float(np.sum(self.w_L * np.minimum(sL, 1.0)))
        SC = float(np.sum(self.w_C * np.minimum(sC, 1.0)))
        return dict(score=0.5 * (SL + SC), lidar_score=SL, camera_score=SC)

    def support(self, T):
        rL, rC, valid = self.residuals(T)
        mL = np.einsum('ij,ij->i', rL, rL) <= 1.0
        mC = (np.einsum('ij,ij->i', rC, rC) <= 1.0) & valid
        return dict(n_lidar=int(mL.sum()), n_camera=int(mC.sum()),
                    lidar_ratio=float(mL.mean()), camera_ratio=float(mC.mean()),
                    lidar_inlier_mask=mL, camera_inlier_mask=mC)

    # ---- joint refinement on the support set ----------------------------
    def refine(self, T0):
        cfg = self.cfg
        sup = self.support(T0)
        mL, mC = sup["lidar_inlier_mask"], sup["camera_inlier_mask"]
        if mL.sum() < 3 or mC.sum() < 3:
            return T0, dict(success=False, reason="insufficient_support")

        p, P = self.p_body[mL], self.p_world[mL]
        wL = self.w_L[mL]
        uv, X = self.uv[mC], self.xyz_world[mC]
        wC = self.w_C[mC]
        K, E = self.K, self.T_BC

        def fun(x):
            T = _increment(np.asarray(T0, dtype=float), x, cfg)
            rL = (p @ T[:3, :3].T + T[:3, 3] - P) / cfg.lidar_scale_m
            sL = np.einsum('ij,ij->i', rL, rL)
            fL = rL * np.sqrt((wL * np.log1p(sL) / np.maximum(sL, 1e-18)))[:, None]
            T_WC = T @ E
            q = (X - T_WC[:3, 3]) @ T_WC[:3, :3]
            proj = q @ K.T
            uv_proj = proj[:, :2] / proj[:, 2:]
            rC = (uv_proj - uv) / cfg.camera_scale_px
            sC = np.einsum('ij,ij->i', rC, rC)
            fC = rC * np.sqrt((wC * np.log1p(sC) / np.maximum(sC, 1e-18)))[:, None]
            return np.concatenate((fL.ravel(), fC.ravel()))

        fit = least_squares(fun, np.zeros(6), method="lm", x_scale=1.0,
                            max_nfev=cfg.refine_max_nfev,
                            ftol=cfg.refine_tolerance, xtol=cfg.refine_tolerance,
                            gtol=cfg.refine_tolerance)
        return _increment(np.asarray(T0, dtype=float), fit.x, cfg), \
            dict(success=bool(fit.success), nfev=int(fit.nfev))

    # ---- observability ---------------------------------------------------
    def observability(self, T):
        cfg = self.cfg
        sup = self.support(T)
        mL, mC = sup["lidar_inlier_mask"], sup["camera_inlier_mask"]
        rL, rC, _ = self.residuals(T)
        sL = np.einsum('ij,ij->i', rL, rL)[mL]
        sC = np.einsum('ij,ij->i', rC, rC)[mC]

        def residual(x):
            rLx, rCx, _ = self.residuals(_increment(np.asarray(T, dtype=float), x, cfg))
            return np.concatenate((rLx[mL].ravel(), rCx[mC].ravel()))

        wL = 0.5 * self.w_L[mL] / (1.0 + sL)
        wC = 0.5 * self.w_C[mC] / (1.0 + sC)
        weights = np.concatenate((np.repeat(wL, 3), np.repeat(wC, 2)))
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


def _nondegenerate(points, cfg):
    pts = np.asarray(points, dtype=float)
    s = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
    return s[0] > 1e-9 and (len(s) < 2 or s[1] / s[0] > 1e-4)


def p3p_candidates(uv, xyz_world, K, T_BC, cfg, rng):
    """Region-diverse P3P sampling from the full camera correspondence set.

    Each sample takes 3 correspondences from 3 distinct image regions; every
    finite, legal solution of cv2.solveP3P (world->camera) is converted with
    H = inv(T_CW) @ inv(E) and returned as a T_WB candidate."""
    import cv2

    uv = np.asarray(uv, dtype=float)
    X = np.asarray(xyz_world, dtype=float)
    good = np.isfinite(uv).all(axis=1) & np.isfinite(X).all(axis=1)
    uv, X = uv[good], X[good]
    if len(uv) < 3:
        return [], dict(p3p_samples=0, p3p_solutions=0, p3p_failures=0)
    K = np.asarray(K, dtype=float)
    inverse_E = np.linalg.inv(validate_pose(T_BC, cfg.pose_validation_tolerance))
    grid = cfg.camera_grid
    h_cells = np.clip((uv[:, 1] / max(uv[:, 1].max(), 1e-9) * grid).astype(int), 0, grid - 1)
    w_cells = np.clip((uv[:, 0] / max(uv[:, 0].max(), 1e-9) * grid).astype(int), 0, grid - 1)
    cells = h_cells * grid + w_cells
    by_cell = [np.flatnonzero(cells == k) for k in range(grid * grid)]
    occupied = [idx for idx in by_cell if len(idx) >= 3]

    candidates, failures, solutions = [], 0, 0
    if not occupied:
        return [], dict(p3p_samples=0, p3p_solutions=0, p3p_failures=0)
    for _ in range(cfg.camera_sample_budget):
        cell_ids = rng.choice(len(occupied), size=min(3, len(occupied)), replace=False)
        idx = np.concatenate([rng.choice(occupied[c], 1, replace=False) for c in cell_ids])
        if len(idx) < 3:
            continue
        if not (_nondegenerate(uv[idx], cfg) and _nondegenerate(X[idx], cfg)):
            continue
        try:
            ok, rvecs, tvecs = cv2.solveP3P(
                np.ascontiguousarray(X[idx]), np.ascontiguousarray(uv[idx]), K, None,
                flags=cv2.SOLVEPNP_P3P)
        except cv2.error:
            failures += 1
            continue
        if not ok:
            failures += 1
            continue
        for rv, tv in zip(rvecs, tvecs):
            T_CW = np.eye(4)
            T_CW[:3, :3] = cv2.Rodrigues(rv)[0]
            T_CW[:3, 3] = tv.ravel()
            candidates.append(np.linalg.inv(T_CW) @ inverse_E)
            solutions += 1
    return candidates, dict(p3p_samples=cfg.camera_sample_budget, p3p_solutions=solutions,
                            p3p_failures=failures)


def _clusters(scored, cfg):
    kept = []
    for T, source, sc in sorted(scored, key=lambda h: h[2]["score"]):
        if not any(_near(T, other[0], cfg.cluster_dt_m, cfg.cluster_dR_rad) for other in kept):
            kept.append((T, source, sc))
    return kept


def _near(A, B, dt, dr):
    t, r = pose_distance(A, B)
    return t <= dt and r <= dr


def _passes_joint_support(sc, sup, cfg):
    return (sup["n_lidar"] >= cfg.min_lidar_inliers
            and sup["n_camera"] >= cfg.min_camera_inliers
            and sup["lidar_ratio"] >= cfg.min_lidar_ratio
            and sup["camera_ratio"] >= cfg.min_camera_ratio
            and sc["score"] <= cfg.max_score)


def _passes_single_modal(sc, sup, modality, cfg):
    if modality == "LIDAR":
        return (sup["n_lidar"] >= cfg.single_modal_min_inliers
                and sup["lidar_ratio"] >= cfg.single_modal_min_ratio
                and sc["lidar_score"] <= cfg.single_modal_max_score)
    return (sup["n_camera"] >= cfg.single_modal_min_inliers
            and sup["camera_ratio"] >= cfg.single_modal_min_ratio
            and sc["camera_score"] <= cfg.single_modal_max_score)


def solve(problem: JointProblem, T_L=None, T_C=None, seedwise_T_WB=None, *,
          config: Optional[JointSolverConfig] = None,
          mode: str = "joint_refine") -> JointResult:
    """Shared solver. mode: 'select' (per-modality scoring, pick winner),
    'joint' (joint scoring, no refinement), 'joint_refine' (full)."""
    started = perf_counter()
    cfg = config if config is not None else problem.cfg
    diagnostics = {"pose_convention": "T_WB", "mode": mode}
    rng = np.random.default_rng(cfg.seed)

    def finish(pose, status, source=None, modality=None, reason=None):
        diagnostics["elapsed_s"] = perf_counter() - started
        return JointResult(None if pose is None else np.asarray(pose, dtype=float).copy(),
                           status, source, modality, reason, diagnostics)

    originals = []
    if T_L is not None:
        originals.append((validate_pose(T_L, cfg.pose_validation_tolerance), "LIDAR_ORIGINAL"))
    if T_C is not None:
        originals.append((validate_pose(T_C, cfg.pose_validation_tolerance), "CAMERA_ORIGINAL"))
    extras = []
    if seedwise_T_WB is not None:
        for k, T in enumerate(np.asarray(seedwise_T_WB, dtype=float)):
            try:
                extras.append((validate_pose(T, cfg.pose_validation_tolerance),
                               f"SC2_SEED_{k}"))
            except ValueError:
                continue

    p3p, p3p_stats = p3p_candidates(problem.uv, problem.xyz_world, problem.K,
                                    problem.T_BC, cfg, rng)
    diagnostics.update(p3p_stats)
    diagnostics["n_seedwise"] = len(extras)

    pool = originals + extras + [(T, "CAMERA_P3P") for T in p3p]
    if not pool:
        return finish(None, "REJECTED", reason="NO_CANDIDATES")

    scored = []
    for T, source in pool:
        try:
            sc = problem.score(T)
            sup = problem.support(T)
            scored.append((T, source, sc, sup))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
    diagnostics["n_scored_candidates"] = len(scored)
    if not scored:
        return finish(None, "REJECTED", reason="NO_VALID_CANDIDATES")

    if mode == "select":
        # Per-modality scoring: each candidate is judged only on its own modality.
        best = None
        counts = {"LIDAR": 0, "CAMERA": 0}
        for T, source, sc, sup in scored:
            if source.startswith("LIDAR") or source.startswith("SC2"):
                own = dict(score=sc["lidar_score"])
                ok = (sup["n_lidar"] >= cfg.min_lidar_inliers
                      and sup["lidar_ratio"] >= cfg.min_lidar_ratio)
                modality = "LIDAR"
            else:
                own = dict(score=sc["camera_score"])
                ok = (sup["n_camera"] >= cfg.min_camera_inliers
                      and sup["camera_ratio"] >= cfg.min_camera_ratio)
                modality = "CAMERA"
            if ok:
                counts[modality] += 1
                if best is None or own["score"] < best[2]["score"]:
                    best = (T, source, own, sup, modality)
        diagnostics["select_supported"] = counts
        if best is None:
            return finish(None, "REJECTED", reason="INSUFFICIENT_SUPPORT")
        T, source, _, sup, modality = best
        ok, spectrum = problem.observability(T)
        diagnostics["observability"] = spectrum
        if not ok:
            return finish(None, "DEGENERATE", source=source, modality=modality)
        return finish(T, "JOINT" if counts["LIDAR"] and counts["CAMERA"] else "SINGLE_MODAL",
                      source=source, modality=None if counts["LIDAR"] and counts["CAMERA"] else modality)

    clusters = _clusters([(T, s, sc) for T, s, sc, _ in scored], cfg)
    diagnostics["n_clusters"] = len(clusters)

    if mode == "joint_refine":
        seeds = clusters[:cfg.retain_candidates]
        diagnostics["refinements"] = []
        refined = []
        for T, source, _ in seeds:
            try:
                T_ref, info = problem.refine(T)
                info["source"] = source
                diagnostics["refinements"].append(info)
                sc = problem.score(T_ref)
                sup = problem.support(T_ref)
                refined.append((T_ref, source + "_REFINED", sc, sup))
                # keep the original candidate too
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                diagnostics["refinements"].append(dict(source=source, success=False,
                                                       error=str(exc)))
        all_scored = scored + refined
        clusters = _clusters([(T, s, sc) for T, s, sc, _ in all_scored], cfg)
        diagnostics["n_clusters_after_refine"] = len(clusters)
        support_map = {}
        for T, s, sc, sup in all_scored:
            support_map[id(T)] = sup
    else:
        support_map = {id(T): sup for T, s, sc, sup in scored}

    def sup_of(cluster):
        T = cluster[0]
        if id(T) in support_map:
            return support_map[id(T)]
        return problem.support(T)

    joint_supported = [c for c in clusters
                       if _passes_joint_support(c[2], sup_of(c), cfg)]
    diagnostics["n_joint_supported"] = len(joint_supported)
    if joint_supported:
        if len(joint_supported) > 1:
            margin = joint_supported[1][2]["score"] - joint_supported[0][2]["score"]
            diagnostics["second_score_margin"] = float(margin)
            if margin < cfg.min_score_separation:
                return finish(None, "AMBIGUOUS", reason="CLOSE_COMPETING_HYPOTHESES")
        else:
            diagnostics["second_score_margin"] = None
        T, source, sc = joint_supported[0]
        diagnostics["best_support"] = {k: v for k, v in sup_of(joint_supported[0]).items()
                                       if not k.endswith("mask")}
        diagnostics["best_score"] = sc
        ok, spectrum = problem.observability(T)
        diagnostics["observability"] = spectrum
        if not ok:
            return finish(None, "DEGENERATE", source=source)
        return finish(T, "JOINT", source=source)

    # No jointly supported candidate: per-modality degraded output decision.
    lidar_ok = [c for c in clusters
                if _passes_single_modal(c[2], sup_of(c), "LIDAR", cfg)]
    camera_ok = [c for c in clusters
                 if _passes_single_modal(c[2], sup_of(c), "CAMERA", cfg)]
    diagnostics["single_modal_lidar_supported"] = len(lidar_ok)
    diagnostics["single_modal_camera_supported"] = len(camera_ok)
    unique_lidar = len(lidar_ok) == 1 or (
        len(lidar_ok) > 1
        and (lidar_ok[1][2]["lidar_score"] - lidar_ok[0][2]["lidar_score"])
        >= cfg.single_modal_min_separation)
    unique_camera = len(camera_ok) == 1 or (
        len(camera_ok) > 1
        and (camera_ok[1][2]["camera_score"] - camera_ok[0][2]["camera_score"])
        >= cfg.single_modal_min_separation)
    if lidar_ok and unique_lidar and not camera_ok:
        T, source, sc = lidar_ok[0]
        ok, spectrum = problem.observability(T)
        diagnostics["observability"] = spectrum
        if not ok:
            return finish(None, "DEGENERATE", source=source, modality="LIDAR")
        return finish(T, "SINGLE_MODAL", source=source, modality="LIDAR")
    if camera_ok and unique_camera and not lidar_ok:
        T, source, sc = camera_ok[0]
        ok, spectrum = problem.observability(T)
        diagnostics["observability"] = spectrum
        if not ok:
            return finish(None, "DEGENERATE", source=source, modality="CAMERA")
        return finish(T, "SINGLE_MODAL", source=source, modality="CAMERA")
    return finish(None, "REJECTED", reason="INSUFFICIENT_SUPPORT")
