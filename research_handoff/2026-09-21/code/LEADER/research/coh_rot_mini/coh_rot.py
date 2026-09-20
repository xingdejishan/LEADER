"""L_coh-rot: frame-level coherent-rotation loss (drop-in next to official TRR).

For each frame, on the same point set used by TRR:

    X_i = c_hat_i - sum_j w_j c_hat_j        (weighted de-centering of the prediction)
    Y_i = c_i     - sum_j w_j c_j            (weighted de-centering of the GT)
    Q_e = argmin_{Q in SO(3)} sum_i w_i || X_i Q - Y_i ||^2      (weighted Kabsch / SVD)
    z   = ||Q_e - I||_F^2 / (2 * theta0^2)
    L   = sqrt(1 + z) - 1

Q_e is NOT the vehicle pose: it is "how much the predicted coordinate field as a whole
must still be rotated to line up with GT".  Pure whole-frame translation triggers nothing
(de-centering); a rigidly consistent but wrongly-oriented field is penalised.

Weights w are the *frozen* model's reliability conversion, computed once from the same
cache and fixed for the entire run, so the network cannot re-weight its way around the
points that need fixing.  w_raw = exp(atan(u) * log(scale)/pi), normalised per frame.

Robustness (all required, per protocol):
  * skipped for the new term (TRR still applies) when: too few points, rank < 2,
    near-repeated singular values, non-finite coordinates, or weight over-concentration.
  * angle gate: frames whose Kabsch angle > gate_deg (default 20) skip the new term.
  * the backward pass never differentiates the SVD singular vectors.  Gradients flow
    through the first-order linearised rotation
        omega = A^{-1} b,   A = sum_i w_i (|x_i|^2 I - x_i x_i^T),  b = sum_i w_i (x_i x y_i)
    which is the Kabsch solution to first order (exact for small angles, the regime the
    loss targets).  The exact SVD angle is used only for gating and logging.
  * every skipped frame is counted and reported; skipped frames still enter evaluation.

Returned stats: per-frame SVD angle, linear omega norm, valid/skip flags.
"""

import math

import numpy as np
import torch

__all__ = ["weights_from_scores", "coherent_rotation_loss", "kabsch_forward", "linear_omega"]


def weights_from_scores(scores: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    """TRR-style reliability conversion -- un-normalised weights (per frame)."""
    scaler = math.log(scale) / math.pi
    return torch.exp(scores.clamp(min=-10 * math.pi, max=10 * math.pi).atan() * scaler)


def kabsch_forward(X: torch.Tensor, Y: torch.Tensor, w: torch.Tensor, detach: bool = True):
    """Weighted Kabsch. X,Y: (n,3); w: (n,) sums to 1. Returns Q (3,3) with X @ Q ~ Y."""
    if detach:
        with torch.no_grad():
            return _kabsch(X, Y, w)
    return _kabsch(X, Y, w)


def _kabsch(X, Y, w):
    H = (X * w[:, None]).T @ Y                      # (3,3)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.T @ U.T).sign()
    D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    Q = Vt.T @ D @ U.T
    return Q, S


def linear_omega(X: torch.Tensor, Y: torch.Tensor, w: torch.Tensor, ridge: float = 1e-9):
    """First-order rotation omega solving A omega = b (differentiable, no SVD)."""
    eye = torch.eye(3, dtype=X.dtype, device=X.device)
    xx = torch.einsum("ni,nj->nij", X, X)
    A = (w[:, None, None] * ((X ** 2).sum(1)[:, None, None] * eye - xx)).sum(0)
    b = (w[:, None] * torch.cross(X, Y, dim=1)).sum(0)
    scale = A.diagonal().mean().clamp_min(1e-12)
    A = A + ridge * scale * eye
    return torch.linalg.solve(A, b)


def coherent_rotation_loss(
    pred_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    weights: torch.Tensor,
    batch_idx: torch.Tensor,
    theta0_deg: float = 1.0,
    gate_deg: float = 20.0,
    min_points: int = 16,
    concentration_max: float = 5.0,
):
    """Frame-wise coherent-rotation loss.

    pred_xyz, target_xyz: (N,3) same point set, same world frame.
    weights: (N,) fixed reliability weights (un-normalised; from the frozen model).
    batch_idx: (N,) frame index (one frame per index; frames are never merged).

    Returns (loss, stats). loss is the mean over *valid* frames (0 if none valid).
    """
    device = pred_xyz.device
    n_frames = int(batch_idx.max().item()) + 1
    theta0 = math.radians(theta0_deg)

    total = pred_xyz.new_zeros(())
    valid = 0
    skipped_gate = 0
    skipped_degenerate = 0
    angles = []
    omegas = []

    for f in range(n_frames):
        m = batch_idx == f
        n = int(m.sum().item())
        if n < min_points:
            skipped_degenerate += 1
            continue
        X = pred_xyz[m]
        Y = target_xyz[m]
        w = weights[m]
        w = w / w.sum().clamp_min(1e-12)

        # weighted de-centering (kills whole-frame translation)
        Xc = X - (w[:, None] * X).sum(0, keepdim=True)
        Yc = Y - (w[:, None] * Y).sum(0, keepdim=True)

        # forward: exact Kabsch for gating/logging (no grad)
        Q, S = kabsch_forward(Xc.detach(), Yc.detach(), w.detach())
        cos = ((Q.diagonal().sum() - 1) / 2).clamp(-1, 1)
        angle = torch.rad2deg(torch.acos(cos)).item()

        # degeneracy checks
        scale = float(S[0].clamp_min(1e-12))
        rank_ok = bool(S[1] > scale * 1e-7) and bool(((S[:-1] - S[1:]).abs() > scale * 1e-7).all())
        conc = float((w ** 2).sum()) * n                      # 1 = uniform, n = one point
        finite = bool(torch.isfinite(X).all() and torch.isfinite(Y).all())
        if (not rank_ok) or conc > concentration_max or (not finite):
            skipped_degenerate += 1
            continue
        if angle > gate_deg:
            skipped_gate += 1
            continue

        # differentiable term through the linearised omega (no SVD backward)
        omega = linear_omega(Xc, Yc, w)
        z = (omega ** 2).sum() / (theta0 ** 2)
        L = torch.sqrt(1.0 + z) - 1.0
        if not torch.isfinite(L):
            skipped_degenerate += 1
            continue
        total = total + L
        valid += 1
        angles.append(angle)
        omegas.append(float(torch.linalg.norm(omega.detach())))

    if valid == 0:
        loss = pred_xyz.new_zeros(())
    else:
        loss = total / valid
    stats = dict(
        valid=valid,
        skipped_gate=skipped_gate,
        skipped_degenerate=skipped_degenerate,
        angle_mean=(float(np.mean(angles)) if angles else None),
        angle_max=(float(np.max(angles)) if angles else None),
        omega_mean_deg=(float(np.degrees(np.mean(omegas))) if omegas else None),
    )
    return loss, stats
